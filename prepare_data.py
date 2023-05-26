from pathlib import Path
from argparse import ArgumentParser
import logging

from multiprocessing import cpu_count
from functools import partial
from collections import Counter

import pandas as pd

from rdkit import RDLogger

import src.utils as ut
from src.preprocessing import ReactionPreprocessingPipeline, HeuristicRoleClassifier
from src.tokenizer import smi_tokenizer
from src.augmenter import augment_rxn
from src.pysmilesutils.pysmilesutils_augmenter import SMILESAugmenter


def main(args):
    """
    Makes input files with tokenized reactions for OpenNMT.
    Takes raw reactions, like USPTO SMILES, as input.
    :param args: Command line arguments. For more information, run 'python3 prepare_data.py --help'
    :return:
    """
    filepath = Path(args.filepath)
    logging.info("Reading data from %s" % filepath)
    data = pd.read_csv(filepath, sep=args.separator)
    logging.info("Number of entries in data: %d" % data.shape[0])

    stages = [
        ut.drop_cxsmiles_info,
        ut.mix_reagents,
        ut.reassign_reaction_roles,
        ut.canonical_remove_aam_rxn,
        ut.drop_isotopes,
        ut.assemble_ions
    ]

    pipeline = ReactionPreprocessingPipeline(stages)

    # === 1. Processing reactions in a predefined pipeline ===
    logging.debug("Processing reactions, assigning reaction roles... Number of processes: %d" % args.n_jobs)
    data["ProcessedReaction"] = ut.parallelize_on_rows(data[args.source_column], pipeline.run, args.n_jobs)
    assert len(data[data.ProcessedReaction.str.startswith("!")]) == 0

    # === Removing reactions with more than 10 unique molecules on the left side of a reaction
    long_reactions = data[data["ProcessedReaction"].apply(lambda x: len(set(x.split(">>")[0].split("."))) > 10)]
    logging.info(f"Removing {len(long_reactions)} too long reactions")
    data.drop(long_reactions.index, inplace=True)
    # ===

    roles = data["ProcessedReaction"].str.split(">", expand=True)
    roles.columns = ["Reactants", "Reagents", "Products"]

    # === 2. Getting reagent statistics ===
    reagent_statistics_counter = ut.get_reagent_statistics(roles["Reagents"])

    # === 2.1. (Optionally) Removing rare reagents from reaction SMILES
    min_rep = args.min_reagent_occurances
    if min_rep is not None:
        logging.info("Removing rare reagents from data")
        n_rxns_without_reagents = len(roles[roles['Reagents'] == ''])
        repeated_reagents_counter = Counter({k: v for k, v in reagent_statistics_counter.items() if v >= min_rep})
        rare_reagents = {k for k, v in reagent_statistics_counter.items() if v < min_rep}
        logging.info("Num of unique reagents: %d" % len(reagent_statistics_counter))

        logging.info("Num of reagents repeated "
                     "more than %d times: %d (%.3f%%)" % (min_rep,
                                                          len(repeated_reagents_counter),
                                                          len(repeated_reagents_counter) * 100 / len(
                                                              reagent_statistics_counter)))

        reagents_encountered_once = {k for k, v in reagent_statistics_counter.items() if v == 1}
        logging.info("Num of reagents encountered only once: %d (%.3f%%)" % (len(reagents_encountered_once),
                                                                             len(reagents_encountered_once) * 100 / len(
                                                                                 reagent_statistics_counter)))

        roles["Reagents"] = roles["Reagents"].apply(
            lambda x: ".".join([m for m in x.split(".") if m not in rare_reagents]))
        logging.info("After removing rare reagents %d more reactions lost all reagents. (There was %d originally)" % (
            len(roles[roles['Reagents'] == '']) - n_rxns_without_reagents,
            n_rxns_without_reagents))

    # === 3. Dropping invalid entries ===

    data["ProcessedReaction"] = roles["Reactants"] + ">" + roles["Reagents"] + ">" + roles["Products"]

    data = pd.concat((data, roles), axis=1)

    no_reagents_data = data[roles.Reagents == '']
    logging.info("Dropping reactions without reagents: %d" % len(no_reagents_data))
    data = data.drop(no_reagents_data.index)
    logging.info("Reactions left: %d" % len(data))

    logging.info("Dropping duplicate reactions")
    logging.info("Reactions before: %d" % data.shape[0])
    data.drop_duplicates(subset=['ProcessedReaction'], inplace=True)
    logging.info("Reactions after: %d" % data.shape[0])

    logging.info("Dropping reactions where product appears among reactants or reagents")
    logging.info("Reactions before: %d" % data.shape[0])

    data = data.drop(
        data.where(data[["Reactants", "Reagents", "Products"]].apply(
            lambda x: (len(set(x[0].split(".")).intersection(set(x[2].split(".")))) > 0) or (
                    len(set(x[1].split(".")).intersection(set(x[2].split(".")))) > 0), axis=1)).dropna().index)
    logging.info("Reactions after: %d" % data.shape[0])

    # === 4. Rearranging reagents according to role priorities
    roles = data["ProcessedReaction"].str.split(">", expand=True)
    roles.columns = ["Reactants", "Reagents", "Products"]
    data.drop(["Reactants", "Reagents", "Products"], axis=1, inplace=True)
    logging.info("Rearranging reagents according to their role priorities")
    roles["Reagents"] = roles["Reagents"].apply(HeuristicRoleClassifier.rearrange_reagents)
    data["ProcessedReaction"] = roles["Reactants"] + ">" + roles["Reagents"] + ">" + roles["Products"]

    # === 4. Separating the training set from the validation set
    data = data.sample(frac=1).reset_index(drop=True)
    if args.val_size != 0:
        data_train = data.iloc[:-args.val_size, :]
        data_val = data.iloc[-args.val_size:, :]
    else:
        data_train = data
        data_val = None

    # === 5. (Optional) Augmenting the training set ===

    if args.use_augmentations:
        smiles_randomizer = SMILESAugmenter(restricted=True)
        logging.info("Augmenting reactions...")
        data_train["ProcessedReaction"] = ut.parallelize_on_rows(data_train["ProcessedReaction"],
                                                                 partial(augment_rxn,
                                                                         smiles_randomizer,
                                                                         args.use_role_augmentations),
                                                                 num_of_processes=args.n_jobs)
        data_train = data_train.explode("ProcessedReaction")
        data_train.reset_index(drop=True, inplace=True)

    # === 7. Preparing and saving files for OpenNMT ===
    save_path = Path("data/tokenized").resolve() / args.output_dir
    save_path.mkdir(parents=True, exist_ok=True)

    data_train_roles = data_train["ProcessedReaction"].str.split(">", expand=True)
    data_train_roles.columns = ["Reactants", "Reagents", "Products"]
    data_train["domain"] = data_train_roles["Reactants"] + ">>" + data_train_roles["Products"]
    data_train["target"] = data_train_roles["Reagents"]
    save_path_src_tokd = (save_path / "src-train").with_suffix(".txt")
    data_train["domain"].apply(smi_tokenizer).to_csv(
        save_path_src_tokd,
        index=False,
        header=False,
    )
    save_path_tgt_tokd = (save_path / "tgt-train").with_suffix(".txt")
    data_train["target"].apply(smi_tokenizer).to_csv(
        save_path_tgt_tokd,
        index=False,
        header=False,
    )
    logging.info("OpenNMT input train source saved in %s" % save_path_src_tokd)
    logging.info("OpenNMT input train target saved in %s" % save_path_tgt_tokd)

    if data_val is not None:
        data_val_roles = data_val["ProcessedReaction"].str.split(">", expand=True)
        data_val_roles.columns = ["Reactants", "Reagents", "Products"]
        data_val["domain"] = data_val_roles["Reactants"] + ">>" + data_val_roles["Products"]
        data_val["target"] = data_val_roles["Reagents"]
        save_path_src_tokd = (save_path / "src-val").with_suffix(".txt")
        data_val["domain"].apply(smi_tokenizer).to_csv(
            save_path_src_tokd,
            index=False,
            header=False,
        )
        save_path_tgt_tokd = (save_path / "tgt-val").with_suffix(".txt")
        data_val["target"].apply(smi_tokenizer).to_csv(
            save_path_tgt_tokd,
            index=False,
            header=False,
        )
        logging.info("OpenNMT input val source saved in %s" % save_path_src_tokd)
        logging.info("OpenNMT input val target saved in %s" % save_path_tgt_tokd)


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    logging.basicConfig(filename='prepare_data.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s')

    parser = ArgumentParser()
    group_input = parser.add_argument_group("Input settings")
    group_input.add_argument("--filepath", type=str,
                             help="Path to the raw data (.csv) that needs preprocessing.")
    group_input.add_argument("--output_dir", type=str, default="", help="Name of the directory with tokenized files.")
    group_input.add_argument("--source_column", type=str,
                             help="Name of the column in the input that needs preprocessing.",
                             default="OriginalReaction")
    group_input.add_argument("--separator", type=str, default="\t",
                             help="Separator in the input .csv file.")
    group_input.add_argument("--n_jobs", type=int, default=cpu_count(),
                             help="Number of processes to use in parallelized functions.")
    group_input.add_argument("--val_size", type=int, default=0,
                             help="Size of the validation set that will be separated from the input dataset")

    group_prep = parser.add_argument_group("Preprocessing flags")
    group_prep.add_argument("--use_augmentations", action="store_true",
                            help="Whether to augment reactions using SMILES augmentations.")
    group_prep.add_argument("--use_role_augmentations", action="store_true",
                            help="Whether to augment reaction SMILES by moving random reagents to the reactant side. "
                                 "Only makes sense if --use_augmentations is used.")
    group_prep.add_argument("--min_reagent_occurances", type=int, default=None,
                            help="If not None, all reagent with number of occurrences less than this "
                                 "number will be removed.")

    main(parser.parse_args())
