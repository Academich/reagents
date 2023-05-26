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

    # === 4. (Optional) Augmentations ===
    if args.use_augmentations:
        smiles_randomizer = SMILESAugmenter(restricted=True)
        logging.info("Augmenting reactions...")
        data["ProcessedReaction"] = ut.parallelize_on_rows(data["ProcessedReaction"],
                                                           partial(augment_rxn,
                                                                   smiles_randomizer,
                                                                   args.use_role_augmentations),
                                                           num_of_processes=args.cpu_count)
        data = data.explode("ProcessedReaction")
        data.reset_index(drop=True, inplace=True)

    roles = data["ProcessedReaction"].str.split(">", expand=True)
    roles.columns = ["Reactants", "Reagents", "Products"]
    data.drop(["Reactants", "Reagents", "Products"], axis=1, inplace=True)
    reagents = roles["Reagents"]

    # === 5. Rearranging reagents according to role priorities
    logging.info("Rearranging reagents according to their detailed roles.")
    reagents = reagents.apply(HeuristicRoleClassifier.rearrange_reagents)

    # === 6. Separating solvents from the rest of the reagents ===
    logging.info("Separating solvents from the other reagents")
    solvents_smiles = set(SOLVENTS)

    reagents = reagents.apply(partial(ut.separate_solvents, solvents_smiles)).str.split("&", expand=True)
    reagents.columns = ["Reagents", "Solvents"]
    roles["Reagents"] = reagents["Reagents"]
    roles = pd.concat((roles, reagents["Solvents"]), axis=1)

    # === 5. (Optional) Deciding on common reagents that should become individual tokens in the target decoder ===
    single_token_reagents = []
    molecule_length_threshold = 10
    if args.use_special_tokens:

        # Here those are some common ligands and catalysts

        if args.train:
            for r in most_common_reagents:
                long_phosph_comp = len(r) > molecule_length_threshold and "P" in r
                if long_phosph_comp:
                    single_token_reagents.append(r)

            if len(single_token_reagents) > 0:
                logging.info("Common reagents becoming individual tokens:\n %s", ", ".join(single_token_reagents))

    # === 6. Deriving and saving tokenizer vocabulary ===
    data = pd.concat((data, roles), axis=1)

    domain = data[["Reactants", "Products"]].apply(lambda x: x[0] + ">>" + x[1], axis=1)
    target = data[["Reagents", "Solvents"]].apply(lambda x: x[0] + "." * (len(x[0]) > 0 and len(x[1]) > 0) + x[1],
                                                  axis=1)

    save_path_intermediate_vocab = Path("data/vocabs").resolve()
    save_path_src_vocab = (save_path_intermediate_vocab / (args.output_suffix + "_src_vocab")).with_suffix(".json")
    save_path_tgt_vocab = (save_path_intermediate_vocab / (args.output_suffix + "_tgt_vocab")).with_suffix(".json")

    if args.train:
        logging.info("Deriving tokenization vocabulary...")
        t_source = ChemSMILESTokenizer.based_on_smiles(
            domain.values.flatten().tolist() + data["Reagents"].values.flatten().tolist())
        t_target = ChemSMILESTokenizer.based_on_smiles(
            domain.values.flatten().tolist() + data["Reagents"].values.flatten().tolist())
        if len(single_token_reagents) > 0:
            t_target.add_tokens(single_token_reagents, regex=False)
            t_target.vocabulary.update({s: i + len(t_target.vocabulary) for i, s in enumerate(single_token_reagents)})

        logging.info("Saving source tokenizer vocabulary to %s" % save_path_src_vocab)
        t_source.save_vocabulary(save_path_src_vocab)
        logging.info("Saving target tokenizer vocabulary to %s" % save_path_tgt_vocab)
        t_target.save_vocabulary(save_path_tgt_vocab)
    else:
        logging.info("Loading tokenizer vocabularies")
        t_source = ChemSMILESTokenizer()
        t_source.load_vocabulary(save_path_src_vocab)

        t_target = ChemSMILESTokenizer()
        t_target.load_vocabulary(save_path_tgt_vocab)
        long_tokens = [k for k, v in t_target.vocabulary.items() if len(k) > molecule_length_threshold]
        t_target.add_tokens(long_tokens, regex=False)

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
    group_input.add_argument("--output_suffix", type=str, default="", help="Additional suffix to the output files.")
    group_input.add_argument("--train", action="store_true",
                             help="If true, derive and save tokenizer vocabulary from the processed data.")
    group_input.add_argument("--source_column", type=str,
                             help="Name of the column in the input that needs preprocessing.",
                             default="OriginalReaction")
    group_input.add_argument("--separator", type=str, default="\t",
                             help="Separator in the input .csv file.")
    group_input.add_argument("--n_jobs", type=int, default=cpu_count(),
                             help="Number of processes to use in parallelized functions.")

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
