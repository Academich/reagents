import shutil
from pathlib import Path
from multiprocessing import cpu_count
from argparse import ArgumentParser
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from rdkit import RDLogger

import utils as ut
from prediction import MolecularTransformerReagentPredictor
from tokenizer import smi_tokenizer

RDLogger.DisableLog('rdApp.*')

np.random.seed(123456)


def num_mols(smi: str) -> int:
    return len(smi.split('.'))


def shuffle_molecules(smi: str) -> str:
    """
    Randomly changes the order of molecules in a sequence of molecules
    :param smi: SMILES of molecules separated by dots
    :return: SMILES with the altered sequence of molecules
    """
    bag = smi.split('.')
    np.random.shuffle(bag)
    return '.'.join(bag)


def drop_reagents(smi: str) -> str:
    left, _, right = smi.split(">")
    return left + ">>" + right


def extract_reagents(smi: str) -> str:
    _, center, _ = smi.split(">")
    return center


def get_files_for_forward_prediction(path: Path,
                                     subset: str,
                                     reag_predictor: 'MolecularTransformerReagentPredictor',
                                     mixed: bool = False) -> None:
    """
    Replaces reagents in the tokenized files for product prediction according to different strategies.
    :param path: Path to the directory with tokenized files for OpenNMT
    :param subset: 'train' or 'val'
    :param reag_predictor: A reagent predictor instance. Requires a trained reagents prediction model.
    :return:
    """
    with open(path / f"src-{subset}.txt") as f, open(path / f"tgt-{subset}.txt") as h:
        data_src = pd.DataFrame([i.strip().replace(" ", "") for i in f.readlines()])
        data_tgt = pd.DataFrame([i.strip().replace(" ", "") for i in h.readlines()])

    # Assemble full reactions
    logging.info(f"Mixed setting: {mixed}")
    arrow = ">>" if mixed else ">"
    data = data_src + arrow + data_tgt
    data.columns = ["rxn_original"]

    # Determine reagents by RDKit
    logging.info("Detecting reagents using RDKit...")
    data["rxn_reagents_rdkit"] = ut.parallelize_on_rows(data["rxn_original"],
                                                        ut.reassign_reaction_roles,
                                                        cpu_count(),
                                                        use_tqdm=False)

    # If it left no reactants revert to the original reaction
    data.loc[data[data["rxn_reagents_rdkit"].str.startswith(">")].index, "rxn_reagents_rdkit"] = np.nan
    data["rxn_reagents_rdkit"].fillna(data["rxn_original"], inplace=True)

    # Drop reagents, leave only reactants>>product
    data["rxn_no_reagents"] = data["rxn_reagents_rdkit"].apply(drop_reagents)

    reactants = data["rxn_no_reagents"].str.split(">>", expand=True)[0]

    # Extract reagents
    data["rgs_rdkit"] = data["rxn_reagents_rdkit"].apply(extract_reagents)

    # Predict new reagents for all reactions with a trained model
    logging.info(f"Predicting new reagents using {reag_predictor.model_path}")
    reag_predictor.make_and_store_predictions(data["rxn_no_reagents"])
    reag_predictor.load_predictions()
    predicted_reagents = reag_predictor.predictions

    # ===============================================================================
    # === Strategy 0: Delete all reagents entirely
    # ===============================================================================
    logging.info("Strategy 0: Delete all reagents entirely")
    direct = Path(str(path) + "_no_reags/")
    direct.mkdir(parents=True, exist_ok=True)
    src_save_path = direct / f"src-{subset}.txt"
    tgt_save_path = direct / f"tgt-{subset}.txt"
    logging.info("Saving files...")
    logging.info(f"Source: {src_save_path.resolve()}")
    logging.info(f"Target: {tgt_save_path.resolve()}")
    with open(src_save_path, 'w') as h:
        h.write(
            "\n".join(reactants.apply(smi_tokenizer))
        )
    shutil.copy(path / f"tgt-{subset}.txt", tgt_save_path)

    # ===============================================================================
    # === Strategy 1: Replace reagents with top-1 reagent prediction in all cases
    # ===============================================================================
    logging.info("Strategy 1: Replace reagents with top-1 reagent prediction in all cases")
    data["rgs_top1"] = predicted_reagents["p_reagents_1"]
    if mixed:
        data["src_reagents_top1"] = (reactants + '.' + data["rgs_top1"]).str.strip('.').apply(shuffle_molecules)
    else:
        data["src_reagents_top1"] = reactants + '>' + data["rgs_top1"]

    direct = Path(str(path) + "_reags_top1/")
    direct.mkdir(parents=True, exist_ok=True)
    src_save_path = direct / f"src-{subset}.txt"
    tgt_save_path = direct / f"tgt-{subset}.txt"
    logging.info("Saving files...")
    logging.info(f"Source: {src_save_path.resolve()}")
    logging.info(f"Target: {tgt_save_path.resolve()}")
    with open(src_save_path, 'w') as h:
        h.write(
            "\n".join(data["src_reagents_top1"].apply(smi_tokenizer))
        )
    shutil.copy(path / f"tgt-{subset}.txt", tgt_save_path)

    # ===============================================================================
    # === Strategy 2: Replace reagents with top-1 reagent prediction if there are more
    # === molecules in the prediction and the prediction is valid
    # ===============================================================================
    logging.info(
        "Strategy 2: Replace reagents with top-1 valid reagent prediction if there are more molecules in the prediction"
    )
    data["rgs_top1_and_rdkit"] = np.nan

    data["src_reagents_top1_and_rdkit"] = np.nan

    needs_replacement = (data["rgs_rdkit"].apply(num_mols) < data["rgs_top1"].apply(num_mols)) & (
            data["rgs_top1"].apply(ut.canonicalize_smiles) != '')
    replace_idx = data[needs_replacement].index
    logging.info(
        f"Reactions altered in top1+rdkit strategy: {len(replace_idx)} ({(100 * len(replace_idx)) / len(data):.2f}%)")
    logging.debug("Examples of indexes of reactions with replaced reagents:")
    logging.debug(replace_idx[:100])
    leave_idx = data[~needs_replacement].index

    data.loc[leave_idx, "src_reagents_top1_and_rdkit"] = data_src.loc[leave_idx, 0]
    data.loc[replace_idx, "src_reagents_top1_and_rdkit"] = data.loc[replace_idx, "src_reagents_top1"]

    direct = Path(str(path) + "_reags_top1_and_rdkit/")
    direct.mkdir(parents=True, exist_ok=True)
    src_save_path = direct / f"src-{subset}.txt"
    tgt_save_path = direct / f"tgt-{subset}.txt"
    logging.info("Saving files...")
    logging.info(f"Source: {src_save_path.resolve()}")
    logging.info(f"Target: {tgt_save_path.resolve()}")
    with open(src_save_path, 'w') as h:
        h.write(
            "\n".join(data["src_reagents_top1_and_rdkit"].apply(smi_tokenizer))
        )
    shutil.copy(path / f"tgt-{subset}.txt", tgt_save_path)
    # ===============================================================================
    # === Strategy 3: Replace reagents with predicted strings in which the reagents
    # === in all roles are those that are repeated the most across all predictions
    # === from top-1 to top-5
    # ===============================================================================
    logging.info(
        "Strategy 3: Role voting"
    )
    data["rgs_role_voting"] = reag_predictor.suggestions_by_roles(n_catal=1, n_solv=1, n_redox=1, n_unspec=2)

    data["rgs_role_voting"] = data["rgs_role_voting"].apply(lambda x: ".".join([i for i in x if i]))

    if mixed:
        data["src_reagents_role_voting"] = (reactants + '.' + data["rgs_role_voting"]).str.strip(
            '.').apply(shuffle_molecules)
    else:
        data["src_reagents_role_voting"] = reactants + '>' + data["rgs_role_voting"]

    direct = Path(str(path) + "_reags_role_voting/")
    direct.mkdir(parents=True, exist_ok=True)
    src_save_path = direct / f"src-{subset}.txt"
    tgt_save_path = direct / f"tgt-{subset}.txt"
    logging.info("Saving files...")
    logging.info(f"Source: {src_save_path.resolve()}")
    logging.info(f"Target: {tgt_save_path.resolve()}")
    with open(src_save_path, 'w') as h:
        h.write(
            "\n".join(data["src_reagents_role_voting"].apply(smi_tokenizer))
        )
    shutil.copy(path / f"tgt-{subset}.txt", tgt_save_path)


if __name__ == '__main__':
    logging.basicConfig(
        filename=ut.get_root_dir() / "logs" / f"reagent_substitution_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s')

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a directory with tokenized files for product prediction")
    parser.add_argument("--subset", type=str, required=True,
                        help="Name that specifies the data split, e.g. 'train' for 'src-train.txt' and 'tgt-train.txt'")
    parser.add_argument("--reagent_model", type=str,
                        help="Path to a trained reagents prediction model", required=True)
    parser.add_argument("--mixed_precursors", action="store_true",
                        help="Whether to mix reactants and reagents together in the generated files")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index, if not specified, CPU is used")
    args = parser.parse_args()

    path = Path(args.data_dir).resolve()

    logging.info(f"Processing in {path}")
    logging.info(f"Reactants with reagents: src-{args.subset}.txt")
    logging.info(f"Products: tgt-{args.subset}.txt")
    tokenized_path = ut.get_root_dir() / "data" / "test" / f"{path.name.lower()}_no_reagents_{args.subset}.txt"
    output_path = ut.get_root_dir() / "experiments" / "results" / f"{path.name.lower()}_new_reagents_{args.subset}.txt"
    reag_predictor = MolecularTransformerReagentPredictor(
        model_path=args.reagent_model,
        tokenized_path=str(tokenized_path),
        output_path=str(output_path),
        beam_size=args.beam_size,
        n_best=args.beam_size,
        gpu=args.gpu
    )
    get_files_for_forward_prediction(path,
                                     args.subset,
                                     reag_predictor,
                                     mixed=args.mixed_precursors)
