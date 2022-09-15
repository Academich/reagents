import sys
from pathlib import Path
from multiprocessing import cpu_count
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from rdkit import RDLogger

import src.utils as ut
from src.prediction import MTProductPredictor, MTReagentPredictor

RDLogger.DisableLog('rdApp.*')

tokenizer = MTProductPredictor(None, None, None)._smi_tokenizer

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


def get_files_for_forward_prediction(path,
                                     subset: str,
                                     reag_predictor: 'MTReagentPredictor',
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
    arrow = ">>" if mixed else ">"
    data = data_src + arrow + data_tgt
    data.columns = ["rxn_original"]

    # Determine reagents by RDKit
    data["rxn_reagents_rdkit"] = ut.parallelize_on_rows(data["rxn_original"],
                                                        ut.assign_reaction_roles_schneider,
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
    reag_predictor.predict(data["rxn_no_reagents"])
    reag_predictor.load_predictions()
    predicted_reagents = reag_predictor.predictions

    # ===============================================================================
    # === Strategy 0: Delete all reagents entirely
    # ===============================================================================

    direct = Path(str(path) + "_no_reags/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(reactants.apply(tokenizer))
        )

    # ===============================================================================
    # === Strategy 1: Replace reagents with top-1 reagent prediction in all cases
    # ===============================================================================

    data["rgs_top1"] = predicted_reagents["p_reagents_1"]
    if mixed:
        data["src_reagents_top1"] = (reactants + '.' + data["rgs_top1"]).str.strip('.').apply(shuffle_molecules)
    else:
        data["src_reagents_top1"] = reactants + '>' + data["rgs_top1"]

    direct = Path(str(path) + "_reags_top1/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(data["src_reagents_top1"].apply(tokenizer))
        )

    # ===============================================================================
    # === Strategy 2: Replace reagents with top-1 reagent prediction if there are more
    # === molecules in the prediction and the prediction is valid
    # ===============================================================================
    data["rgs_top1_and_rdkit"] = np.nan

    data["src_reagents_top1_and_rdkit"] = np.nan

    needs_replacement = (data["rgs_rdkit"].apply(num_mols) < data["rgs_top1"].apply(num_mols)) & (
                data["rgs_top1"].apply(ut.canonicalize_smiles) != '')
    replace_idx = data[needs_replacement].index
    print(f"Reactions altered in top1+rdkit strategy: {len(replace_idx)} ({(100 * len(replace_idx)) / len(data):.2f}%)")
    print("Examples of reactions with replaced reagents:")
    print(replace_idx[:100])
    leave_idx = data[~needs_replacement].index

    data.loc[leave_idx, "src_reagents_top1_and_rdkit"] = data_src.loc[leave_idx, 0]
    data.loc[replace_idx, "src_reagents_top1_and_rdkit"] = data.loc[replace_idx, "src_reagents_top1"]

    direct = Path(str(path) + "_reags_top1_and_rdkit/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(data["src_reagents_top1_and_rdkit"].apply(tokenizer))
        )

    # ===============================================================================
    # === Strategy 3: Replace reagents with predicted strings in which the reagents
    # === in all roles are those that are repeated the most across all predictions
    # === from top-1 to top-5
    # ===============================================================================
    data["rgs_role_voting"] = reag_predictor.suggestions_by_roles(n_catal=1, n_solv=1, n_redox=1, n_unspec=2)

    data["rgs_role_voting"] = data["rgs_role_voting"].apply(lambda x: ".".join([i for i in x if i]))

    if mixed:
        data["src_reagents_role_voting"] = (reactants + '.' + data["rgs_role_voting"]).str.strip(
            '.').apply(shuffle_molecules)
    else:
        data["src_reagents_role_voting"] = reactants + '>' + data["rgs_role_voting"]

    direct = Path(str(path) + "_reags_role_voting/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(data["src_reagents_role_voting"].apply(tokenizer))
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a directory with tokenized files for product prediction")
    parser.add_argument("--reagent_model", type=str,
                        help="Path to a trained reagents prediction model", required=True)
    parser.add_argument("--reagent_model_vocab", type=str, required=True,
                        help="Path to the source vocabulary of a trained reagent prediction model")
    parser.add_argument("--including_test", action="store_true",
                        help="Whether to chenge reagents in the test set as well")
    parser.add_argument("--mixed_precursors", action="store_true",
                        help="Whether to mix reactants and reagents together in the generated files")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index, if not specified, CPU is used")
    args = parser.parse_args()

    path = Path(args.data_dir).resolve()

    subsets = [
        # "train",
        "val"
    ]
    if args.including_test:
        subsets.append("test")

    for subset in subsets:
        reag_predictor = MTReagentPredictor(vocabulary_path=args.reagent_model_vocab,
                                            model_path=args.reagent_model,
                                            tokenized_path=f"data/test/{path.name.lower()}_no_reagents_{subset}.txt",
                                            output_path=f"experiments/results/{path.name.lower()}_new_reagents_{subset}.txt",
                                            beam_size=5,
                                            gpu=0)
        get_files_for_forward_prediction(path,
                                         subset,
                                         reag_predictor,
                                         mixed=args.mixed_precursors)
