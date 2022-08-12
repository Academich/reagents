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


def drop_reagents(smi: str) -> str:
    left, _, right = smi.split(">")
    return left + ">>" + right


def extract_reagents(smi: str) -> str:
    _, center, _ = smi.split(">")
    return center


def include_reagents(rxn: str, reags: str) -> str:
    left, right = rxn.split(">>")
    return left + ">" + reags.strip(".") + ">" + right


def get_files_for_forward_prediction(path, subset: str, reag_predictor: 'MTReagentPredictor') -> None:
    """
    Replaces reagents in the tokenized files for product prediction according to different strategies.
    :param path: Path to the directory with tokenized files for OpenNMT
    :param subset: 'train' or 'val'
    :param reag_predictor: A reagent predictor instance. Requires a trained reagents prediction model.
    :return:
    """
    with open(path / f"src-{subset}.txt") as f, open(path / f"tgt-{subset}.txt") as h:
        data_src = [i.strip().replace(" ", "") for i in f.readlines()]
        data_tgt = [i.strip().replace(" ", "") for i in h.readlines()]

    # Assemble full reactions
    data = pd.DataFrame([i + ">" + j for i, j in zip(data_src, data_tgt)])
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
            "\n".join(data["rxn_no_reagents"].apply(lambda x: x[:x.index(">")]).apply(tokenizer))
        )

    # ===============================================================================
    # === Strategy 1: Replace reagents with top-1 reagent prediction in all cases
    # ===============================================================================

    data["rgs_top1"] = predicted_reagents["p_reagents_1"]
    data["rxn_reagents_top1"] = data[["rxn_no_reagents",
                                      "rgs_top1"]].apply(lambda x: include_reagents(*x), axis=1)
    data["rxn_reagents_top1"] = data["rxn_reagents_top1"].apply(lambda x: x[::-1][x[::-1].index(">") + 1:][::-1])
    direct = Path(str(path) + "_reags_top1/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(data["rxn_reagents_top1"].apply(tokenizer))
        )

    # ===============================================================================
    # === Strategy 2: Replace reagents with top-1 reagent prediction if the length
    # === of the prediction is larger
    # ===============================================================================
    data["rgs_top1_and_rdkit"] = np.nan
    needs_replacement = data[
        (data["rgs_rdkit"].apply(lambda x: len(x)) < predicted_reagents["p_reagents_1"].apply(lambda x: len(x)))]
    data.loc[needs_replacement.index, "rgs_top1_and_rdkit"] = predicted_reagents.loc[needs_replacement.index][
        "p_reagents_1"]
    data["rgs_top1_and_rdkit"].fillna(data["rgs_rdkit"], inplace=True)
    data["rxn_reagents_top1_and_rdkit"] = data[["rxn_no_reagents",
                                                "rgs_top1_and_rdkit"]].apply(lambda x: include_reagents(*x), axis=1)
    data["rxn_reagents_top1_and_rdkit"] = data["rxn_reagents_top1_and_rdkit"].apply(
        lambda x: x[::-1][x[::-1].index(">") + 1:][::-1])

    direct = Path(str(path) + "_reags_top1_and_rdkit/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(data["rxn_reagents_top1_and_rdkit"].apply(tokenizer))
        )

    # ===============================================================================
    # === Strategy 3: Replace reagents with predicted strings in which the reagents
    # === in all roles are those that are repeated the most across all predictions
    # === from top-1 to top-5
    # ===============================================================================
    data["rgs_role_voting"] = reag_predictor.suggestions_by_roles()
    data["rgs_role_voting"] = data["rgs_role_voting"].apply(lambda x: ".".join([i for i in x if i]))
    data["rxn_reagents_role_voting"] = data[["rxn_no_reagents",
                                             "rgs_role_voting"]].apply(lambda x: include_reagents(*x), axis=1)
    data["rxn_reagents_role_voting"] = data["rxn_reagents_role_voting"].apply(
        lambda x: x[::-1][x[::-1].index(">") + 1:][::-1])
    direct = Path(str(path) + "_reags_role_voting/")
    direct.mkdir(parents=True, exist_ok=True)
    with open(direct / f"src-{subset}.txt", 'w') as h:
        h.write(
            "\n".join(data["rxn_reagents_role_voting"].apply(tokenizer))
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a directory with tokenized files for product prediction")
    parser.add_argument("--reagent_model", type=str,
                        help="Path to a trained reagents prediction model", required=True)
    parser.add_argument("--reagent_model_vocab", type=str, required=True,
                        help="Path to the source vocabulary of a trained reagent prediction model")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index, if not specified, CPU is used")
    args = parser.parse_args()

    path = Path(args.data_dir).resolve()
    subset = "val"
    reag_predictor = MTReagentPredictor(vocabulary_path=args.reagent_model_vocab,
                                        model_path=args.reagent_model,
                                        tokenized_path=f"data/test/{path.name.lower()}_no_reagents_{subset}.txt",
                                        output_path=f"experiments/results/{path.name.lower()}_new_reagents_{subset}.txt",
                                        beam_size=5,
                                        gpu=0)
    get_files_for_forward_prediction(path, subset, reag_predictor)

    subset = "train"
    reag_predictor = MTReagentPredictor(vocabulary_path=args.reagent_model_vocab,
                                        model_path=args.reagent_model,
                                        tokenized_path=f"data/test/{path.name.lower()}_no_reagents_{subset}.txt",
                                        output_path=f"experiments/results/{path.name.lower()}_new_reagents_{subset}.txt",
                                        beam_size=5,
                                        gpu=0)
    get_files_for_forward_prediction(path, subset, reag_predictor)
