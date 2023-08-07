import argparse
from datetime import datetime
import logging

from rdkit import RDLogger

import numpy as np
import pandas as pd

from utils import match_accuracy, canonicalize_smiles, get_root_dir


def detokenize(smi: str) -> str:
    return smi.replace(" ", "")


def main(args):
    targets = pd.read_csv(args.targets, header=None)
    targets.columns = ["ground_truth"]
    targets["ground_truth"] = targets["ground_truth"].apply(detokenize)
    # TODO Move to prepare_data.py
    # targets = targets.apply(remove_redundant_separators)
    # targets = targets.apply(ut.IonAssembler.run)
    # targets = targets.apply(standardize_pd_pph3)
    with open(args.predictions) as f:
        # predictions = [canonicalize_smiles(detokenize(i.strip())) for i in f.readlines()]
        predictions = [canonicalize_smiles(detokenize(i.strip())) for i in f.readlines()]
    predictions = pd.DataFrame(np.array(predictions).reshape(-1, args.beam_size))
    predictions.columns = [f"pred_{i + 1}" for i in range(args.beam_size)]

    logging.info("=== Invalid predicted SMILES ===")
    for c in predictions.columns:
        logging.info(f"{c}:\t{100 * (predictions[c] == '').mean():.3f} %\t({(predictions[c] == '').sum()})")

    target_and_predictions = pd.concat((targets, predictions), axis=1)

    topn_exact_match_acc = target_and_predictions.apply(
        lambda x: match_accuracy(x, 'exact'),
        axis=1
    )
    topn_exact_match_acc = pd.DataFrame(topn_exact_match_acc.to_list())
    topn_exact_match_acc.columns = [f"top_{i + 1}_exact" for i in range(args.beam_size)]

    logging.info("=== Top-N exact match accuracy ===")
    for c in topn_exact_match_acc.columns:
        logging.info(f"{c}:\t{100 * topn_exact_match_acc[c].mean():.3f} %")

    topn_partial_match_acc = target_and_predictions.apply(
        lambda x: match_accuracy(x, 'partial'),
        axis=1
    )
    topn_partial_match_acc = pd.DataFrame(topn_partial_match_acc.to_list())
    topn_partial_match_acc.columns = [f"top_{i + 1}_partial" for i in range(args.beam_size)]

    logging.info("=== Top-N partial match accuracy ===")
    for c in topn_partial_match_acc.columns:
        logging.info(f"{c}:\t{100 * topn_partial_match_acc[c].mean():.3f} %")


if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')

    logging.basicConfig(
        filename=get_root_dir() / "logs" / f"score_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('--predictions', '-p', type=str, default="",
                        help="Path to file containing the predictions")
    parser.add_argument('--targets', '-t', type=str, default="",
                        help="Path to file containing targets")

    args = parser.parse_args()

    logging.info(f"Target: {args.targets}")
    logging.info(f"Predictions: {args.predictions}")

    main(args)
