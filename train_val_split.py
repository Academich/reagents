from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
RDLogger.DisableLog('rdApp.info')

np.random.seed(123456)

parser = ArgumentParser()
parser.add_argument("--path_data", type=str)
parser.add_argument("--save_folder", type=str)
parser.add_argument("--val_size", type=int)
args = parser.parse_args()

uspto_full = pd.read_csv(args.path_data, sep="\t", usecols=["OriginalReaction"])
uspto_full = uspto_full.sample(frac=1).reset_index(drop=True)

val_index = uspto_full.sample(args.val_size).index
uspto_val = uspto_full.loc[val_index]
uspto_train = uspto_full.drop(val_index)

print("     Train: {} reactions.".format(uspto_train.shape[0]))
print("Validation: {} reactions.".format(uspto_val.shape[0]))

if __name__ == '__main__':
    # === Saving ===
    save_path = Path(args.save_folder).resolve()
    uspto_train.to_csv(save_path / "uspto_train.csv", index=False, columns=["OriginalReaction"], sep="\t")
    uspto_val.to_csv(save_path / "uspto_val.csv", index=False, columns=["OriginalReaction"], sep="\t")
