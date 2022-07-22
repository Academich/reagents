import random
from typing import List
from src.pysmilesutils.pysmilesutils_augmenter import SMILESAugmenter

import numpy as np


def augment_rxn(augmenter: 'SMILESAugmenter', smi: str) -> List[str]:
    np.random.seed(123456)

    num_augmentations = np.random.choice(range(1, 4 + 1), p=[0.4, 0.3, 0.2, 0.1])

    left, center, right = smi.split(">")
    left_smiles = left.split('.')
    center_smiles = center.split('.')
    n_agents = len(center_smiles)
    right_smiles = right.split('.')

    augmented_examples = [smi]
    for _ in range(num_augmentations):
        _left_smi_augm = '.'.join([augmenter(s for s in left_smiles)].pop())
        _right_smi_augm = '.'.join([augmenter(s for s in right_smiles).pop()])

        if n_agents > 1:
            idx_of_reagents_to_move = random.sample(range(n_agents),
                                                    np.random.choice(range(0, n_agents),
                                                                     p=[0.65] + (n_agents - 1) * [
                                                                         0.35 / (n_agents - 1)]))
            center_smiles_moved = [center_smiles[i] for i in idx_of_reagents_to_move]
            center_smiles_moved = '.'.join(center_smiles_moved)
            center_smiles_intact = [center_smiles[i] for i in range(n_agents) if i not in idx_of_reagents_to_move]
            center = '.'.join(center_smiles_intact)
            _left_smi_augm = _left_smi_augm + '.' * bool(center_smiles_moved) + center_smiles_moved

        smiles_augmented_rxn = _left_smi_augm + ">" + center + ">" + _right_smi_augm
        augmented_examples.append(smiles_augmented_rxn)

    return augmented_examples


if __name__ == '__main__':
    from functools import partial
    import pandas as pd

    smiles_randomizer = SMILESAugmenter(restricted=True)
    d = pd.DataFrame.from_dict({"smi": [
        'COc1ccc(-c2csc(CCCCCCC=O)n2)cc1.[O-]Cl>[C].[Al+3].[H-].[H-].[H-].[H-]>COc1ccc(-c2csc(CCCCCCC(=O)O)n2)cc1'
    ]})
    print(d)
    print()
    print(
        d['smi'].apply(partial(augment_rxn, smiles_randomizer)).explode('smi').values
    )
