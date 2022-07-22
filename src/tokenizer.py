import re
import json
from typing import List

import torch

from src.pysmilesutils.pysmilesutils_tokenizer import SMILESTokenizer

from src.preprocessing.atoms_and_groups import NONMETALS

RE_BLOCK_ATOM = r"(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]|[A-Z][a-z]?(?<!c|n|o|p|s)|se|as|si|sn|sb|ge|.)"
RE_TOKEN_PATTERNS = [r"\[[^\]]*\]", r"%[0-9]{2}"]


class ChemSMILESTokenizer(SMILESTokenizer):
    """
    A subclass of the `SMILESTokenizer` that treats all atoms as tokens.

    This tokenizer works by applying two different sets of regular expressions:
    one for atoms inside blocks ([]) and another for all other cases. This allows
    the tokenizer to find all atoms as blocks without having a comprehensive list
    of all atoms in the token list.
    """

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self._vocabulary = self._reset_vocabulary()
        self.add_tokens(NONMETALS, regex=False)
        self.add_tokens(RE_TOKEN_PATTERNS, regex=True)
        self.re_block_atom = re.compile(RE_BLOCK_ATOM)

    def tokenize(self, smiles: List[str]) -> List[List[str]]:
        """
        Converts a list of SMILES into a list of lists of tokens, where all atoms are
        considered to be tokens.

        The function first scans the SMILES for atoms and bracketed expressions
        using regular expressions. These bracketed expressions are then parsed
        again using a different regular expression.

        :param smiles: List of SMILES.

        :return: List of tokenized SMILES.
        """
        data_tokenized = super().tokenize(smiles)
        final_data = []
        for tokens in data_tokenized:
            final_tokens = []
            for token in tokens:
                if token.startswith("["):
                    final_tokens += self.re_block_atom.findall(token)
                else:
                    final_tokens.append(token)
            final_data.append(final_tokens)

        return final_data

    def save_vocabulary(self, voc_save_path: str) -> None:
        with open(voc_save_path, "w") as f:
            json.dump(self.vocabulary, f)

    def load_vocabulary(self, voc_save_path: str) -> None:
        with open(voc_save_path, "r") as f:
            self._vocabulary = json.load(f)
        self._decoder_vocabulary = self._reset_decoder_vocabulary()

    @property
    def padding_token_index(self):
        return self._vocabulary[self.pad_token]

    @property
    def eos_token_index(self):
        return self._vocabulary[self.eos_token]

    @property
    def bos_token_index(self):
        return self._vocabulary[self.bos_token]


if __name__ == '__main__':
    t = ChemSMILESTokenizer.based_on_smiles(["CCCC"])
    print(t.vocabulary)
    s = torch.tensor([[1, 0, 2]]).long()
    e = t.encode("CCCC")
    print(e)
    print(t.decode(e))
    print(t.decode(s))
