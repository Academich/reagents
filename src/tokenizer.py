import re

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(PATTERN)


def smi_tokenizer(smi: str) -> str:
    """
    Tokenize a SMILES molecule or reaction like in Molecular Transformer (Schwaller et al. 2019)
    """
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

if __name__ == '__main__':
    s = "c1ccccc1.CC#N.O=C[O-][Na+].[NH4+].[Cl-].[235U]"
    print(smi_tokenizer(s))
