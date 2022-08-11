"""SMILES Tokenizer module.
"""
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch

Tokens = List[str]


class SMILESTokenizer:
    """A class for tokenizing and encoding SMILES.

    The tokenizer has a vocabulary that maps tokens to unique integers (a dictionary Dict[str, int]),
    and is created from a set of SMILES. Unless specified otherwise all single character are treated as tokens,
    but the user can specify additional tokens with a list of strings, as well as a list of regular expressions.
    Using the tokenized SMILES the tokenizer also encodes data to lists of `torch.Tensor`.

    This class can also be extended to allow for different and more advanced tokenization schemes.
    When extending the class the functions `tokenize`, `convert_tokens_to_ids`, and `convert_ids_to_encodings`
    can be overridden to change the tokenizer's behaviour. These three function are all used in the `encode` function
    which constitutes the entire tokenization pipeline from SMILES to encodings. When modifying the three
    aforementioned functions the inverses should be modified if necessary,
    these are: `convert_encoding_to_ids`, `convert_ids_to_tokens`, and `detokenize`,
    and are used in the `decode` function.

    Calling an instance of the class on a list of SMILES (or a single SMILES) will return a  list of torch tensors
    with the encoded data, and is equivalent to calling `encode`.

    Inspiration for this tokenizer Class was taken from https://huggingface.co/transformers/main_classes/tokenizer.html
    and https://github.com/MolecularAI/Reinvent/blob/master/models/vocabulary.py

    Initializes the SMILESTokenizer by setting necessary parameters as well as
    compiling regular expressions form the given token, and regex_token lists.
    If a list of SMILES is provided a vocabulary is also created using this list.

    Note that both the token and regex list are used when creating the vocabulary of the tokenizer.
    Note also that the list of regular expressions take priority when parsing the SMILES,
    and tokens earlier are in the lists are also prioritized.

    The `encoding_type` argument specifies the type of encoding used. Must be either 'index' or 'one hot'.
    The former means that the encoded data are integer representations of the tokens found,
    while the latter is one hot encodings of these ids. Defaults to "index".


    :param tokens:  A list of tokens (strings) that the tokenizer uses when tokenizing SMILES. Defaults to None.
    :param regex_token_patterns: A list of regular expressions that the tokenizer uses when tokenizing SMILES.

    :raises: ValueError: If the `encoding_type` is invalid.
    """

    def __init__(
            self,
            tokens: Optional[List[str]] = None,
            regex_token_patterns: Optional[List[str]] = None,
            bos_token: str = "^",
            eos_token: str = "&",
            pad_token: str = " ",
            unk_token: str = "?") -> None:

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self._regex_tokens = []
        self._tokens = []

        regex_token_patterns = regex_token_patterns or []
        tokens = tokens or []

        self.add_tokens(regex_token_patterns, regex=True)
        self.add_tokens(tokens, regex=False)

        self._re = None
        self._vocabulary = {}
        self._decoder_vocabulary = {}

    @classmethod
    def based_on_smiles(cls, smiles, *args, **kwargs):
        """
        :param smiles: A list of SMILES that are used to create the vocabulary for the tokenizer.
        :param smiles:
        :param args: Tokens and regex token patterns
        :param kwargs: Service tokens
        :return:
        """
        t = cls(*args, **kwargs)
        t.create_vocabulary_from_smiles(smiles)
        return t

    @property
    def special_tokens(self) -> Dict[str, str]:
        """ Returns a dictionary of non-character tokens"""
        return {
            "start": self.bos_token,
            "end": self.eos_token,
            "pad": self.pad_token,
            "unknown": self.unk_token,
        }

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Tokens vocabulary.

        :return: Tokens vocabulary
        """
        if not self._vocabulary:
            self._vocabulary = self._reset_vocabulary()
        return self._vocabulary

    @property
    def decoder_vocabulary(self) -> Dict[int, str]:
        """Decoder tokens vocabulary.

        :return: Decoder tokens vocabulary
        """
        if not self._decoder_vocabulary:
            self._decoder_vocabulary = self._reset_decoder_vocabulary()
        return self._decoder_vocabulary

    @property
    def re(self):
        """Tokens Regex Object.

        :return: Tokens Regex Object
        """
        if not self._re:
            self._re = self._get_compiled_regex(self._tokens, self._regex_tokens)
        return self._re

    def _reset_vocabulary(self) -> Dict[str, int]:
        """Create a new token vocabulary.

        :return: New tokens vocabulary
        """
        return {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }

    def _reset_decoder_vocabulary(self) -> Dict[int, str]:
        """Create a new decoder tokens vocabulary.

        :return: New decoder tokens vocabulary
        """
        return {i: t for t, i in self.vocabulary.items()}

    def encode(
            self,
            data: Union[List[str], str]
    ) -> List[torch.Tensor]:
        """Encodes a list of SMILES or a single SMILES into torch tensor(s).

        The encoding is specified by the tokens and regex supplied to the tokenizer
        class. This function uses the three functions `tokenize`,
        `convert_tokens_to_ids`, and `convert_ids_to_encodings` as the encoding
        process.

        :param data: A list of SMILES or a single SMILES.

        :raises ValueError: If the `encoding_type` is invalid.

        :return:  A list of tensors containing the encoded SMILES.
        """

        if isinstance(data, str):
            data = [data]

        return self.convert_tokens_to_ids(self.tokenize(data))

    def tokenize(self, data: List[str]) -> List[List[str]]:
        """Tokenizes a list of SMILES into lists of tokens.

        The conversion is done by parsing the SMILES using regular expressions, which have been
        compiled using the token and regex lists specified in the tokenizer. This
        function is part of the SMILES encoding process and is called in the
        `encode` function.

        :param data: A list os SMILES to be tokenized.

        :return: Lists of tokens.
        """
        tokenized_data = []

        for smi in data:
            tokens = self.re.findall(smi)
            tokenized_data.append(
                [self.bos_token] + tokens + [self.eos_token]
            )

        return tokenized_data

    def convert_tokens_to_ids(self, token_data: List[List[str]]) -> List[torch.Tensor]:
        """Converts lists of tokens to lists of token ids.

        The tokens are converted to ids using the tokenizers vocabulary.

        :param token_data: Lists of tokens to be converted.

        :return: Lists of token ids that have been converted from tokens.
        """
        tokens_lengths = list(map(len, token_data))
        ids_list = []

        for tokens, length in zip(token_data, tokens_lengths):
            ids_tensor = torch.zeros(length, dtype=torch.long)
            for tdx, token in enumerate(tokens):
                ids_tensor[tdx] = self.vocabulary.get(
                    token, self.vocabulary[self.unk_token]
                )
            ids_list.append(ids_tensor)

        return ids_list

    def decode(self, encoded_data: List[torch.Tensor]) -> List[str]:
        """Decodes a list of SMILES encodings back into SMILES.

        This function is the inverse of `encode` and utilizes the three functions
        `convert_encoding_to_ids`, `convert_ids_to_tokens`, and `detokenize`.

        :param encoded_data: The encoded SMILES data to be
                decoded into SMILES.


        :return: A list of SMILES.
        """
        smiles = self.detokenize(self.convert_ids_to_tokens(encoded_data))

        return smiles

    def detokenize(self,
                   token_data: List[List[str]],
                   include_end_of_line_token: bool = False) -> List[str]:
        """Detokenizes lists of tokens into SMILES by concatenating the token strings.

        This function is used in the `decode` function when decoding
        data into SMILES, and it is the inverse of `tokenize`.

        :param token_data: Lists of tokens to be detokenized.
        :param include_end_of_line_token: If `True` end of line
            characters `\\n` are added to the detokenized SMILES. Defaults to False

        :return: A list of detokenized SMILES.
        """

        character_lists = [self._strip_list(tokens.copy()) for tokens in token_data]

        if include_end_of_line_token:
            for s in character_lists:
                s.append("\n")

        strings = ["".join(s) for s in character_lists]

        return strings

    def convert_ids_to_tokens(self, ids_list: List[torch.Tensor]) -> List[List[str]]:
        """Converts lists of token ids to a token tensors.

        This function is used when decoding data using the `decode` function,
        and is the inverse of `convert_tokens_to_ids`.

        :param ids_list: A list of Tensors where each
                Tensor containts the ids of the tokens it represents.

        :return: A list where each element is a list of the
                tokens corresponding to the input ids.
        """
        tokens_data = []
        for ids in ids_list:
            tokens = [self.decoder_vocabulary[i] for i in ids.tolist()]
            tokens_data.append(tokens)

        return tokens_data

    def add_tokens(self, tokens: List[str], regex: bool) -> None:
        """Adds tokens to the classes list of tokens.

        The new tokens are added to the front of the token list and take priority over old tokens. Note that the
        vocabulary of the tokenizer is not updated after the tokens are added,
        and must be updated by calling `create_vocabulary_from_smiles`.

        :param tokens: List of tokens to be added.
        :param regex: If `True` the input tokens are treated as
                regular expressions and are added to the list of regular expressions
                instead of token list. Defaults to False.

        :raises ValueError: If any of the tokens supplied are already in the list
                of tokens.
        """
        existing_tokens = self._regex_tokens if regex else self._tokens
        for token in tokens:
            if token in existing_tokens:
                raise ValueError('"{}" already present in list of tokens.'.format(token))

        if regex:
            self._regex_tokens[0:0] = tokens
        else:
            self._tokens[0:0] = tokens

        # Get a compiled tokens regex
        self._re = self._get_compiled_regex(self._tokens, self._regex_tokens)

    def create_vocabulary_from_smiles(self, smiles: List[str]) -> None:
        """Creates a vocabulary by iteratively tokenizing the SMILES and adding
        the found tokens to the vocabulary.

        A `vocabulary` is a dictionary that maps tokens (str) to integers.
        The tokens vocabulary is not the same as the list of tokens,
        since tokens are also found by applying the list of regular expressions.

        A `decoder_vocabulary` is the inverse of the
        vocabulary. It is always possible to create an inverse since the vocabulary
        always maps to unique integers.


        :param smiles: List of SMILES whose tokens are used to create
                the vocabulary.
        """
        # Reset Tokens Vocabulary
        self._vocabulary = self._reset_vocabulary()

        for tokens in self.tokenize(smiles):
            for token in tokens:
                self._vocabulary.setdefault(token, len(self._vocabulary))

        # Reset decoder tokens vocabulary
        self._decoder_vocabulary = self._reset_decoder_vocabulary()

    def remove_token_from_vocabulary(self, token: str) -> None:
        """Removes a token from the tokenizers `vocabulary` and the corresponding
        entry in the `decoder_vocabulary`.

        :param token: Token to be removed from `vocabulary`.

        :raises ValueError: If the specified token can't be found on the `vocabulary`.
        """
        vocabulary_tokens = list(self.vocabulary.keys())

        if token not in vocabulary_tokens:
            raise ValueError("{} is not in the vocabulary".format(token))

        vocabulary_tokens.remove(token)

        # Recreate tokens vocabulary
        self._vocabulary = {t: i for i, t in enumerate(vocabulary_tokens)}

    def _strip_list(self, tokens: List[str]) -> List[str]:
        """Cleanup tokens list from control tokens.

        :param tokens: List of tokens
        """
        strip_characters = {self.pad_token, self.bos_token, self.eos_token}
        res = []
        for i in tokens:
            if i not in strip_characters:
                res.append(i)
            if i == self.eos_token:
                break
        return res

    def _get_compiled_regex(self,
                            tokens: List[str],
                            regex_tokens: List[str]) -> 'Pattern':
        """Create a Regular Expression Object from a list of tokens and regular expression tokens.

        :param tokens: List of tokens
        :param regex_tokens: List of regular expression tokens
        :return: Regular Expression Object
        """
        regex_string = r"("
        tokens.sort(key=lambda x: -len(x))
        for token in tokens:
            processed_token = token
            for special_character in "()[]":
                processed_token = processed_token.replace(
                    special_character, "\\{}".format(special_character)
                )
            regex_string += processed_token + r"|"
        for token in regex_tokens:
            regex_string += token + r"|"
        regex_string += r".)"

        return re.compile(regex_string)
