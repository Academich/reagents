import re
import subprocess
from typing import Optional

import pandas as pd
import numpy as np

from src.tokenizer import ChemSMILESTokenizer
from src.preprocessing.reagents_classification import HeuristicRoleClassifier
from src import translate_script_path


class MTPredictor:

    def __init__(self,
                 model_path: str,
                 tokenized_path: str,
                 output_path: str,
                 beam_size: int = 5,
                 n_best: int = 5,
                 gpu: Optional[int] = None,
                 batch_size: int = 64,
                 max_length: int = 200):
        self.model_path = model_path
        self.tokenized_path = tokenized_path
        self.output_path = output_path
        self.beam_size = beam_size
        self.n_best = n_best
        self.gpu = gpu
        self.batch_size = batch_size
        self.max_length = max_length

        self.predictions = None
        self.pred_probs = None

    def _smi_tokenizer(self, smi):
        raise NotImplementedError

    def _run_inference(self):
        command = ["python3", translate_script_path,
                   "-model", self.model_path,
                   "-src", self.tokenized_path,
                   "-output", self.output_path,
                   "-batch_size", str(self.batch_size),
                   "-max_length", str(self.max_length),
                   "-beam_size", str(self.beam_size),
                   "-n_best", str(self.n_best),
                   "-log_probs",
                   "-replace_unk",
                   "-fast"]
        if self.gpu is not None:
            command = command + ["-gpu", str(self.gpu)]
        print("Running the command:")
        print(" ".join(command))
        subprocess.run(command, check=True)

    def _load_predictions(self):
        with open(self.output_path) as f:
            pred = f.readlines()
        pred = [i.strip().replace(" ", "") for i in pred]
        pred = pd.DataFrame(np.array(pred).reshape(-1, self.n_best))
        try:
            with open(self.output_path + "_log_probs") as f:
                probs = f.readlines()
            probs = [np.exp(float(i)) for i in probs]
            probs = pd.DataFrame(np.array(probs).reshape(-1, self.n_best))
        except FileNotFoundError:
            probs = None
        return pred, probs

    def predict(self, s: 'pd.Series'):
        input_tokens = s.apply(self._smi_tokenizer)
        with open(self.tokenized_path, "w") as f:
            f.write("\n".join(input_tokens))
        self._run_inference()

    def load_predictions(self):
        _pred, _probs = self._load_predictions()
        self.predictions = _pred
        self.pred_probs = _probs


class MTProductPredictor(MTPredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tkz_pattern = re.compile(
            "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )

    def _smi_tokenizer(self, smi):
        """
        Tokenize a SMILES molecule or reaction. Used in Schwaller et al. 2019
        """
        regex = self.tkz_pattern
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)

    def load_predictions(self):
        super().load_predictions()
        self.predictions.columns = [f"p_products_{i + 1}" for i in range(self.n_best)]
        if self.pred_probs is not None:
            self.pred_probs.columns = [f"p_products_{i + 1}_conf" for i in range(self.n_best)]


class MTReagentPredictor(MTPredictor):

    def __init__(self, vocabulary_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer_source = ChemSMILESTokenizer()
        self.tokenizer_source.load_vocabulary(vocabulary_path)

    def _smi_tokenizer(self, smi):
        tokens = self.tokenizer_source.tokenize([smi])
        return " ".join(tokens[0][1:-1])

    def load_predictions(self):
        super().load_predictions()
        self.predictions.columns = [f"p_reagents_{i + 1}" for i in range(self.n_best)]
        self.pred_probs.columns = [f"p_reagents_{i + 1}_conf" for i in range(self.n_best)]

    def suggestions_by_roles(self, n_catal, n_solv, n_redox, n_unspec):
        bag_of_predictions = self.predictions.apply(lambda x: '.'.join(x), axis=1)
        bag_of_predictions = bag_of_predictions.apply(lambda x: HeuristicRoleClassifier.role_voting(x,
                                                                                                    n_catal,
                                                                                                    n_solv,
                                                                                                    n_redox,
                                                                                                    n_unspec))
        return bag_of_predictions

