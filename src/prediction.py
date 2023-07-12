import subprocess

import pandas as pd
import numpy as np

from src.tokenizer import smi_tokenizer
from src.preprocessing.reagents_classification import HeuristicRoleClassifier


class MolecularTransformerPredictor:

    def __init__(self,
                 model_path: str,
                 tokenized_path: str,
                 output_path: str,
                 beam_size: int = 5,
                 n_best: int = 5,
                 gpu: int | None = None,
                 batch_size: int = 64,
                 max_length: int = 200,
                 with_score: bool = False):
        self.model_path = model_path
        self.tokenized_path = tokenized_path
        self.output_path = output_path
        self.beam_size = beam_size
        self.n_best = n_best
        self.gpu = gpu
        self.batch_size = batch_size
        self.max_length = max_length
        self.with_score = with_score

        self.predictions = None
        self.pred_probs = None

    def _run_inference(self):
        command = ["onmt_translate",
                   "-model", self.model_path,
                   "-src", self.tokenized_path,
                   "-output", self.output_path,
                   "-batch_size", str(self.batch_size),
                   "-max_length", str(self.max_length),
                   "-beam_size", str(self.beam_size),
                   "-n_best", str(self.n_best),
                   "-replace_unk"]
        if self.with_score:
            command = command + ["-with_score"]
        if self.gpu is not None:
            command = command + ["-gpu", str(self.gpu)]
        print("Running the command:")
        print(" ".join(command))
        subprocess.run(command, check=True)

    def _load_predictions(self):
        with open(self.output_path) as f:
            predictions = f.readlines()

        probs = None
        smiles, prob_scores = [], []
        for line in predictions:
            if self.with_score:
                *tokens, score = line.split()
                prob_scores.append(np.exp(float(score)))
            else:
                tokens = line.split()
            smiles.append("".join(tokens))

        pred_smiles = pd.DataFrame(np.array(smiles).reshape(-1, self.n_best))
        if self.with_score:
            probs = pd.DataFrame(np.array(prob_scores).reshape(-1, self.n_best))
        return pred_smiles, probs

    def make_and_store_predictions(self, s: pd.Series):
        input_tokens = s.apply(smi_tokenizer)
        with open(self.tokenized_path, "w") as f:
            f.write("\n".join(input_tokens))
        self._run_inference()

    def load_predictions(self):
        self.predictions, self.pred_probs = self._load_predictions()


class MolecularTransformerProductPredictor(MolecularTransformerPredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_predictions(self):
        super().load_predictions()
        self.predictions.columns = [f"p_products_{i + 1}" for i in range(self.n_best)]
        if self.pred_probs is not None:
            self.pred_probs.columns = [f"p_products_{i + 1}_conf" for i in range(self.n_best)]


class MolecularTransformerReagentPredictor(MolecularTransformerPredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_predictions(self):
        super().load_predictions()
        self.predictions.columns = [f"p_reagents_{i + 1}" for i in range(self.n_best)]
        if self.pred_probs is not None:
            self.pred_probs.columns = [f"p_reagents_{i + 1}_conf" for i in range(self.n_best)]

    def suggestions_by_roles(self, n_catal, n_solv, n_redox, n_unspec):
        bag_of_predictions = self.predictions.apply(lambda x: '.'.join(x), axis=1)
        bag_of_predictions = bag_of_predictions.apply(lambda x: HeuristicRoleClassifier.role_voting(x,
                                                                                                    n_catal,
                                                                                                    n_solv,
                                                                                                    n_redox,
                                                                                                    n_unspec))
        return bag_of_predictions
