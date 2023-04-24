from argparse import ArgumentParser, Namespace

import pandas as pd

from src import root
from src.prediction import MTReagentPredictor, MTProductPredictor


def predict_reagents(cliargs: Namespace) -> None:
    src_data = pd.read_csv(cliargs.data, header=None)
    reagent_predictor = MTReagentPredictor(
        vocabulary_path=cliargs.vocab,
        model_path=cliargs.model,
        tokenized_path=str(
            (root / "data" / "test" / f"src_rgs_{cliargs.name}").with_suffix(".txt")
        ),
        output_path=str(
            (root / "experiments" / "results" / f"rgs_{cliargs.name}").with_suffix(".txt")
        ),
        beam_size=cliargs.beam_size,
        n_best=cliargs.n_best,
        batch_size=cliargs.batch_size,
        gpu=cliargs.gpu
    )
    reagent_predictor.predict(src_data)


def predict_products(cliargs: Namespace) -> None:
    src_data = pd.read_csv(cliargs.data, header=None)
    product_predictor = MTProductPredictor(
        model_path=cliargs.model,
        tokenized_path=str(
            (root / "data" / "test" / f"src_prd_{cliargs.name}").with_suffix(".txt")
        ),
        output_path=str(
            (root / "experiments" / "results" / f"prd_{cliargs.name}").with_suffix(".txt")
        ),
        beam_size=cliargs.beam_size,
        n_best=cliargs.n_best,
        batch_size=cliargs.batch_size,
        gpu=cliargs.gpu
    )
    product_predictor.predict(src_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["reagents", "products"],
                        help="Specifies the task - to predict either reagents or products")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Path to the untokenized input data")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to the weights of the model")
    parser.add_argument("--vocab", "-v", type=str, required=True,
                        help="Path to the model's vocabulary")
    parser.add_argument("--name", type=str, default="inference",
                        help="Name of the run")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size")
    parser.add_argument("--n_best", type=int, default=5,
                        help="Number of best options used to grow beams")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Defaults to None, i.e. CPU")
    args = parser.parse_args()
    if args.task == "reagents":
        predict_reagents(args)
    elif args.task == "products":
        predict_products(args)
    else:
        pass
