from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd

from openTSNE import TSNE

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit.Chem import rdChemReactions
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

np.random.seed(123456)

CLASS_COLORS = {
    "Acylation and related processes": "#C0392B",
    "Heteroatom alkylation and arylation": "#E67E22",
    "C-C bond formation": "#27AE60",
    "Heterocycle formation": "#F1C40F",
    "Protections": "#1186F3",
    "Deprotections": "#707B7C",
    "Reductions": "#40C4DE",
    "Oxidations": "#DC40DE",
    "Functional group interconversion (FGI)": "#CCAF1C",
    "Functional group addition (FGA)": "#8E44AD",

    "Functional group addition (FGA)/Heteroatom alkylation and arylation": "#00FF00"
}


class ReactionFPS:
    """
    Calculates reaction fingerprints using RDKit.
    """

    FP_TYPES = {"AtomPairFP": rdChemReactions.FingerprintType.AtomPairFP,
                "MorganFP": rdChemReactions.FingerprintType.MorganFP,
                "PatternFP": rdChemReactions.FingerprintType.PatternFP,
                "RDKitFP": rdChemReactions.FingerprintType.RDKitFP,
                "TopologicalTorsion": rdChemReactions.FingerprintType.TopologicalTorsion
                }

    def calculate(self,
                  rx_smi: str,
                  fp_method: str,
                  n_bits: int,
                  fp_type: str,
                  include_agents: bool,
                  agent_weight: int,
                  non_agent_weight: int,
                  bit_ratio_agents: float = 0.2
                  ) -> 'np.array':
        """
        Calculates reaction fingerprints for a given reaction SMILES string.
        More info on arguments: https://www.rdkit.org/docs/cppapi/structRDKit_1_1ReactionFingerprintParams.html
        :param rx_smi: Reaction SMILES to calculate fingerprints for.
        :param fp_method: 'structural' or 'difference'.
        :param n_bits: Number of bits in the fingerprint vectors
        :param fp_type: the algorithm for fingerprints, e.g. AtompairFP.
        Be aware that only AtompairFP, TopologicalTorsion and MorganFP are supported in the difference fingerprint.
        :param include_agents: a flag: include the agents of a reaction for fingerprint generation or not
        :param agent_weight: if agents are included, agents could
        be weighted compared to reactants and products in difference fingerprints.
        :param non_agent_weight: in difference fingerprints weight factor for reactants and products compared to agents
        :param bit_ratio_agents: in structural fingerprints it determines the ratio of bits of the agents in the fingerprint
        :return: fingerprint vector (numpy array)
        """
        # === Parameters section
        params = rdChemReactions.ReactionFingerprintParams()
        params.fpSize = n_bits
        params.includeAgents = include_agents
        params.fpType = self.FP_TYPES[fp_type]
        # ===

        rxn = rdChemReactions.ReactionFromSmarts(
            rx_smi,
            useSmiles=True)

        arr = np.zeros((1,))
        if fp_method == "difference":
            params.agentWeight = agent_weight
            params.nonAgentWeight = non_agent_weight
            # NOTE: difference fingerprints are not binary
            fps = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn, params)

        elif fp_method == "structural":
            params.bitRatioAgents = bit_ratio_agents
            # NOTE: structural fingerprints are binary
            fps = rdChemReactions.CreateStructuralFingerprintForReaction(rxn, params)
        else:
            raise ValueError("Invalid fp_method. Allowed are 'difference' and 'structural'")

        ConvertToNumpyArray(fps, arr)
        return arr


def diff_fp(smi: str) -> 'np.array':
    return ReactionFPS().calculate(smi,
                                   fp_method="difference",
                                   n_bits=2048,
                                   fp_type="MorganFP",
                                   include_agents=True,
                                   agent_weight=1,
                                   non_agent_weight=1)


def plot_2d_distribution(x,
                         y,
                         save_path=None,
                         ax=None,
                         title=None,
                         draw_legend=True,
                         colors=None,
                         legend_kwargs=None,
                         label_order=None,
                         **kwargs) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 20))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="best", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    if save_path is not None:

        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_rxn_class_proportions(classes_in: 'pd.Series',
                               classes_out: 'pd.Series',
                               save_path=None) -> None:
    """
    Plot the graph comparing the proportions of the ten reaction classes from USPTO 50K
    between the in-distribution data and the out-of-distribution data.
    :param save_path: Path to save the image.
    :param classes_in: Proportions of reaction classes for the in-distribution data.
    :param classes_out: Proportions of reaction classes for the out-of-distribution data.
    :return:
    """

    plt.figure(figsize=(10, 5))
    width = 0.3

    plt.bar(
        np.arange(len(classes_in)),
        classes_in,
        width,
        color='#abebc6',
        alpha=1,
        label="USPTO 50K"
    )
    plt.bar(
        np.arange(len(classes_in)) + width,
        classes_out[classes_in.index],
        width,
        color='#cd6155',
        alpha=1,
        label="Reaxys Test"
    )
    plt.legend()
    plt.xticks(np.arange(len(classes_in)) + width / 2,
               classes_in.index,
               rotation=75,
               fontsize=11)
    plt.title("Proportion of reaction classes")
    plt.grid(axis='y')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--in_distr", type=str,
                        help="Path to the data which is in-distribution compared to the training set. "
                             "Example: ./uspto50k.csv")
    parser.add_argument("--out_of_distr", type=str,
                        help="Path to the data which is out-of-distribution compared to the training set. "
                             "Example: ./test_reaxys.csv")
    parser.add_argument("--image_dir", type=str,
                        help="Path to the directory to save the generated images in")
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    img_dir.mkdir(exist_ok=True)

    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=cpu_count(),
        random_state=123456,
        verbose=True
    )

    data_in = pd.read_csv(args.in_distr)
    data_out = pd.read_csv(args.out_of_distr)

    stat_in = data_in["General class"].value_counts(normalize=True)
    stat_out = data_out["General class"].value_counts(normalize=True)

    stat_out.loc["Heterocycle formation"] = 0.0

    plot_rxn_class_proportions(stat_in,
                               stat_out,
                               save_path=str(img_dir / "class_proportions.png"))

    fp_in = np.vstack(
        data_in["Reaction"].apply(diff_fp)
    )
    embedding_in = tsne.fit(fp_in)

    fp_out = np.vstack(
        data_out["FullR"].apply(diff_fp)
    )
    embedding_out = embedding_in.transform(fp_out)

    plot_2d_distribution(embedding_in,
                         data_in["General class"],
                         colors=CLASS_COLORS,
                         s=10,
                         save_path=str(img_dir / "uspto_50_tsne.png"))
    plot_2d_distribution(embedding_out,
                         data_out["General class"],
                         s=10,
                         colors={i: CLASS_COLORS.get(i, "#00FF00") for i in data_out["General class"].unique()},
                         save_path=str(img_dir / "reaxys_test_tsne.png"))
    plot_2d_distribution(
        np.vstack((embedding_in, embedding_out)),
        np.array(["USPTO 50K"] * embedding_in.shape[0] + ["Reaxys Test"] * embedding_out.shape[0]),
        save_path=str(img_dir / "uspto50_and_reaxys_tsne.png")
    )

    for c in CLASS_COLORS:
        if c in data_in["General class"].unique() and c in data_out["General class"].unique():
            _u = embedding_in[data_in["General class"] == c]
            _r = embedding_out[data_out["General class"] == c]
            both = np.vstack((_u, _r))
            h = sns.jointplot(x=both[:, 0],
                              y=both[:, 1],
                              kind='scatter',
                              alpha=0.7,
                              palette=["#000000", CLASS_COLORS[c]],
                              hue=np.array(["USPTO 50K"] * _u.shape[0] + ["Reaxys Test"] * _r.shape[0]))
            h.set_axis_labels("TSNE 1", "TSNE 2")
            plt.suptitle(f"TSNE embeddings of points in USPTO 50K and the test set.\nClass: {c}")
            plt.tight_layout()
            plt.savefig(
                str(img_dir / f"tsne_test_and_50k_{c}.png"),
                dpi=300
            )
