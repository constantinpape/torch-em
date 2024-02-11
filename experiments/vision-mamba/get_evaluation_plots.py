import os
import pandas as pd
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt

from run_livecell import ROOT


def get_plots(root_dir):
    all_methods_dir = sorted(glob(os.path.join(root_dir, "*")))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10), sharex="col", sharey="row")

    for i, method_dir in enumerate(all_methods_dir):
        _method = os.path.split(method_dir)[-1]
        per_method_res = []
        for experiment_dir in sorted(glob(os.path.join(method_dir, "*")), reverse=True):
            experiment_name = os.path.split(experiment_dir)[-1]
            df = pd.read_csv(os.path.join(experiment_dir, "results.csv"))
            msa_score = df.iloc[0]["mSA"]
            tmp_res = pd.DataFrame(
                [
                    {"name": _method, "type": experiment_name,  "results": msa_score}
                ]
            )
            per_method_res.append(tmp_res)

        res = pd.concat(per_method_res, ignore_index=True)
        container = sns.barplot(
            x="name", y="results", hue="type", data=res, ax=ax[i]
        )

        # adding the numnbers over the barplots
        for j in container.containers:
            container.bar_label(j, fmt='%.3f')

        ax[i].set(xlabel="Experiments", ylabel="Segmentation Quality")
        ax[i].grid(axis="y")

    all_lines, all_labels = ax[-1].get_legend_handles_labels()
    for ax in fig.axes:
        ax.get_legend().remove()

    fig.legend(all_lines, all_labels)
    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.89)
    fig.suptitle("ViMUNet - LiveCELL", fontsize=20)
    plt.savefig("plot.png")


def main():
    get_plots(os.path.join(ROOT, "experiments", "vision-mamba", "scratch"))


if __name__ == "__main__":
    main()
