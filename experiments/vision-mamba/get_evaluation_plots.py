import os
import pandas as pd
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt

from run_livecell import ROOT


def get_vimunet_plots(root_dir):
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
    plt.savefig("vimunet.png")


def get_unetr_plots():
    experiments = ["affinities", "boundaries", "distances"]

    fig, ax = plt.subplots(1, 3, figsize=(20, 10), sharex="col", sharey="row")

    for i, experiment in enumerate(experiments):
        all_experiments_dir = sorted(
            glob(os.path.join(
                "/home/nimanwai/torch-em/experiments/vision-transformer/unetr/livecell/results/torch-em-scratch",
                "vit_*", experiment, "livecell.csv"
            ))
        )
        per_method_res = []
        for result_path in all_experiments_dir:
            df = pd.read_csv(result_path)
            model_name = result_path.split("/")[-3]
            try:
                score = df.iloc[0]["mSA"]
            except KeyError:
                score = df.iloc[0]["ws1_mSA"]

            tmp_res = pd.DataFrame(
                [
                    {"name": experiment, "type": model_name, "results": score}
                ]
            )
            per_method_res.append(tmp_res)

        res = pd.concat(per_method_res, ignore_index=True)
        container = sns.barplot(
            x="name", y="results", hue="type", data=res, ax=ax[i]
        )

        def get_unet_res(model_name, ax, color):
            unet_res_path = os.path.join(
                "/home/nimanwai/torch-em/experiments/vision-transformer/unetr/livecell/results/torch-em-scratch",
                model_name, experiment, "livecell.csv"
            )
            try:
                unet_score = pd.read_csv(unet_res_path).iloc[0]["mSA"]
            except KeyError:
                unet_score = pd.read_csv(unet_res_path).iloc[0]["ws1_mSA"]
            ax.axhline(unet_score, label=unet_res_path.split("/")[-3], color=color)

        get_unet_res("unet-conv-transpose", ax[i], "darkorange")
        get_unet_res("unet-bilinear", ax[i], "forestgreen")

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
    fig.suptitle("UNETR - LiveCELL", fontsize=20)
    plt.savefig("unetr.png")


def main():
    get_vimunet_plots(os.path.join(ROOT, "experiments", "vision-mamba", "scratch"))
    get_unetr_plots()


if __name__ == "__main__":
    main()
