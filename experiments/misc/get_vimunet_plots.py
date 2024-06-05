import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


LIVECELL_RESULTS = {
    "UNet": {"boundaries": 0.372, "distances": 0.429},
    r"UNETR$_{Base}$": {"boundaries": 0.11, "distances": 0.145},
    r"UNETR$_{Large}$": {"boundaries": 0.171, "distances": 0.157},
    r"UNETR$_{Huge}$": {"boundaries": 0.216, "distances": 0.136},
    r"nnUNet$_{v2}$": {"boundaries": 0.228},
    r"UMamba$_{Bot}$": {"boundaries": 0.234},
    r"UMamba$_{Enc}$": {"boundaries": 0.23},
    r"$\bf{ViMUNet}$$_{Tiny}$": {"boundaries": 0.269, "distances": 0.381},
    r"$\bf{ViMUNet}$$_{Small}$": {"boundaries": 0.274, "distances": 0.397},
}

CREMI_RESULTS = {
    "UNet": {"boundaries": 0.354},
    r"UNETR$_{Base}$": {"boundaries": 0.285},
    r"UNETR$_{Large}$": {"boundaries": 0.325},
    r"UNETR$_{Huge}$": {"boundaries": 0.324},
    r"nnUNet$_{v2}$": {"boundaries": 0.452},
    r"UMamba$_{Bot}$": {"boundaries": 0.471},
    r"UMamba$_{Enc}$": {"boundaries": 0.467},
    r"$\bf{ViMUNet}$$_{Tiny}$": {"boundaries": 0.518},
    r"$\bf{ViMUNet}$$_{Small}$": {"boundaries": 0.53},
}

DATASET_MAPPING = {
    "livecell": "LIVECell",
    "cremi": "CREMI"
}

plt.rcParams["font.size"] = 24


def plot_per_dataset(dataset_name):
    if dataset_name == "livecell":
        results = LIVECELL_RESULTS
    else:
        results = CREMI_RESULTS

    models = list(results.keys())
    metrics = list(results[models[0]].keys())

    markers = ['^', '*']

    fig, ax = plt.subplots(figsize=(15, 12))

    x_pos = np.arange(len(models))

    bar_width = 0.05

    for i, metric in enumerate(metrics):
        scores_list = []
        for model in models:
            try:
                score = results[model][metric]
            except KeyError:
                score = None

            scores_list.append(score)

        ax.scatter(x_pos + i * bar_width - bar_width, scores_list, s=250, label=metric, marker=markers[i])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, va='top', ha='center', rotation=45)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if dataset_name == "cremi":
        ax.set_yticks(np.linspace(0, 0.5, 11)[1:])
    else:
        ax.set_yticks(np.linspace(0, 0.4, 9)[1:])

    ax.set_ylabel('Segmentation Accuracy', labelpad=15)
    ax.set_xlabel(None)
    ax.set_title(DATASET_MAPPING[dataset_name], fontsize=32, y=1.025)
    ax.set_ylim(0)
    ax.legend(loc='lower center', fancybox=True, shadow=True, ncol=2)

    best_models = sorted(models, key=lambda x: max(results[x].values()), reverse=True)[:3]
    sizes = [100, 70, 40]
    for size, best_model in zip(sizes, best_models):
        best_scores = [results[best_model].get(metric, 0) for metric in metrics]
        best_index = models.index(best_model)

        # HACK
        offset = 0 if dataset_name == "livecell" else 0.05

        ax.plot(
            best_index - offset, max(best_scores), marker='o', markersize=size, linestyle='dotted',
            markerfacecolor='gray', markeredgecolor='black', markeredgewidth=2, alpha=0.2
        )

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{dataset_name}.png")
    plt.savefig(f"{dataset_name}.svg", transparent=True)
    plt.savefig(f"{dataset_name}.pdf")


def main():
    plot_per_dataset("livecell")
    plot_per_dataset("cremi")


if __name__ == "__main__":
    main()
