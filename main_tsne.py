"""Pas 7: t-SNE 2D cu multiple perplexity, pe primele 50 componente PCA."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]


def main():
    functii.goleste_data_out(tokens=["_tSNE.", "Scoruri_tSNE", "KL_tSNE", "Grid_tSNE"])
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    print(f"[info] t-SNE pe X = {x.shape}, PCA intermediar 50 dim")

    perplexity_list = (5, 30, 50, 100)
    rez = reducere_dim.aplica_tsne(df, x, metadata, perplexity_list=perplexity_list, n_pca=50)
    e = rez.extra

    # CSV pentru fiecare perplexity
    for perp, d in e["rezultate_per_perp"].items():
        pd.DataFrame(d["scoruri"], columns=["t1", "t2"]
                     ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
            functii.DATA_OUT / f"Scoruri_tSNE_perp{perp}.csv", index=False)

    pd.DataFrame({
        "perplexity": list(e["rezultate_per_perp"].keys()),
        "KL_divergence": [d["kl_divergence"] for d in e["rezultate_per_perp"].values()],
    }).to_csv(functii.DATA_OUT / "KL_tSNE.csv", index=False)

    # Grid 2×2: panou per perplexity, pe fiecare coloraj
    for by in ["artist", "stil", "epoca", "gen"]:
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        for ax, perp in zip(axes.flat, perplexity_list):
            scoruri = e["rezultate_per_perp"][perp]["scoruri"]
            categorii = sorted(metadata[by].dropna().unique().tolist())
            culori = grafice.generare_culori(len(categorii))
            for j, cat in enumerate(categorii):
                mask = (metadata[by] == cat).values
                ax.scatter(scoruri[mask, 0], scoruri[mask, 1], color=culori[j],
                           alpha=0.6, s=12, edgecolors="none", label=str(cat))
            ax.set_title(f"t-SNE perplexity={perp} — {by}")
            ax.set_xticks([])
            ax.set_yticks([])
            if len(categorii) <= 30:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
        plt.tight_layout()
        plt.savefig(functii.DATA_OUT / f"Grid_tSNE_{by}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
    grafice.show()


if __name__ == "__main__":
    main()
