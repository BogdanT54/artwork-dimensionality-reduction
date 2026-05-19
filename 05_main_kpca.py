"""Pas 5: Kernel PCA (RBF) vs PCA liniar (comparație side-by-side)."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "kpca"


def main():
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)

    n = len(metadata)
    if n > 2000:
        sub_idx = (metadata.groupby("artist", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), max(1, 2000 // metadata["artist"].nunique())),
                                              random_state=42),
                          include_groups=False)
                   ).index.values
        sub_idx = np.array(sub_idx)
    else:
        sub_idx = np.arange(n)
    x_s = x[sub_idx]
    meta_s = metadata.iloc[sub_idx].reset_index(drop=True)
    print(f"[info] KPCA pe sub-eșantion {x_s.shape}  →  data_out/{SUBDIR}/")

    rez = reducere_dim.aplica_kpca(df, x_s, meta_s, kernel="rbf", n_components=20)
    e = rez.extra

    pd.DataFrame(rez.scoruri, columns=[f"K{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).assign(**{c: meta_s[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_KPCA.csv", index=False)
    pd.DataFrame(e["scoruri_pca_referinta"], columns=[f"C{i+1}" for i in range(20)]
                 ).assign(**{c: meta_s[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_PCA_referinta_KPCA.csv", index=False)

    import matplotlib.pyplot as plt
    for by in ["artist", "stil", "epoca", "gen"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        for ax, scoruri, titlu in [(ax1, e["scoruri_pca_referinta"], "PCA liniar"),
                                    (ax2, rez.scoruri, "Kernel PCA (RBF)")]:
            categorii = sorted(meta_s[by].dropna().unique().tolist())
            culori = grafice.generare_culori(len(categorii))
            for j, cat in enumerate(categorii):
                mask = (meta_s[by] == cat).values
                ax.scatter(scoruri[mask, 0], scoruri[mask, 1],
                           color=culori[j], alpha=0.6, s=14, edgecolors="none",
                           label=str(cat))
            ax.set_title(f"{titlu} — colorat pe {by}")
            ax.set_xlabel("Comp1")
            ax.set_ylabel("Comp2")
            ax.grid(True, alpha=0.3)
            if len(categorii) <= 30:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
        plt.tight_layout()
        plt.savefig(OUT / f"Comparatie_KPCA_PCA_{by}.pdf",
                    format="pdf", bbox_inches="tight")
        plt.close(fig)
    grafice.show()


if __name__ == "__main__":
    main()
