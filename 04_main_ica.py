"""Pas 4: FastICA + entropie + kurtosis (non-gaussianitate)."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "ica"


def main():
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)

    rez_pca = reducere_dim.aplica_pca(df, x, metadata, n_max=50)
    k = min(30, max(2, rez_pca.extra["n_kaiser"]))
    print(f"[info] ICA cu k = {k} componente (din Kaiser PCA)  →  data_out/{SUBDIR}/")

    rez = reducere_dim.aplica_ica(df, x, metadata, k=k)
    e = rez.extra

    e["entropie_kurtosis"].to_csv(OUT / "Entropie_Kurtosis_ICA.csv")
    pd.DataFrame(rez.scoruri, columns=[f"IC{i+1}" for i in range(k)]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_ICA.csv", index=False)

    ek = e["entropie_kurtosis"]
    grafice.plot_bar(ek["Kurtosis"].values, ek.index.tolist(),
                     "Kurtosis_ICA.pdf", titlu="Kurtosis pe componente ICA (non-gaussianitate)",
                     x_label="Componentă independentă", y_label="Kurtosis (Fisher)")
    grafice.plot_bar(ek["Entropie"].values, ek.index.tolist(),
                     "Entropie_ICA.pdf", titlu="Entropia diferențială pe componente ICA",
                     x_label="Componentă independentă", y_label="Entropie")
    for by in ["artist", "stil", "epoca", "gen"]:
        grafice.f_scatter_picturi(rez.scoruri, metadata, by=by,
                                  fisier=f"Scatter_ICA_{by}.pdf",
                                  titlu=f"ICA — scatter pe {by}")
    paths = metadata["path"].values
    n_show = min(4, rez.scoruri.shape[1])
    abs_sc = np.abs(rez.scoruri)
    for c in range(n_show):
        idx_top = np.argsort(-abs_sc[:, c])[:5]
        titluri = [f"IC{c+1} |s|={abs_sc[i, c]:.2f}" for i in idx_top]
        grafice.plot_poze_picturi([paths[i] for i in idx_top], titluri,
                                  f"Top_picturi_IC{c+1}.pdf",
                                  suptitlu=f"ICA componenta {c+1} — top 5 picturi cu activare maximă",
                                  n_cols=5)
    grafice.show()


if __name__ == "__main__":
    main()
