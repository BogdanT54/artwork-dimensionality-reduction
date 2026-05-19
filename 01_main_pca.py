"""Pas 1: PCA — varianță, eigenpicturi semantice, corelograma, scatter 4 coloraje."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]


def _citeste():
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    return df, x, metadata


def main():
    functii.goleste_data_out(tokens=["_PCA.", "Selectie_PCA", "Cercul_PCA",
                                       "Eigenpicturi_PCA", "Elbow_PCA",
                                       "Corelograma_PCA", "Varianta_PCA",
                                       "Scoruri_PCA.", "r_xc_PCA", "Scatter_PCA"])
    df, x, metadata = _citeste()
    print(f"[info] PCA pe X = {x.shape}")

    rez = reducere_dim.aplica_pca(df, x, metadata, n_max=150)
    e = rez.extra

    # tabele
    tabel_var = functii.tabelare_varianta(e["varianta_ratio"])
    tabel_var.to_csv(functii.DATA_OUT / "Varianta_PCA.csv")

    pd.DataFrame(rez.scoruri,
                 columns=[f"Comp{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        functii.DATA_OUT / "Scoruri_PCA.csv", index=False)

    n_show = min(20, e["corelatii"].shape[1])
    pd.DataFrame(e["corelatii"][:, :n_show],
                 columns=[f"Comp{i+1}" for i in range(n_show)]
                 ).to_csv(functii.DATA_OUT / "r_xc_PCA.csv")

    raport = pd.DataFrame({
        "Criteriu": ["Kaiser (>1)", "Prag 80% varianță", "Elbow scree"],
        "Nr componente": [e["n_kaiser"], e["n_80"], e["n_elbow"]],
    })
    raport.to_csv(functii.DATA_OUT / "Selectie_PCA.csv", index=False)
    print(raport.to_string(index=False))

    # grafice
    grafice.plot_varianta(e["varianta_cum"], "Varianta_PCA.pdf",
                          n_kaiser=e["n_kaiser"], n_elbow=e["n_elbow"])
    grafice.plot_elbow(e["varianta_ratio"], k_optim=e["n_elbow"],
                       fisier="Elbow_PCA.pdf", titlu="Scree plot PCA",
                       x_label="Componentă", y_label="Procent varianță")
    # corelograma: doar pe primele 50 features cu cele mai mari norme (lizibilitate)
    norme = np.linalg.norm(e["corelatii"][:, :n_show], axis=1)
    top_idx = np.argsort(-norme)[:50]
    grafice.corelograma(e["corelatii"][top_idx, :n_show],
                        "Corelograma_PCA.pdf",
                        "Corelații features (top 50) - componente PCA")
    grafice.plot_scoruri_corelatii(e["corelatii"], "Cercul_PCA.pdf")
    grafice.plot_eigenpicturi_pca(rez.scoruri, metadata["path"].values,
                                  "Eigenpicturi_PCA.pdf", n_comp=6, k=5)
    var_x_pct = float(e["varianta_ratio"][0]) * 100
    var_y_pct = float(e["varianta_ratio"][1]) * 100
    for by in ["artist", "stil", "epoca", "gen"]:
        grafice.f_scatter_picturi(rez.scoruri, metadata, by=by,
                                  fisier=f"Scatter_PCA_{by}.pdf",
                                  titlu=f"PCA — scatter pe {by}",
                                  var_x=var_x_pct, var_y=var_y_pct)
    grafice.show()


if __name__ == "__main__":
    main()
