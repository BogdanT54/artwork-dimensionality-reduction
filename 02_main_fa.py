"""Pas 2: Analiză factorială cu rotație Varimax + Bartlett + KMO + comunalități."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "fa"


def main():
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)

    var_per_feat = x.var(axis=0)
    top_feat = np.argsort(-var_per_feat)[:500]
    x_top = x[:, top_feat]
    print(f"[info] FA pe top-500 features cu cea mai mare varianță; X = {x_top.shape}  →  data_out/{SUBDIR}/")

    rez = reducere_dim.aplica_fa(df, x_top, metadata, n_factori=None)
    e = rez.extra
    bk = e["bartlett_kmo"]

    raport = pd.DataFrame({
        "Test": ["Bartlett chi2", "Bartlett p-value", "KMO total", "Nr factori"],
        "Valoare": [bk["chi2"], bk["p_value"], bk["kmo_total"], e["n_factori"]],
    })
    raport.to_csv(OUT / "Bartlett_KMO_FA.csv", index=False)
    print(raport.to_string(index=False))

    pd.DataFrame(e["comunalitati"], index=[f"f{i+1}" for i in top_feat],
                 columns=["Comunalitate"]).to_csv(OUT / "Comunalitati_FA.csv")

    pd.DataFrame(e["loadings"],
                 index=[f"f{i+1}" for i in top_feat],
                 columns=[f"F{i+1}" for i in range(e["n_factori"])]
                 ).to_csv(OUT / "Incarcare_FA.csv")

    pd.DataFrame(rez.scoruri,
                 columns=[f"F{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_FA.csv", index=False)

    var_arr = np.array(e["varianta"])
    pd.DataFrame({
        "Varianta": var_arr[0],
        "Proportie": var_arr[1],
        "Cumulativ": var_arr[2],
    }, index=[f"F{i+1}" for i in range(e["n_factori"])]).to_csv(OUT / "Varianta_FA.csv")

    top_com = np.argsort(-e["comunalitati"])[:40]
    grafice.plot_bar(e["comunalitati"][top_com], [f"f{i+1}" for i in top_feat[top_com]],
                     "Comunalitati_FA.pdf", titlu="Comunalități FA (top 40 features)",
                     x_label="Feature", y_label="h²")
    grafice.corelograma(e["loadings"][top_com], "Loadings_FA.pdf",
                        "Loadings FA (top 40 features × factori)", vmin=-1, vmax=1)
    for by in ["artist", "stil", "epoca", "gen"]:
        grafice.f_scatter_picturi(rez.scoruri, metadata, by=by,
                                  fisier=f"Scatter_FA_{by}.pdf",
                                  titlu=f"FA — scatter pe {by}")
    grafice.show()


if __name__ == "__main__":
    main()
