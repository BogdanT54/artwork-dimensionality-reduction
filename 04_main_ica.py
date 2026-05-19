"""Pas 4: FastICA + entropie + kurtosis (non-gaussianitate) + 3D."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "ica"


def _construieste_validitate(e, k, n_obs):
    ek = e["entropie_kurtosis"]
    kurt_mean = float(ek["Kurtosis"].mean())
    kurt_max = float(ek["Kurtosis"].max())
    ent_mean = float(ek["Entropie"].mean())
    n_supragaus = int((ek["Kurtosis"] > 0).sum())
    rows = [
        {"Criteriu": "Nr componente independente",
         "Valoare": f"{k}",
         "Interpretare": "Determinat din criteriul Kaiser PCA"},
        {"Criteriu": "Kurtosis medie",
         "Valoare": f"{kurt_mean:.3f}",
         "Interpretare": "Fisher: >0 supra-gaus, <0 sub-gaus, =0 gauss"},
        {"Criteriu": "Kurtosis maximă",
         "Valoare": f"{kurt_max:.3f}",
         "Interpretare": "Cea mai non-gaussiană componentă (cea mai info)"},
        {"Criteriu": "Componente supra-gaussiene",
         "Valoare": f"{n_supragaus} / {k}",
         "Interpretare": "Kurtosis > 0: surse cu vârf ascuțit"},
        {"Criteriu": "Entropie diferențială medie",
         "Valoare": f"{ent_mean:.3f}",
         "Interpretare": "Mai mică = mai non-gaussian (mai informativ)"},
        {"Criteriu": "Convergence FastICA",
         "Valoare": "fitted",
         "Interpretare": "Verifică warning-uri pentru non-convergență"},
        {"Criteriu": "Observații",
         "Valoare": f"{n_obs}",
         "Interpretare": "Imagini procesate"},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("ICA", total=5)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv + determinare k din Kaiser PCA")
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    rez_pca = reducere_dim.aplica_pca(df, x, metadata, n_max=50)
    k = min(30, max(2, rez_pca.extra["n_kaiser"]))
    pasi.info(f"k = {k} componente (Kaiser PCA={rez_pca.extra['n_kaiser']})  →  data_out/{SUBDIR}/")

    pasi.pas(f"Fit FastICA cu k={k}")
    rez = reducere_dim.aplica_ica(df, x, metadata, k=k)
    e = rez.extra
    ek = e["entropie_kurtosis"]
    pasi.info(f"Kurtosis medie = {ek['Kurtosis'].mean():.2f}, max = {ek['Kurtosis'].max():.2f}")

    pasi.pas("Salvare scoruri + entropie/kurtosis")
    ek.to_csv(OUT / "Entropie_Kurtosis_ICA.csv")
    pd.DataFrame(rez.scoruri, columns=[f"IC{i+1}" for i in range(k)]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_ICA.csv", index=False)

    pasi.pas("Tabel de validitate")
    df_val = _construieste_validitate(e, k, len(metadata))
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_ICA.csv")
    grafice.plot_validitate(df_val, "Validitate_ICA.pdf",
                            titlu="Validitate ICA — non-gaussianitate")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (kurtosis, entropie, scatter 2D+3D, top picturi)")
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
        if rez.scoruri.shape[1] >= 3:
            grafice.f_scatter_picturi_3d(rez.scoruri, metadata, by=by,
                                          fisier=f"Scatter3D_ICA_{by}.pdf",
                                          titlu=f"ICA 3D — primele 3 IC pe {by}")
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
    pasi.terminat()


if __name__ == "__main__":
    main()
