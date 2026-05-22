"""Pas 1: PCA — varianță, eigenpicturi semantice, corelograma, scatter 4 coloraje + 3D."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "pca"


def _citeste():
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    return df, x, metadata


def _construieste_validitate(e, n_total, n_features, n_max_calculat):
    """Tabel de validitate PCA: criterii standard + interpretare."""
    var_first2 = float(e["varianta_ratio"][:2].sum()) * 100
    var_first3 = float(e["varianta_ratio"][:3].sum()) * 100

    kaiser_val = e["n_kaiser"]
    if kaiser_val >= n_max_calculat - 5:
        kaiser_interp = (f"ATENȚIE: {kaiser_val} ≈ n_max ({n_max_calculat}) → Kaiser inaplicabil pe features CNN "
                         f"(sute de componente au λ>1); folosiți Elbow ca criteriu operațional")
    else:
        kaiser_interp = "Componente cu varianță peste medie (eigenvalue > 1)"

    n80_val = e["n_80"]
    n80_interp = ("Nr. minim pentru a reține 80% din info"
                  if n80_val <= n_max_calculat
                  else f"Depășește n_max calculat ({n_max_calculat}) — run PCA cu mai multe componente")

    rows = [
        {"Criteriu": "Kaiser (eigenvalue > 1)",
         "Valoare": f"{kaiser_val} componente",
         "Interpretare": kaiser_interp},
        {"Criteriu": "Prag 80% varianță cumulativă",
         "Valoare": f"{n80_val} componente",
         "Interpretare": n80_interp},
        {"Criteriu": "Elbow scree",
         "Valoare": f"{e['n_elbow']} componente",
         "Interpretare": "Punct de cot al scree plot-ului"},
        {"Criteriu": "Varianță explicată Comp1+Comp2",
         "Valoare": f"{var_first2:.2f}%",
         "Interpretare": "Cât din info se vede în scatter 2D"},
        {"Criteriu": "Varianță explicată Comp1+Comp2+Comp3",
         "Valoare": f"{var_first3:.2f}%",
         "Interpretare": "Cât din info se vede în scatter 3D"},
        {"Criteriu": "Dimensiune originală",
         "Valoare": f"{n_features} features",
         "Interpretare": "VGG16 fc2 (4096) pe {} imagini".format(n_total)},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("PCA", total=5)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv")
    df, x, metadata = _citeste()
    pasi.info(f"X = {x.shape}  →  data_out/{SUBDIR}/")

    n_max_pca = min(x.shape[1], x.shape[0] - 1, 500)
    pasi.pas(f"Fit PCA (n_max={n_max_pca} componente)")
    rez = reducere_dim.aplica_pca(df, x, metadata, n_max=n_max_pca)
    e = rez.extra
    kaiser_note = " (≈ toate comp. au λ>1, critic Kaiser inaplicabil pe CNN)" if e["n_kaiser"] >= n_max_pca - 5 else ""
    pasi.info(f"Kaiser: {e['n_kaiser']}{kaiser_note} | 80%: {e['n_80']} | Elbow: {e['n_elbow']}")

    pasi.pas("Salvare scoruri / corelații / varianță")
    tabel_var = functii.tabelare_varianta(e["varianta_ratio"])
    tabel_var.to_csv(OUT / "Varianta_PCA.csv")
    pd.DataFrame(rez.scoruri,
                 columns=[f"Comp{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_PCA.csv", index=False)
    n_show = min(20, e["corelatii"].shape[1])
    pd.DataFrame(e["corelatii"][:, :n_show],
                 columns=[f"Comp{i+1}" for i in range(n_show)]
                 ).to_csv(OUT / "r_xc_PCA.csv")
    raport = pd.DataFrame({
        "Criteriu": ["Kaiser (>1)", "Prag 80% varianță", "Elbow scree"],
        "Nr componente": [e["n_kaiser"], e["n_80"], e["n_elbow"]],
    })
    raport.to_csv(OUT / "Selectie_PCA.csv", index=False)
    print(raport.to_string(index=False))

    pasi.pas("Tabel de validitate")
    df_val = _construieste_validitate(e, len(metadata), x.shape[1], n_max_pca)
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_PCA.csv")
    grafice.plot_validitate(df_val, "Validitate_PCA.pdf",
                            titlu="Validitate PCA — criterii de selecție")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (varianță, scree, corelograma, scatter 2D+3D, eigenpicturi)")
    grafice.plot_varianta(e["varianta_cum"], "Varianta_PCA.pdf",
                          n_kaiser=e["n_kaiser"], n_elbow=e["n_elbow"])
    grafice.plot_elbow(e["varianta_ratio"], k_optim=e["n_elbow"],
                       fisier="Elbow_PCA.pdf", titlu="Scree plot PCA",
                       x_label="Componentă", y_label="Procent varianță")
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
    var_z_pct = float(e["varianta_ratio"][2]) * 100
    for by in ["artist", "stil", "epoca", "gen"]:
        grafice.f_scatter_picturi(rez.scoruri, metadata, by=by,
                                  fisier=f"Scatter_PCA_{by}.pdf",
                                  titlu=f"PCA — scatter pe {by}",
                                  var_x=var_x_pct, var_y=var_y_pct)
        grafice.f_scatter_picturi_3d(rez.scoruri, metadata, by=by,
                                      fisier=f"Scatter3D_PCA_{by}.pdf",
                                      titlu=f"PCA 3D — primele 3 componente colorate pe {by}",
                                      var_x=var_x_pct, var_y=var_y_pct, var_z=var_z_pct)
        grafice.f_scatter_interactiv_2d(rez.scoruri, metadata, by=by,
                                         fisier=f"Scatter_PCA_{by}.html",
                                         titlu=f"PCA — scatter 2D pe {by}")
        if rez.scoruri.shape[1] >= 3:
            grafice.f_scatter_interactiv_3d(rez.scoruri, metadata, by=by,
                                             fisier=f"Scatter3D_PCA_{by}.html",
                                             titlu=f"PCA 3D — primele 3 componente pe {by}")
    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
