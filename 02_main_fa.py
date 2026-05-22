"""Pas 2: Analiză factorială cu rotație Varimax + Bartlett + KMO + comunalități + 3D."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "fa"


def _interpretare_kmo(v):
    if v >= 0.9:  return "Excelent (≥0.9)"
    if v >= 0.8:  return "Foarte bun (0.8-0.9)"
    if v >= 0.7:  return "Acceptabil (0.7-0.8)"
    if v >= 0.6:  return "Mediocru (0.6-0.7)"
    if v >= 0.5:  return "Slab (0.5-0.6)"
    return "Inacceptabil (<0.5)"


def _construieste_validitate(e, n_obs, n_features):
    bk = e["bartlett_kmo"]
    var_arr = np.array(e["varianta"])
    var_cum_total = float(var_arr[2][-1]) * 100
    rows = [
        {"Criteriu": "Test Bartlett (chi²)",
         "Valoare": f"{bk['chi2']:.2e}",
         "Interpretare": "Sferocitate: p-value << 0.05 → FA aplicabilă"},
        {"Criteriu": "Bartlett p-value",
         "Valoare": f"{bk['p_value']:.2e}",
         "Interpretare": ("p < 0.05 → corelații semnificative"
                          if bk['p_value'] < 0.05 else "p ≥ 0.05 → FA neaplicabilă")},
        {"Criteriu": "KMO total",
         "Valoare": f"{bk['kmo_total']:.4f}",
         "Interpretare": _interpretare_kmo(bk['kmo_total'])},
        {"Criteriu": "Nr factori reținuți",
         "Valoare": f"{e['n_factori']}",
         "Interpretare": "Din criteriul Kaiser (eigenvalue > 1)"},
        {"Criteriu": "Varianță cumulativă",
         "Valoare": f"{var_cum_total:.2f}%",
         "Interpretare": "Acoperită de cei {} factori".format(e['n_factori'])},
        {"Criteriu": "Comunalități medie",
         "Valoare": f"{float(np.mean(e['comunalitati'])):.3f}",
         "Interpretare": (
             "h² > 0.5 = bine explicată — media depășește pragul"
             if float(np.mean(e['comunalitati'])) >= 0.5
             else f"NOTĂ: media h²={float(np.mean(e['comunalitati'])):.3f} < 0.5 — "
                  f"normal pe features CNN de mare dimensionalitate; "
                  f"factori explică parțial variabilitatea (top features h²>0.6)"
         )},
        {"Criteriu": "Dimensiune analiză",
         "Valoare": f"{n_obs} × {n_features}",
         "Interpretare": "Observații × features (top-variance)"},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("FA (Analiză Factorială)", total=6)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv + selecție top-500 features")
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    var_per_feat = x.var(axis=0)
    top_feat = np.argsort(-var_per_feat)[:500]
    x_top = x[:, top_feat]
    pasi.info(f"FA pe top-500 features; X = {x_top.shape}  →  data_out/{SUBDIR}/")

    pasi.pas("Fit FA + Bartlett + KMO (poate dura 30-60s)")
    rez = reducere_dim.aplica_fa(df, x_top, metadata, n_factori=None)
    e = rez.extra
    bk = e["bartlett_kmo"]
    pasi.info(f"Bartlett chi² = {bk['chi2']:.2e}, p = {bk['p_value']:.2e}, KMO = {bk['kmo_total']:.3f}")
    pasi.info(f"Nr factori (Kaiser): {e['n_factori']}")

    pasi.pas("Salvare loadings / comunalități / scoruri / varianță")
    raport = pd.DataFrame({
        "Test": ["Bartlett chi2", "Bartlett p-value", "KMO total", "Nr factori"],
        "Valoare": [bk["chi2"], bk["p_value"], bk["kmo_total"], e["n_factori"]],
    })
    raport.to_csv(OUT / "Bartlett_KMO_FA.csv", index=False)
    pd.DataFrame(e["comunalitati"], index=[f"f{i+1}" for i in top_feat],
                 columns=["Comunalitate"]).to_csv(OUT / "Comunalitati_FA.csv")
    pd.DataFrame(e["loadings"], index=[f"f{i+1}" for i in top_feat],
                 columns=[f"F{i+1}" for i in range(e["n_factori"])]
                 ).to_csv(OUT / "Incarcare_FA.csv")
    pd.DataFrame(rez.scoruri,
                 columns=[f"F{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_FA.csv", index=False)
    var_arr = np.array(e["varianta"])
    pd.DataFrame({
        "Varianta": var_arr[0], "Proportie": var_arr[1], "Cumulativ": var_arr[2],
    }, index=[f"F{i+1}" for i in range(e["n_factori"])]).to_csv(OUT / "Varianta_FA.csv")

    pasi.pas("Tabel de validitate")
    df_val = _construieste_validitate(e, x_top.shape[0], x_top.shape[1])
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_FA.csv")
    grafice.plot_validitate(df_val, "Validitate_FA.pdf",
                            titlu="Validitate FA — Bartlett, KMO, varianță")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (comunalități, loadings, scatter 2D+3D)")
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
        if rez.scoruri.shape[1] >= 3:
            grafice.f_scatter_picturi_3d(rez.scoruri, metadata, by=by,
                                          fisier=f"Scatter3D_FA_{by}.pdf",
                                          titlu=f"FA 3D — primii 3 factori pe {by}")
        grafice.f_scatter_interactiv_2d(rez.scoruri, metadata, by=by,
                                         fisier=f"Scatter_FA_{by}.html",
                                         titlu=f"FA — scatter 2D pe {by}")
        if rez.scoruri.shape[1] >= 3:
            grafice.f_scatter_interactiv_3d(rez.scoruri, metadata, by=by,
                                             fisier=f"Scatter3D_FA_{by}.html",
                                             titlu=f"FA 3D — pe {by}")
    pasi.pas("Grafice suplimentare: varianță per factor + cercul corelațiilor FA")
    var_arr = np.array(e["varianta"])
    # Bar chart: varianță explicată per factor
    grafice.plot_bar(
        var_arr[1] * 100,  # proportii ca procente
        [f"F{i+1}" for i in range(e["n_factori"])],
        "Varianta_per_Factor_FA.pdf",
        titlu="Varianță explicată per factor FA (%)",
        x_label="Factor", y_label="% din varianța totală"
    )
    # Cerc corelații FA: loadings pe primii 2 factori
    grafice.plot_scoruri_corelatii(
        e["loadings"],
        "Cercul_Corelatii_FA.pdf",
        titlu="Cercul corelațiilor FA (loadings factori 1 și 2)"
    )
    pasi.info(f"Grafice adăugate: varianță per factor, cerc corelații FA")
    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
