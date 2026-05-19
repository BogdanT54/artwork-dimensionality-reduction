"""Pas 3: NMF cu Elbow pe eroare reconstrucție + componente ca picturi semantice + 3D."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "nmf"


def _construieste_validitate(e, x_shape, x_min):
    erori = e["erori"]
    q_list = e["q_list"]
    q_optim = e["q_optim"]
    err_optim = erori[q_list.index(q_optim)]
    err_norm = err_optim / np.sqrt(x_shape[0] * x_shape[1])
    rows = [
        {"Criteriu": "Verificare pozitivitate input",
         "Valoare": f"min = {x_min:.4f}",
         "Interpretare": "OK (≥0)" if x_min >= 0 else "PROBLEMĂ: NMF cere X≥0"},
        {"Criteriu": "q optim (Elbow)",
         "Valoare": f"{q_optim} componente",
         "Interpretare": "Punct cot al curbei eroare-vs-q"},
        {"Criteriu": "Eroare Frobenius @ q_optim",
         "Valoare": f"{err_optim:.2f}",
         "Interpretare": "||X − WH||_F (mai mic = reconstrucție bună)"},
        {"Criteriu": "Eroare normalizată",
         "Valoare": f"{err_norm:.4f}",
         "Interpretare": "Eroare / √(n·m), pentru comparație"},
        {"Criteriu": "Eroare q=min vs q=max",
         "Valoare": f"{erori[0]:.0f} → {erori[-1]:.0f}",
         "Interpretare": "Scădere cu mai multe componente"},
        {"Criteriu": "Sparsity W (% zerouri)",
         "Valoare": f"{100 * np.mean(e['rezultate_per_q'][q_optim]['W'] == 0):.1f}%",
         "Interpretare": "NMF produce reprezentări rare (parte-întreg)"},
        {"Criteriu": "Dimensiune X",
         "Valoare": f"{x_shape[0]} × {x_shape[1]}",
         "Interpretare": "Imagini × features"},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("NMF", total=6)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv")
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    pasi.info(f"X = {x.shape}, min = {x.min():.4f}  →  data_out/{SUBDIR}/")

    q_list = [5, 10, 15, 20, 30, 50]
    pasi.pas(f"Fit NMF pentru q ∈ {q_list} (poate dura 2-5 min)")
    rez = reducere_dim.aplica_nmf(df, x, metadata, q_list=q_list)
    e = rez.extra
    pasi.info(f"Erori per q: {[f'{err:.0f}' for err in e['erori']]}")
    pasi.info(f"q optim Elbow = {e['q_optim']}")

    pasi.pas("Salvare W, H, erori")
    pd.DataFrame({"q": e["q_list"], "Eroare_Frobenius": e["erori"]}
                 ).to_csv(OUT / "Erori_NMF.csv", index=False)
    W = e["rezultate_per_q"][e["q_optim"]]["W"]
    H = e["H_optim"]
    pd.DataFrame(W, columns=[f"C{i+1}" for i in range(W.shape[1])]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "W_NMF.csv", index=False)
    pd.DataFrame(H, index=[f"C{i+1}" for i in range(H.shape[0])]
                 ).to_csv(OUT / "H_NMF.csv")

    pasi.pas("Tabel de validitate")
    df_val = _construieste_validitate(e, x.shape, float(x.min()))
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_NMF.csv")
    grafice.plot_validitate(df_val, "Validitate_NMF.pdf",
                            titlu="Validitate NMF — Elbow, eroare reconstrucție")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (Elbow, componente picturi, scatter 2D+3D)")
    grafice.plot_elbow(e["erori"], k_optim=e["q_list"].index(e["q_optim"]) + 1,
                       fisier="Elbow_NMF.pdf", titlu="Eroare reconstrucție NMF",
                       x_label="Index q", y_label="||X - WH||_F")
    grafice.plot_componente_nmf(W, metadata["path"].values,
                                "Componente_NMF.pdf",
                                n_comp=min(8, W.shape[1]), k=5)
    for by in ["artist", "stil", "epoca", "gen"]:
        grafice.f_scatter_picturi(W, metadata, by=by,
                                  fisier=f"Scatter_NMF_{by}.pdf",
                                  titlu=f"NMF (q={e['q_optim']}) — scatter pe {by}")
        if W.shape[1] >= 3:
            grafice.f_scatter_picturi_3d(W, metadata, by=by,
                                          fisier=f"Scatter3D_NMF_{by}.pdf",
                                          titlu=f"NMF 3D (q={e['q_optim']}) — pe {by}")
        grafice.f_scatter_interactiv_2d(W, metadata, by=by,
                                         fisier=f"Scatter_NMF_{by}.html",
                                         titlu=f"NMF (q={e['q_optim']}) — scatter 2D pe {by}")
        if W.shape[1] >= 3:
            grafice.f_scatter_interactiv_3d(W, metadata, by=by,
                                             fisier=f"Scatter3D_NMF_{by}.html",
                                             titlu=f"NMF 3D (q={e['q_optim']}) — pe {by}")
    pasi.pas("Corelograma H matrix (corelații features-componente NMF)")
    # H are forma (q, n_features); selectăm top-80 features cu cea mai mare variație în H
    top_var_H = np.argsort(-H.var(axis=0))[:80]
    H_sub = H[:, top_var_H]
    grafice.corelograma(
        pd.DataFrame(H_sub,
                     index=[f"C{i+1}" for i in range(H.shape[0])],
                     columns=[f"f{j+1}" for j in top_var_H]),
        "H_Matrix_NMF.pdf",
        "NMF: matricea H (top 80 features cu variație maximă între componente)",
        vmin=float(H_sub.min()), vmax=float(H_sub.max()), cmap="YlOrRd"
    )
    pasi.info(f"H matrix corelograma: {H.shape[0]}×80 features salvată")
    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
