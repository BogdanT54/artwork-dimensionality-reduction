"""Pas 3: NMF cu Elbow pe eroare reconstrucție + componente ca picturi semantice."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "nmf"


def main():
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    print(f"[info] NMF (VGG16 fc2 ≥ 0); X = {x.shape}, min = {x.min():.4f}  →  data_out/{SUBDIR}/")

    q_list = [5, 10, 15, 20, 30, 50]
    rez = reducere_dim.aplica_nmf(df, x, metadata, q_list=q_list)
    e = rez.extra

    pd.DataFrame({"q": e["q_list"], "Eroare_Frobenius": e["erori"]}
                 ).to_csv(OUT / "Erori_NMF.csv", index=False)
    print(f"[info] q optim Elbow = {e['q_optim']}")

    W = e["rezultate_per_q"][e["q_optim"]]["W"]
    H = e["H_optim"]
    pd.DataFrame(W, columns=[f"C{i+1}" for i in range(W.shape[1])]
                 ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
        OUT / "W_NMF.csv", index=False)
    pd.DataFrame(H, index=[f"C{i+1}" for i in range(H.shape[0])]
                 ).to_csv(OUT / "H_NMF.csv")

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
    grafice.show()


if __name__ == "__main__":
    main()
