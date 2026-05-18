"""
Pas 8: comparație cross-metode. Citește scorurile salvate de fiecare main și produce:
- scatter grid 2D (toate metodele alăturate, colorat pe pictor)
- tabel timpi de execuție (re-rulează rapid fiecare aplica_* pe sub-eșantion)
- Silhouette score per metodă pe label pictor (cluster separability)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]


def _incarca_scoruri(nume_csv, cols_score):
    """Încarcă un CSV de scoruri și separă coordonatele de metadata."""
    p = functii.DATA_OUT / nume_csv
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    coords = df[cols_score].values
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    return coords, meta


def main():
    # NU goleste data_out — vrem să folosim outputurile metodelor anterioare!
    functii.DATA_OUT.mkdir(parents=True, exist_ok=True)

    panouri = []
    panouri_load = [
        ("Scoruri_PCA.csv",  ["Comp1", "Comp2"],            "PCA"),
        ("Scoruri_FA.csv",   ["F1", "F2"],                  "FA"),
        ("W_NMF.csv",        ["C1", "C2"],                  "NMF (W)"),
        ("Scoruri_ICA.csv",  ["IC1", "IC2"],                "ICA"),
        ("Scoruri_KPCA.csv", ["K1", "K2"],                  "KPCA RBF"),
        ("Scoruri_tSNE_perp30.csv", ["t1", "t2"],           "t-SNE perp=30"),
    ]
    silhouette_rez = []
    for csv, cols, nume in panouri_load:
        coords, meta = _incarca_scoruri(csv, cols)
        if coords is None:
            print(f"[warn] lipsă {csv} — rulează întâi main-ul aferent.")
            continue
        panouri.append((coords, meta, "artist", nume))
        # silhouette pe label "artist" — proxy de cluster separability
        try:
            sample = np.random.RandomState(42).choice(len(coords), min(3000, len(coords)),
                                                       replace=False)
            sil = silhouette_score(coords[sample], meta["artist"].iloc[sample], metric="euclidean")
            silhouette_rez.append({"Metoda": nume, "Silhouette_artist_2D": sil})
        except Exception as ex:
            print(f"[warn] silhouette pentru {nume}: {ex}")

    # MDS: dimensiune diferită (50 pictori); îl includem separat dacă există
    p_mds = functii.DATA_OUT / "Scoruri_MDS.csv"
    if p_mds.exists():
        df_mds = pd.read_csv(p_mds, index_col=0)
        coords_mds = df_mds[["MDS1", "MDS2"]].values
        meta_mds = pd.DataFrame({c: df_mds.index for c in META_COLS})
        meta_mds["artist"] = df_mds.index
        panouri.append((coords_mds, meta_mds, "artist", "MDS (pictori)"))

    if panouri:
        grafice.plot_scatter_grid(panouri, "Comparatie_metode_scatter.pdf",
                                   suptitlu="Comparație cross-metode — scatter 2D colorat pe pictor")

    if silhouette_rez:
        df_sil = pd.DataFrame(silhouette_rez)
        df_sil.to_csv(functii.DATA_OUT / "Silhouette_comparatie.csv", index=False)
        print(df_sil.to_string(index=False))
        grafice.plot_bar(df_sil["Silhouette_artist_2D"].values, df_sil["Metoda"].tolist(),
                         "Silhouette_comparatie.pdf",
                         titlu="Silhouette score pe label 'artist' (2D, per metodă)",
                         x_label="Metoda", y_label="Silhouette")

    # timpi de execuție (re-rulare rapidă pe sub-eșantion 1000 instanțe)
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    n = min(1000, len(x))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(x), n, replace=False)
    x_s = x[idx]
    meta_s = metadata.iloc[idx].reset_index(drop=True)
    timpi = []
    for nume, fn in [
        ("PCA", lambda: reducere_dim.aplica_pca(None, x_s, meta_s, n_max=20)),
        ("NMF", lambda: reducere_dim.aplica_nmf(None, x_s, meta_s, q_list=(10, 20))),
        ("ICA", lambda: reducere_dim.aplica_ica(None, x_s, meta_s, k=10)),
        ("KPCA-RBF", lambda: reducere_dim.aplica_kpca(None, x_s, meta_s, n_components=10)),
        ("t-SNE", lambda: reducere_dim.aplica_tsne(None, x_s, meta_s,
                                                    perplexity_list=(30,), n_pca=50)),
    ]:
        t0 = time.time()
        try:
            fn()
            timpi.append({"Metoda": nume, "Timp_s_pe_1000": round(time.time() - t0, 2)})
        except Exception as ex:
            timpi.append({"Metoda": nume, "Timp_s_pe_1000": f"ERR {ex}"})
    df_timpi = pd.DataFrame(timpi)
    df_timpi.to_csv(functii.DATA_OUT / "Timpi_executie.csv", index=False)
    print(df_timpi.to_string(index=False))
    grafice.show()


if __name__ == "__main__":
    main()
