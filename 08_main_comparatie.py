"""
Pas 8: comparație cross-metode. Citește scorurile din subfolderele fiecărei metode
(data_out/pca/, data_out/fa/, etc.) și produce:
- scatter grid 2D (toate metodele alăturate, colorat pe pictor)
- Silhouette score per metodă (proxy de cluster separability)
- Trustworthiness per metodă (menținere structură locală)
- tabel timpi de execuție (re-rulare rapidă pe sub-eșantion 1000 instanțe)
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "comparatie"


def _incarca_scoruri(subdir, nume_csv, cols_score):
    """Încarcă un CSV de scoruri din data_out/<subdir>/."""
    p = functii.DATA_OUT / subdir / nume_csv
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    coords = df[cols_score].values
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    return coords, meta


def main():
    pasi = functii.Pasi("Comparație cross-metode", total=4)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire scoruri din subfolderele metodelor")
    panouri_load = [
        ("pca",  "Scoruri_PCA.csv",         ["Comp1", "Comp2"], "PCA"),
        ("fa",   "Scoruri_FA.csv",           ["F1", "F2"],       "FA"),
        ("nmf",  "W_NMF.csv",               ["C1", "C2"],       "NMF (W)"),
        ("ica",  "Scoruri_ICA.csv",          ["IC1", "IC2"],     "ICA"),
        ("kpca", "Scoruri_KPCA.csv",         ["K1", "K2"],       "KPCA RBF"),
        ("tsne", "Scoruri_tSNE_perp30.csv",  ["t1", "t2"],       "t-SNE perp=30"),
    ]
    panouri = []
    scoruri_incarcate = {}
    for sub, csv, cols, nume in panouri_load:
        coords, meta = _incarca_scoruri(sub, csv, cols)
        if coords is None:
            print(f"[warn] lipsă data_out/{sub}/{csv} — rulează întâi main-ul aferent.")
            continue
        panouri.append((coords, meta, "artist", nume))
        scoruri_incarcate[nume] = (coords, meta)
        pasi.info(f"Încărcat: {nume} ({coords.shape})")

    # MDS: dimensiune diferită (50 pictori)
    p_mds = functii.DATA_OUT / "mds" / "Scoruri_MDS.csv"
    if p_mds.exists():
        df_mds = pd.read_csv(p_mds, index_col=0)
        cols_mds = [c for c in df_mds.columns if c.startswith("MDS")][:2]
        if len(cols_mds) >= 2:
            coords_mds = df_mds[cols_mds].values
            meta_mds = pd.DataFrame({c: (df_mds.index if c == "artist" else "")
                                     for c in META_COLS})
            meta_mds["artist"] = df_mds.index.tolist()
            panouri.append((coords_mds, meta_mds, "artist", "MDS (pictori)"))
            pasi.info(f"Încărcat: MDS ({coords_mds.shape})")

    pasi.pas("Calcul metrici: Silhouette + Trustworthiness")
    df_feat = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata_full = df_feat[META_COLS].copy()
    x_full = df_feat.drop(columns=META_COLS).values.astype(np.float32)
    x_std_full = StandardScaler().fit_transform(x_full)
    n_full = len(x_std_full)

    silhouette_rez = []
    trust_rez = []
    for sub, csv, cols, nume in panouri_load:
        coords, meta = _incarca_scoruri(sub, csv, cols)
        if coords is None:
            continue

        # Silhouette
        try:
            n_sil = min(3000, len(coords))
            idx_sil = np.random.RandomState(42).choice(len(coords), n_sil, replace=False)
            sil = silhouette_score(coords[idx_sil], meta["artist"].iloc[idx_sil],
                                   metric="euclidean")
            silhouette_rez.append({"Metoda": nume, "Silhouette_artist_2D": round(float(sil), 4)})
        except Exception as ex:
            print(f"[warn] silhouette pentru {nume}: {ex}")
            sil = float("nan")

        # Trustworthiness — doar pentru metode care acoperă tot setul (len(coords) == n_full)
        if len(coords) == n_full:
            try:
                n_trust = min(500, n_full)
                idx_trust = np.random.RandomState(42).choice(n_full, n_trust, replace=False)
                tw = trustworthiness(x_std_full[idx_trust], coords[idx_trust], n_neighbors=12)
                trust_rez.append({"Metoda": nume,
                                  "Trustworthiness_12nn": round(float(tw), 4)})
                pasi.info(f"{nume}: Silhouette={sil:.3f}, Trustworthiness={tw:.3f}")
            except Exception as ex:
                print(f"[warn] trustworthiness pentru {nume}: {ex}")
                pasi.info(f"{nume}: Silhouette={sil:.3f}, Trustworthiness=N/A")
        else:
            pasi.info(f"{nume}: Silhouette={sil:.3f}, Trustworthiness=skipped (sub-eșantion)")

    pasi.pas("Salvare tabele + grafice metrice")
    if silhouette_rez:
        df_sil = pd.DataFrame(silhouette_rez)
        df_sil.to_csv(OUT / "Silhouette_comparatie.csv", index=False)
        print(df_sil.to_string(index=False))
        grafice.plot_bar(df_sil["Silhouette_artist_2D"].values,
                         df_sil["Metoda"].tolist(),
                         "Silhouette_comparatie.pdf",
                         titlu="Silhouette score pe label 'artist' (2D, per metodă)",
                         x_label="Metoda", y_label="Silhouette")

    if trust_rez:
        df_trust = pd.DataFrame(trust_rez)
        df_trust.to_csv(OUT / "Trustworthiness_comparatie.csv", index=False)
        print(df_trust.to_string(index=False))
        grafice.plot_bar(df_trust["Trustworthiness_12nn"].values,
                         df_trust["Metoda"].tolist(),
                         "Trustworthiness_comparatie.pdf",
                         titlu="Trustworthiness (12-NN) per metodă — menținere structură locală",
                         x_label="Metoda", y_label="Trustworthiness")

    if panouri:
        grafice.plot_scatter_grid(panouri, "Comparatie_metode_scatter.pdf",
                                   suptitlu="Comparație cross-metode — scatter 2D colorat pe pictor")

    pasi.pas("Timpi de execuție (sub-eșantion 1000 instanțe)")
    n = min(1000, len(x_full))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(x_full), n, replace=False)
    x_s = x_full[idx]
    meta_s = metadata_full.iloc[idx].reset_index(drop=True)
    timpi = []
    for nume, fn in [
        ("PCA",      lambda: reducere_dim.aplica_pca(None, x_s, meta_s, n_max=20)),
        ("FA",       lambda: reducere_dim.aplica_fa(None, x_s, meta_s, n_factori=10)),
        ("NMF",      lambda: reducere_dim.aplica_nmf(None, x_s, meta_s, q_list=(10, 20))),
        ("ICA",      lambda: reducere_dim.aplica_ica(None, x_s, meta_s, k=10)),
        ("KPCA-RBF", lambda: reducere_dim.aplica_kpca(None, x_s, meta_s, n_components=10)),
        ("t-SNE",    lambda: reducere_dim.aplica_tsne(None, x_s, meta_s,
                                                      perplexity_list=(30,), n_pca=50)),
    ]:
        t0 = time.time()
        try:
            fn()
            timpi.append({"Metoda": nume, "Timp_s_pe_1000": round(time.time() - t0, 2)})
            pasi.info(f"{nume}: {time.time() - t0:.1f}s")
        except Exception as ex:
            timpi.append({"Metoda": nume, "Timp_s_pe_1000": f"ERR {ex}"})

    df_timpi = pd.DataFrame(timpi)
    df_timpi.to_csv(OUT / "Timpi_executie.csv", index=False)
    print(df_timpi.to_string(index=False))

    timp_numeric = []
    for t in df_timpi["Timp_s_pe_1000"]:
        try:
            timp_numeric.append(float(t))
        except (ValueError, TypeError):
            timp_numeric.append(0.0)
    grafice.plot_bar(np.array(timp_numeric), df_timpi["Metoda"].tolist(),
                     "Timpi_executie.pdf",
                     titlu="Timp execuție per metodă (1000 instanțe)",
                     x_label="Metoda", y_label="Timp (s)")

    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
