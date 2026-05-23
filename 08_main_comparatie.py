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
        ("tsne", "Scoruri_tSNE_perp50.csv", ["t1", "t2"], "t-SNE perp=50"),
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

    pasi.pas("Calcul metrici: Silhouette (artist/stil/epoca/gen) + Trustworthiness")
    df_feat = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata_full = df_feat[META_COLS].copy()
    x_full = df_feat.drop(columns=META_COLS).values.astype(np.float32)
    x_std_full = StandardScaler().fit_transform(x_full)
    n_full = len(x_std_full)

    LABELS_SIL = ["artist", "stil", "epoca", "gen"]
    silhouette_rez = []
    trust_rez = []
    for sub, csv, cols, nume in panouri_load:
        coords, meta = _incarca_scoruri(sub, csv, cols)
        if coords is None:
            continue

        # Silhouette pe fiecare etichetă disponibilă în metadata
        n_sil = min(3000, len(coords))
        idx_sil = np.random.RandomState(42).choice(len(coords), n_sil, replace=False)
        row_sil = {"Metoda": nume}
        sil_msg_parts = []
        for lbl in LABELS_SIL:
            if lbl not in meta.columns:
                row_sil[f"Silhouette_{lbl}"] = float("nan")
                continue
            labels_arr = meta[lbl].iloc[idx_sil]
            # Silhouette necesită ≥2 clase și ≥2 samples per clasă
            n_classes = labels_arr.dropna().nunique()
            if n_classes < 2:
                row_sil[f"Silhouette_{lbl}"] = float("nan")
                continue
            try:
                valid_mask = labels_arr.notna().values
                if valid_mask.sum() < 10:
                    row_sil[f"Silhouette_{lbl}"] = float("nan")
                    continue
                sil_lbl = silhouette_score(
                    coords[idx_sil][valid_mask],
                    labels_arr[valid_mask],
                    metric="euclidean",
                )
                row_sil[f"Silhouette_{lbl}"] = round(float(sil_lbl), 4)
                sil_msg_parts.append(f"{lbl}={sil_lbl:.3f}")
            except Exception as ex:
                print(f"[warn] silhouette {nume}/{lbl}: {ex}")
                row_sil[f"Silhouette_{lbl}"] = float("nan")
        silhouette_rez.append(row_sil)

        # Trustworthiness — doar pentru metode care acoperă tot setul (len(coords) == n_full)
        if len(coords) == n_full:
            try:
                n_trust = min(500, n_full)
                idx_trust = np.random.RandomState(42).choice(n_full, n_trust, replace=False)
                tw = trustworthiness(x_std_full[idx_trust], coords[idx_trust], n_neighbors=12)
                trust_rez.append({"Metoda": nume,
                                  "Trustworthiness_12nn": round(float(tw), 4)})
                pasi.info(f"{nume}: Sil[{', '.join(sil_msg_parts)}], Trust={tw:.3f}")
            except Exception as ex:
                print(f"[warn] trustworthiness pentru {nume}: {ex}")
                pasi.info(f"{nume}: Sil[{', '.join(sil_msg_parts)}], Trust=N/A")
        else:
            pasi.info(f"{nume}: Sil[{', '.join(sil_msg_parts)}], Trust=skipped (sub-eșantion)")

    pasi.pas("Salvare tabele + grafice metrice")
    if silhouette_rez:
        df_sil = pd.DataFrame(silhouette_rez)
        df_sil.to_csv(OUT / "Silhouette_comparatie.csv", index=False)
        print(df_sil.to_string(index=False))

        # Grouped bar chart: 4 bare per metodă (artist, stil, epoca, gen)
        metode_nume = df_sil["Metoda"].tolist()
        n_metode = len(metode_nume)
        x_pos = np.arange(n_metode)
        bar_w = 0.20
        culori_lbl = {"artist": "#1f77b4", "stil": "#ff7f0e", "epoca": "#2ca02c", "gen": "#d62728"}

        fig, ax = plt.subplots(figsize=(max(10, n_metode * 1.5), 6))
        for i, lbl in enumerate(LABELS_SIL):
            col = f"Silhouette_{lbl}"
            if col not in df_sil.columns:
                continue
            vals = df_sil[col].values.astype(float)
            offset = (i - (len(LABELS_SIL) - 1) / 2) * bar_w
            bars = ax.bar(x_pos + offset, np.nan_to_num(vals, nan=0.0),
                          bar_w, label=lbl, color=culori_lbl.get(lbl, None),
                          edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.005 if v >= 0 else -0.02),
                            f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metode_nume, rotation=20, ha="right")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette score per metodă × etichetă (artist / stil / epocă / gen, 2D)")
        ax.legend(title="Etichetă", loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(OUT / "Silhouette_comparatie.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

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
