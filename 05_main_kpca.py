"""Pas 5: Kernel PCA (RBF) vs PCA liniar (comparație side-by-side) + 3D."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "kpca"


def _construieste_validitate(rez, x_s_shape, n_comp):
    e = rez.extra
    ev = e.get("eigenvalues")
    gamma_ef = e.get("gamma_efectiv", float("nan"))

    if ev is not None and len(ev) >= 2 and ev.sum() > 0:
        ev_total = float(ev.sum())
        var_2d = float(ev[:2].sum()) / ev_total * 100
        top_ev = float(ev[0])
    else:
        var_2d = float("nan")
        top_ev = float("nan")

    rows = [
        {"Criteriu": "Kernel",
         "Valoare": e["kernel"],
         "Interpretare": "RBF: mapare în spațiu cu dimensiune infinită (Gaussian)"},
        {"Criteriu": "Nr componente",
         "Valoare": f"{n_comp}",
         "Interpretare": "Dimensiuni reținute în spațiul kernel"},
        {"Criteriu": "Sub-eșantion",
         "Valoare": f"{x_s_shape[0]} imagini",
         "Interpretare": "KPCA scalează O(n²) → sub-eșantionare stratificată per pictor"},
        {"Criteriu": "Gamma efectiv (RBF)",
         "Valoare": f"{gamma_ef:.2e}",
         "Interpretare": "1/n_features implicit; controlează lărgimea kernelului Gaussian"},
        {"Criteriu": "Varianță explicată Comp1+2 (kernel)",
         "Valoare": f"{var_2d:.2f}%" if not np.isnan(var_2d) else "N/A",
         "Interpretare": "% din varianța totală în spațiul kernel (eigenvalue ratio)"},
        {"Criteriu": "Top eigenvalue (kernel)",
         "Valoare": f"{top_ev:.2f}" if not np.isnan(top_ev) else "N/A",
         "Interpretare": "Importanța primei componente kernel"},
        {"Criteriu": "Comparare cu PCA liniar",
         "Valoare": "side-by-side",
         "Interpretare": "Aceleași date, PCA liniar vs KPCA RBF — diferențe de separare"},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("KPCA", total=5)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv + sub-eșantionare stratificată")
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    n = len(metadata)
    if n > 2000:
        sub_idx = (metadata.groupby("artist", group_keys=False)
                   .apply(lambda g: g.sample(
                       min(len(g), max(1, 2000 // metadata["artist"].nunique())),
                       random_state=42),
                          include_groups=False)
                   ).index.values
        sub_idx = np.array(sub_idx)
    else:
        sub_idx = np.arange(n)
    x_s = x[sub_idx]
    meta_s = metadata.iloc[sub_idx].reset_index(drop=True)
    pasi.info(f"Sub-eșantion KPCA: {x_s.shape}  →  data_out/{SUBDIR}/")

    pasi.pas("Fit Kernel PCA (RBF, n_components=20)")
    rez = reducere_dim.aplica_kpca(df, x_s, meta_s, kernel="rbf", n_components=20)
    e = rez.extra
    ev = e.get("eigenvalues")
    if ev is not None and ev.sum() > 0:
        pasi.info(f"Top eigenvalue = {ev[0]:.2f}, Comp1+2 = {ev[:2].sum()/ev.sum()*100:.1f}%")
    else:
        pasi.info("Eigenvalues indisponibile pentru această versiune sklearn")

    pasi.pas("Salvare scoruri KPCA + PCA referință")
    pd.DataFrame(rez.scoruri, columns=[f"K{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).assign(**{c: meta_s[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_KPCA.csv", index=False)
    pd.DataFrame(e["scoruri_pca_referinta"], columns=[f"C{i+1}" for i in range(20)]
                 ).assign(**{c: meta_s[c] for c in META_COLS}).to_csv(
        OUT / "Scoruri_PCA_referinta_KPCA.csv", index=False)

    pasi.pas("Tabel de validitate")
    df_val = _construieste_validitate(rez, x_s.shape, 20)
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_KPCA.csv")
    grafice.plot_validitate(df_val, "Validitate_KPCA.pdf",
                            titlu="Validitate KPCA — kernel RBF vs PCA liniar")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (eigenvalues, comparație 2D PCA vs KPCA, scatter 3D)")
    if ev is not None and len(ev) > 0:
        n_ev = min(20, len(ev))
        grafice.plot_bar(ev[:n_ev].astype(float),
                         [f"K{i+1}" for i in range(n_ev)],
                         "Eigenvalues_KPCA.pdf",
                         titlu="Spectrul valorilor proprii Kernel PCA (RBF)",
                         x_label="Componentă kernel", y_label="Eigenvalue")

    for by in ["artist", "stil", "epoca", "gen"]:
        # Comparație side-by-side 2D: PCA liniar vs KPCA
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        for ax, scoruri_ref, titlu_ref in [
            (ax1, e["scoruri_pca_referinta"], "PCA liniar"),
            (ax2, rez.scoruri, "Kernel PCA (RBF)"),
        ]:
            categorii = sorted(meta_s[by].dropna().unique().tolist())
            culori = grafice.generare_culori(len(categorii))
            for j, cat in enumerate(categorii):
                mask = (meta_s[by] == cat).values
                ax.scatter(scoruri_ref[mask, 0], scoruri_ref[mask, 1],
                           color=culori[j], alpha=0.6, s=14, edgecolors="none",
                           label=str(cat))
            ax.set_title(f"{titlu_ref} — colorat pe {by}")
            ax.set_xlabel("Comp1")
            ax.set_ylabel("Comp2")
            ax.grid(True, alpha=0.3)
            if len(categorii) <= 30:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
        plt.tight_layout()
        plt.savefig(OUT / f"Comparatie_KPCA_PCA_{by}.pdf",
                    format="pdf", bbox_inches="tight")
        plt.close(fig)

        # 3D scatter KPCA
        if rez.scoruri.shape[1] >= 3:
            grafice.f_scatter_picturi_3d(rez.scoruri, meta_s, by=by,
                                          fisier=f"Scatter3D_KPCA_{by}.pdf",
                                          titlu=f"KPCA 3D (RBF) — primele 3 componente pe {by}")
        # 3D scatter PCA referință
        if e["scoruri_pca_referinta"].shape[1] >= 3:
            grafice.f_scatter_picturi_3d(e["scoruri_pca_referinta"], meta_s, by=by,
                                          fisier=f"Scatter3D_PCA_ref_{by}.pdf",
                                          titlu=f"PCA liniar 3D (ref KPCA) — pe {by}")
        grafice.f_scatter_interactiv_2d(rez.scoruri, meta_s, by=by,
                                         fisier=f"Scatter_KPCA_{by}.html",
                                         titlu=f"KPCA RBF — scatter 2D pe {by}")
        if rez.scoruri.shape[1] >= 3:
            grafice.f_scatter_interactiv_3d(rez.scoruri, meta_s, by=by,
                                             fisier=f"Scatter3D_KPCA_{by}.html",
                                             titlu=f"KPCA RBF 3D pe {by}")
        grafice.f_scatter_interactiv_2d(e["scoruri_pca_referinta"], meta_s, by=by,
                                         fisier=f"Scatter_PCA_ref_{by}.html",
                                         titlu=f"PCA liniar (ref KPCA) — 2D pe {by}")
        if e["scoruri_pca_referinta"].shape[1] >= 3:
            grafice.f_scatter_interactiv_3d(e["scoruri_pca_referinta"], meta_s, by=by,
                                             fisier=f"Scatter3D_PCA_ref_{by}.html",
                                             titlu=f"PCA liniar (ref KPCA) — 3D pe {by}")
    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
