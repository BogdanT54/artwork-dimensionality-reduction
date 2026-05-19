"""Pas 7: t-SNE 2D cu multiple perplexity, pe primele 50 componente PCA."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "tsne"


def _construieste_validitate(e, n_obs):
    rezultate = e["rezultate_per_perp"]
    kl_values = {perp: d["kl_divergence"] for perp, d in rezultate.items()}
    best_perp = min(kl_values, key=kl_values.get)

    rows = [
        {"Criteriu": "Nr componente PCA intermediar",
         "Valoare": f"{e['n_pca_intermediar']}",
         "Interpretare": "Reducere PCA preliminară pentru eficiență (standard practice)"},
        {"Criteriu": "Perplexity testate",
         "Valoare": str(list(kl_values.keys())),
         "Interpretare": "Controlează nr vecini locali; 30-50 recomandat pentru clustere vizibile"},
        {"Criteriu": "Cel mai bun perplexity (KL minim)",
         "Valoare": f"{best_perp}",
         "Interpretare": "KL Divergence minimă = cel mai bun fit al distribuției locale"},
        *[{"Criteriu": f"KL Divergence (perp={perp})",
           "Valoare": f"{kl:.4f}",
           "Interpretare": "Mai mic = structura locală mai bine păstrată"}
          for perp, kl in kl_values.items()],
        {"Criteriu": "Observații",
         "Valoare": f"{n_obs}",
         "Interpretare": "Imagini procesate"},
        {"Criteriu": "t-SNE 3D",
         "Valoare": f"perplexity={e.get('perp_3d', 50)}",
         "Interpretare": "Calculat separat cu n_components=3; scatter 3D generat per coloraj"},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("t-SNE", total=5)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv")
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    pasi.info(f"X = {x.shape}, PCA intermediar 50 dim  →  data_out/{SUBDIR}/")

    pasi.pas("Fit t-SNE pentru perplexity ∈ {5, 30, 50, 100} (poate dura 5-15 min)")
    perplexity_list = (5, 30, 50, 100)
    rez = reducere_dim.aplica_tsne(df, x, metadata, perplexity_list=perplexity_list, n_pca=50)
    e = rez.extra
    for perp, d in e["rezultate_per_perp"].items():
        pasi.info(f"perp={perp}: KL divergence = {d['kl_divergence']:.4f}")

    pasi.pas("Salvare scoruri + KL divergence + tabel validitate")
    for perp, d in e["rezultate_per_perp"].items():
        pd.DataFrame(d["scoruri"], columns=["t1", "t2"]
                     ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
            OUT / f"Scoruri_tSNE_perp{perp}.csv", index=False)

    pd.DataFrame({
        "perplexity": list(e["rezultate_per_perp"].keys()),
        "KL_divergence": [d["kl_divergence"] for d in e["rezultate_per_perp"].values()],
    }).to_csv(OUT / "KL_tSNE.csv", index=False)

    df_val = _construieste_validitate(e, len(metadata))
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_tSNE.csv")
    grafice.plot_validitate(df_val, "Validitate_tSNE.pdf",
                            titlu="Validitate t-SNE — KL divergence, perplexity")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (grid 4×4: 4 perplexity × 4 coloraje)")
    for by in ["artist", "stil", "epoca", "gen"]:
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        for ax, perp in zip(axes.flat, perplexity_list):
            scoruri = e["rezultate_per_perp"][perp]["scoruri"]
            kl = e["rezultate_per_perp"][perp]["kl_divergence"]
            categorii = sorted(metadata[by].dropna().unique().tolist())
            culori = grafice.generare_culori(len(categorii))
            for j, cat in enumerate(categorii):
                mask = (metadata[by] == cat).values
                ax.scatter(scoruri[mask, 0], scoruri[mask, 1], color=culori[j],
                           alpha=0.6, s=12, edgecolors="none", label=str(cat))
            ax.set_title(f"t-SNE perp={perp} — {by} (KL={kl:.3f})")
            ax.set_xticks([])
            ax.set_yticks([])
            if len(categorii) <= 30:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
        plt.tight_layout()
        plt.savefig(OUT / f"Grid_tSNE_{by}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
    for perp in perplexity_list:
        scoruri_perp = e["rezultate_per_perp"][perp]["scoruri"]
        for by in ["artist", "stil", "epoca", "gen"]:
            grafice.f_scatter_interactiv_2d(scoruri_perp, metadata, by=by,
                                             fisier=f"Scatter_tSNE_perp{perp}_{by}.html",
                                             titlu=f"t-SNE perp={perp} — scatter 2D pe {by}")
    pasi.pas("Scatter 3D t-SNE (perplexity=50)")
    scoruri_3d = e.get("scoruri_3d")
    perp_3d = e.get("perp_3d", 50)
    if scoruri_3d is not None:
        pd.DataFrame(scoruri_3d, columns=["t1", "t2", "t3"]
                     ).assign(**{c: metadata[c] for c in META_COLS}).to_csv(
            OUT / f"Scoruri_tSNE_3D_perp{perp_3d}.csv", index=False)
        for by in ["artist", "stil", "epoca", "gen"]:
            from mpl_toolkits.mplot3d import Axes3D
            categorii = sorted(metadata[by].dropna().unique().tolist())
            culori = grafice.generare_culori(len(categorii))
            fig = plt.figure(figsize=(13, 11))
            ax = fig.add_subplot(111, projection="3d")
            for j, cat in enumerate(categorii):
                mask = (metadata[by] == cat).values
                ax.scatter(scoruri_3d[mask, 0], scoruri_3d[mask, 1], scoruri_3d[mask, 2],
                           color=culori[j], alpha=0.5, s=10, edgecolors="none",
                           label=str(cat) if len(categorii) <= 20 else None)
            if len(categorii) > 20:
                for j, cat in enumerate(categorii):
                    mask = (metadata[by] == cat).values
                    if mask.sum() < 3:
                        continue
                    cx, cy, cz = (float(scoruri_3d[mask, i].mean()) for i in range(3))
                    parts = str(cat).split()
                    ax.text(cx, cy, cz, parts[-1] if len(parts) > 1 else str(cat), fontsize=6)
            else:
                ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=7)
            ax.set_xlabel("t1")
            ax.set_ylabel("t2")
            ax.set_zlabel("t3")
            ax.set_title(f"t-SNE 3D (perp={perp_3d}) — {by}")
            plt.tight_layout()
            plt.savefig(OUT / f"Scatter3D_tSNE_perp{perp_3d}_{by}.pdf",
                        format="pdf", bbox_inches="tight")
            plt.close(fig)
            grafice.f_scatter_interactiv_3d(scoruri_3d, metadata, by=by,
                                             fisier=f"Scatter3D_tSNE_perp{perp_3d}_{by}.html",
                                             titlu=f"t-SNE 3D (perp={perp_3d}) — pe {by}")
        pasi.info(f"Scatter 3D generat pentru perp={perp_3d} × 4 coloraje")
    else:
        pasi.info("3D indisponibil (perp_3d=None)")
    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
