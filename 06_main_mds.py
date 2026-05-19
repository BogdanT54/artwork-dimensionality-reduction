"""Pas 6: MDS pe centroide per pictor + Elbow stres + Shepard + heatmap + 3D."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import functii
import grafice
import reducere_dim

META_COLS = ["path", "artist", "stil", "epoca", "gen"]
SUBDIR = "mds"


def _construieste_validitate(e):
    q_list = e["q_list"]
    stres = e["stres"]
    q_optim = e["q_optim"]
    idx_optim = q_list.index(q_optim)
    stres_optim = stres[idx_optim]

    sc_optim = e["scoruri_per_q"][q_optim]
    D_redus = squareform(pdist(sc_optim, metric="euclidean"))
    n = D_redus.shape[0]
    idx_upper = np.triu_indices(n, k=1)
    d_orig_flat = e["D_orig"][idx_upper]
    d_red_flat = D_redus[idx_upper]
    r_shepard = float(np.corrcoef(d_orig_flat, d_red_flat)[0, 1])

    stres_2d = stres[q_list.index(2)]
    idx_best = int(np.argmin(stres))

    rows = [
        {"Criteriu": "Nr pictori (MDS pe centroide)",
         "Valoare": f"{len(e['etichete'])}",
         "Interpretare": "MDS pe centroide CNN (evită O(n²) memory pentru ~8446 imagini)"},
        {"Criteriu": "q optim (Elbow stres)",
         "Valoare": f"{q_optim} dim",
         "Interpretare": "Punct de cot al curbei Kruskal stress vs q"},
        {"Criteriu": "Kruskal stress @ q_optim",
         "Valoare": f"{stres_optim:.4f}",
         "Interpretare": "<0.05 excelent, <0.10 bun, <0.20 acceptabil, >0.20 slab"},
        {"Criteriu": "Kruskal stress @ q=2 (vizualizare)",
         "Valoare": f"{stres_2d:.4f}",
         "Interpretare": "Stress pentru proiecția 2D standard"},
        {"Criteriu": "Shepard r (corelație distanțe @ q_optim)",
         "Valoare": f"{r_shepard:.4f}",
         "Interpretare": "Corelația dist. originale vs dist. reduse (>0.9 = excelent)"},
        {"Criteriu": "Interval stres (min → max)",
         "Valoare": f"{min(stres):.4f} → {max(stres):.4f}",
         "Interpretare": f"q={q_list[idx_best]} dă cel mai mic stress (max dim testate)"},
    ]
    return pd.DataFrame(rows)


def main():
    pasi = functii.Pasi("MDS", total=5)
    functii.goleste_data_out(subdir=SUBDIR)
    grafice.set_subdir(SUBDIR)
    OUT = functii.subdir(SUBDIR)

    pasi.pas("Citire features_cnn.csv + centroide per pictor")
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)
    pasi.info(f"X = {x.shape}, MDS pe centroide per pictor  →  data_out/{SUBDIR}/")

    pasi.pas("Fit MDS pentru q ∈ (2,3,4,5,6,8,10) — Elbow stres (poate dura 1-2 min)")
    rez = reducere_dim.aplica_mds(df, x, metadata, coloana_grup="artist",
                                  q_list=(2, 3, 4, 5, 6, 8, 10))
    e = rez.extra
    pasi.info(f"{len(e['etichete'])} pictori; q optim = {e['q_optim']}")
    pasi.info(f"Stres per q: {[f'{s:.3f}' for s in e['stres']]}")

    pasi.pas("Salvare distanțe + scoruri + stres")
    pd.DataFrame(e["D_orig"], index=e["etichete"], columns=e["etichete"]
                 ).to_csv(OUT / "Distante_MDS.csv")
    pd.DataFrame(rez.scoruri, index=e["etichete"],
                 columns=[f"MDS{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).to_csv(OUT / "Scoruri_MDS.csv")
    pd.DataFrame({"q": e["q_list"], "Stres": e["stres"]}
                 ).to_csv(OUT / "Stres_MDS.csv", index=False)

    pasi.pas("Tabel de validitate")
    df_val = _construieste_validitate(e)
    functii.salveaza_validitate(df_val, SUBDIR, "Validitate_MDS.csv")
    grafice.plot_validitate(df_val, "Validitate_MDS.pdf",
                            titlu="Validitate MDS — Kruskal stress, Shepard")
    print(df_val.to_string(index=False))

    pasi.pas("Grafice (Elbow stres, heatmap, Shepard, scatter 2D etichetat, 3D)")
    grafice.plot_stres_mds(e["q_list"], e["stres"], "Stres_MDS.pdf")
    grafice.plot_heatmap_distante(e["D_orig"], e["etichete"], "Heatmap_Distante_MDS.pdf",
                                   titlu="Matrice distanțe euclidiene între pictori (CNN centroid)")

    sc2 = e["scoruri_per_q"][2]
    D_redus_2d = squareform(pdist(sc2, metric="euclidean"))
    grafice.plot_shepard(e["D_orig"], D_redus_2d, "Shepard_MDS.pdf")

    # Scatter 2D cu etichete pictori
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(sc2[:, 0], sc2[:, 1], s=80, c="steelblue", alpha=0.6)
    for i, nume in enumerate(e["etichete"]):
        ax.annotate(nume, (sc2[i, 0], sc2[i, 1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("MDS 2D — pictori (centroide CNN)")
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "Scatter_MDS_pictori.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 3D scatter cu q=3 scoruri
    sc3 = e["scoruri_per_q"].get(3)
    if sc3 is not None:
        from mpl_toolkits.mplot3d import Axes3D
        fig3 = plt.figure(figsize=(14, 12))
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.scatter(sc3[:, 0], sc3[:, 1], sc3[:, 2], s=80, c="steelblue", alpha=0.6)
        for i, nume in enumerate(e["etichete"]):
            ax3.text(sc3[i, 0], sc3[i, 1], sc3[i, 2], nume[:12], fontsize=6)
        ax3.set_title("MDS 3D — pictori (centroide CNN)")
        ax3.set_xlabel("MDS1")
        ax3.set_ylabel("MDS2")
        ax3.set_zlabel("MDS3")
        plt.tight_layout()
        plt.savefig(OUT / "Scatter3D_MDS_pictori.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig3)

    grafice.show()
    pasi.terminat()


if __name__ == "__main__":
    main()
