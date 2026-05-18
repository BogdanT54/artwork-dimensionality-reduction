"""Pas 6: MDS pe centroide per pictor + Elbow stres + Shepard + heatmap distanțe."""
import numpy as np
import pandas as pd

import functii
import grafice
import reducere_dim
from scipy.spatial.distance import pdist, squareform

META_COLS = ["path", "artist", "stil", "epoca", "gen"]


def main():
    functii.goleste_data_out(tokens=["_MDS.", "Distante_MDS", "Scoruri_MDS",
                                       "Stres_MDS", "Heatmap_Distante_MDS",
                                       "Shepard_MDS", "Scatter_MDS"])
    df = pd.read_csv(functii.DATA_IN / "features_cnn.csv")
    metadata = df[META_COLS].copy()
    x = df.drop(columns=META_COLS).values.astype(np.float32)

    rez = reducere_dim.aplica_mds(df, x, metadata, coloana_grup="artist",
                                  q_list=(2, 3, 4, 5, 6, 8, 10))
    e = rez.extra
    print(f"[info] MDS pe {len(e['etichete'])} pictori; q optim = {e['q_optim']}")

    pd.DataFrame(e["D_orig"], index=e["etichete"], columns=e["etichete"]
                 ).to_csv(functii.DATA_OUT / "Distante_MDS.csv")
    pd.DataFrame(rez.scoruri, index=e["etichete"],
                 columns=[f"MDS{i+1}" for i in range(rez.scoruri.shape[1])]
                 ).to_csv(functii.DATA_OUT / "Scoruri_MDS.csv")
    pd.DataFrame({"q": e["q_list"], "Stres": e["stres"]}
                 ).to_csv(functii.DATA_OUT / "Stres_MDS.csv", index=False)

    grafice.plot_stres_mds(e["q_list"], e["stres"], "Stres_MDS.pdf")
    grafice.plot_heatmap_distante(e["D_orig"], e["etichete"], "Heatmap_Distante_MDS.pdf",
                                   titlu="Matrice distanțe euclidiene între pictori (CNN centroid)")

    # Shepard pe q optim
    scoruri = rez.scoruri
    D_redus = squareform(pdist(scoruri, metric="euclidean"))
    grafice.plot_shepard(e["D_orig"], D_redus, "Shepard_MDS.pdf")

    # scatter 2D etichetat cu nume pictori
    import matplotlib.pyplot as plt
    sc2 = e["scoruri_per_q"][2]
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(sc2[:, 0], sc2[:, 1], s=80, c="steelblue", alpha=0.6)
    for i, nume in enumerate(e["etichete"]):
        ax.annotate(nume, (sc2[i, 0], sc2[i, 1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("MDS 2D — pictori")
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(functii.DATA_OUT / "Scatter_MDS_pictori.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    grafice.show()


if __name__ == "__main__":
    main()
