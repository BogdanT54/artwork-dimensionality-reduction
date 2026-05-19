"""Funcții de vizualizare. Toate graficele salvate ca PDF în data_out/."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns

DATA_OUT = Path("data_out")
_SUBDIR = ""  # setat de set_subdir() per main script


def set_subdir(name):
    """Setează subfolderul curent (ex. 'pca'); toate plot-urile vor merge în data_out/<name>/."""
    global _SUBDIR
    _SUBDIR = name or ""
    if _SUBDIR:
        (DATA_OUT / _SUBDIR).mkdir(parents=True, exist_ok=True)


def show():
    """Apelat la finalul fiecărui main (ca la profesor)."""
    plt.show()


def generare_culori(n):
    """Returnează o paletă de n culori distincte (combinație tab20 + Set3)."""
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set3").colors)
    base = base + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    while len(base) < n:
        base = base * 2
    return base[:n]


def _ellipsa_confidenta(ax, x_vals, y_vals, color, n_std=1.7, alpha_fill=0.10):
    """Elipsă de confidență bazată pe PCA al norului de puncte (≈ 90% CI pt n_std=1.7)."""
    if len(x_vals) < 5:
        return
    try:
        cov = np.cov(x_vals, y_vals)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if np.any(eigenvalues <= 0):
            return
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        ell = Ellipse(
            xy=(float(np.mean(x_vals)), float(np.mean(y_vals))),
            width=width, height=height, angle=angle,
            facecolor=color, alpha=alpha_fill,
            edgecolor=color, linewidth=1.2, linestyle="--",
        )
        ax.add_patch(ell)
    except Exception:
        pass


def _savefig(fisier):
    target_dir = DATA_OUT / _SUBDIR if _SUBDIR else DATA_OUT
    target_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_dir / fisier, format="pdf", bbox_inches="tight")


def plot_varianta(varianta_cum, fisier="Varianta_PCA.pdf", titlu="Varianță explicată cumulativă",
                  n_kaiser=None, n_elbow=None):
    """Bar + line cu varianța cumulativă PCA și prag 80%, cu markere Kaiser/Elbow."""
    var_cum = np.asarray(varianta_cum)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(1, len(var_cum) + 1)
    var_ind = np.diff(np.concatenate(([0], var_cum)))
    ax.bar(x, var_ind, color="steelblue", alpha=0.5, label="Varianță componentă", zorder=2)
    ax.plot(x, var_cum, "o-", color="crimson", markersize=4, linewidth=1.8,
            label="Cumulativă", zorder=3)
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1.5, label="Prag 80%", zorder=4)

    if n_kaiser is not None:
        ax.axvline(n_kaiser, color="purple", linestyle=":", linewidth=1.5,
                   label=f"Kaiser: {n_kaiser} comp.", zorder=4)
    if n_elbow is not None:
        ax.axvline(n_elbow, color="darkorange", linestyle=":", linewidth=1.5,
                   label=f"Elbow: {n_elbow} comp.", zorder=4)

    if var_cum[-1] >= 0.80:
        idx_80 = int(np.searchsorted(var_cum, 0.80))
        ax.annotate(
            f"80% la comp. {idx_80 + 1}",
            xy=(idx_80 + 1, 0.80), xytext=(idx_80 + max(3, len(x) // 10), 0.74),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.4),
            color="green", fontsize=9,
        )
    else:
        ax.text(
            0.98, 0.84,
            f"80% necesită > {len(var_cum)} componente\n"
            f"(acoperire actuală: {var_cum[-1] * 100:.1f}% din varianță)",
            transform=ax.transAxes, ha="right", va="bottom",
            color="darkgreen", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fff0", alpha=0.9, edgecolor="green"),
        )

    ax.set_xlabel("Componenta principală", fontsize=11)
    ax.set_ylabel("Procent varianță", fontsize=11)
    ax.set_title(titlu, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _savefig(fisier)


def plot_elbow(scoruri, k_optim=None, fisier="Elbow.pdf", titlu="Elbow", x_label="k", y_label="Scor"):
    """Curba Elbow generică (inerție PCA, eroare NMF, stres MDS)."""
    scoruri = np.asarray(scoruri)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(scoruri) + 1)
    ax.plot(x, scoruri, "o-", color="steelblue", linewidth=2)
    if k_optim is not None:
        ax.axvline(k_optim, color="crimson", linestyle="--", label=f"k optim = {k_optim}")
        ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(titlu)
    ax.grid(True, alpha=0.3)
    _savefig(fisier)


def corelograma(matrice, fisier="Corelograma.pdf", titlu="Corelograma", vmin=-1, vmax=1, cmap="RdBu_r", anot=False):
    """Heatmap pentru corelații sau loadings."""
    if isinstance(matrice, np.ndarray):
        matrice = pd.DataFrame(matrice)
    fig, ax = plt.subplots(figsize=(min(20, 1 + matrice.shape[1] * 0.4),
                                     min(20, 1 + matrice.shape[0] * 0.3)))
    sns.heatmap(matrice, vmin=vmin, vmax=vmax, cmap=cmap, annot=anot,
                fmt=".2f", ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(titlu)
    _savefig(fisier)


def plot_scoruri_corelatii(corelatii, fisier="Cercul_corelatiilor.pdf", titlu="Cercul corelațiilor", n_etichete=20):
    """Cercul corelațiilor (PCA biplot): corelații features-componente pe primele 2 axe."""
    corelatii = np.asarray(corelatii)
    fig, ax = plt.subplots(figsize=(10, 10))
    teta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(teta), np.sin(teta), "k--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    norme = np.linalg.norm(corelatii[:, :2], axis=1)
    indici_top = np.argsort(-norme)[:n_etichete]
    for i in range(corelatii.shape[0]):
        ax.scatter(corelatii[i, 0], corelatii[i, 1], s=8, alpha=0.3, color="steelblue")
    for i in indici_top:
        ax.annotate(f"f{i+1}", (corelatii[i, 0], corelatii[i, 1]), fontsize=8, color="crimson")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("C1")
    ax.set_ylabel("C2")
    ax.set_title(titlu)
    ax.set_aspect("equal")
    _savefig(fisier)


def f_scatter_picturi(scoruri, metadata, by="artist", fisier=None, titlu=None,
                       comp_x=0, comp_y=1, alpha=0.35, dim=9,
                       var_x=None, var_y=None):
    """
    Scatter 2D colorat după `by`, cu elipse de confidență per categorie și
    etichete la centroid. Axele arată % varianță dacă var_x/var_y sunt date.
    """
    if fisier is None:
        fisier = f"Scatter_{by}.pdf"
    if titlu is None:
        titlu = f"Scatter colorat după {by}"

    categorii = sorted(metadata[by].dropna().unique().tolist())
    n_cat = len(categorii)
    culori = generare_culori(n_cat)

    fig, ax = plt.subplots(figsize=(dim + 4, dim))

    # 1. Elipse de confidență (desenate primele, în spatele punctelor)
    for i, cat in enumerate(categorii):
        mask = (metadata[by] == cat).values
        if mask.sum() < 5:
            continue
        _ellipsa_confidenta(ax, scoruri[mask, comp_x], scoruri[mask, comp_y], culori[i])

    # 2. Puncte
    for i, cat in enumerate(categorii):
        mask = (metadata[by] == cat).values
        cnt = mask.sum()
        lbl = f"{cat} ({cnt})" if n_cat <= 20 else str(cat)
        ax.scatter(scoruri[mask, comp_x], scoruri[mask, comp_y],
                   color=culori[i], label=lbl, alpha=alpha, s=15, edgecolors="none", zorder=3)

    # 3. Etichete centroid
    for i, cat in enumerate(categorii):
        mask = (metadata[by] == cat).values
        if mask.sum() < 3:
            continue
        cx = float(scoruri[mask, comp_x].mean())
        cy = float(scoruri[mask, comp_y].mean())
        # Pentru multe categorii: doar ultima parte din nume (ex. "van Gogh" din "Vincent van Gogh")
        if n_cat > 20:
            parts = str(cat).split()
            short = parts[-1] if len(parts) > 1 else str(cat)
        else:
            short = str(cat)
        ax.text(cx, cy, short, fontsize=6 if n_cat > 20 else 7,
                color=culori[i], ha="center", va="center", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.75, edgecolor="none"))

    xlabel = f"Comp{comp_x + 1}" + (f"  ({var_x:.1f}% varianță)" if var_x is not None else "")
    ylabel = f"Comp{comp_y + 1}" + (f"  ({var_y:.1f}% varianță)" if var_y is not None else "")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(titlu, fontsize=13)

    ncol = 1 if n_cat <= 25 else 2
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=6 if n_cat > 25 else 7, ncol=ncol, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _savefig(fisier)


def plot_poze_picturi(paths, titluri, fisier, suptitlu=None, n_cols=5):
    """Grid de imagini cu titluri (folosit pentru eigenpicturi semantice și componente NMF)."""
    n = len(paths)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.7))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for i, (p, t) in enumerate(zip(paths, titluri)):
        r, c = divmod(i, n_cols)
        try:
            img = mpimg.imread(p)
            axes[r, c].imshow(img)
        except Exception:
            axes[r, c].text(0.5, 0.5, "n/a", ha="center", va="center")
        axes[r, c].set_title(t, fontsize=7)
        axes[r, c].axis("off")
    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")
    if suptitlu:
        fig.suptitle(suptitlu, fontsize=10)
    plt.tight_layout()
    _savefig(fisier)


def plot_eigenpicturi_pca(scoruri, paths_all, fisier="Eigenpicturi_PCA.pdf",
                           n_comp=6, k=5):
    """
    Pentru fiecare din primele n_comp componente PCA: afișează top k picturi cu scor maxim
    și top k picturi cu scor minim (eigenpicturi semantice).
    """
    paths_all = list(paths_all)
    n = len(paths_all)
    fig, axes = plt.subplots(n_comp * 2, k, figsize=(k * 2.4, n_comp * 2 * 2.6))
    for c in range(n_comp):
        col = scoruri[:, c]
        top_max = np.argsort(col)[-k:][::-1]
        top_min = np.argsort(col)[:k]
        for j, idx in enumerate(top_max):
            r = c * 2
            try:
                img = mpimg.imread(paths_all[idx])
                axes[r, j].imshow(img)
            except Exception:
                axes[r, j].text(0.5, 0.5, "n/a", ha="center", va="center")
            axes[r, j].set_title(f"Comp{c+1} MAX #{j+1}\n{col[idx]:+.2f}", fontsize=7)
            axes[r, j].axis("off")
        for j, idx in enumerate(top_min):
            r = c * 2 + 1
            try:
                img = mpimg.imread(paths_all[idx])
                axes[r, j].imshow(img)
            except Exception:
                axes[r, j].text(0.5, 0.5, "n/a", ha="center", va="center")
            axes[r, j].set_title(f"Comp{c+1} MIN #{j+1}\n{col[idx]:+.2f}", fontsize=7)
            axes[r, j].axis("off")
    fig.suptitle("Eigenpicturi PCA (top max/min picturi pe componentă)", fontsize=11)
    plt.tight_layout()
    _savefig(fisier)


def plot_componente_nmf(W, paths_all, fisier="Componente_NMF.pdf", n_comp=6, k=5):
    """Top k picturi cu cea mai mare activare pe fiecare componentă NMF (interpretare aditivă)."""
    paths_all = list(paths_all)
    n_comp = min(n_comp, W.shape[1])
    fig, axes = plt.subplots(n_comp, k, figsize=(k * 2.4, n_comp * 2.6))
    if n_comp == 1:
        axes = axes.reshape(1, -1)
    for c in range(n_comp):
        col = W[:, c]
        top = np.argsort(col)[-k:][::-1]
        for j, idx in enumerate(top):
            try:
                img = mpimg.imread(paths_all[idx])
                axes[c, j].imshow(img)
            except Exception:
                axes[c, j].text(0.5, 0.5, "n/a", ha="center", va="center")
            axes[c, j].set_title(f"NMF C{c+1} #{j+1}\nw={col[idx]:.2f}", fontsize=7)
            axes[c, j].axis("off")
    fig.suptitle("Componente NMF — top picturi pe componentă (aditiv)", fontsize=11)
    plt.tight_layout()
    _savefig(fisier)


def plot_stres_mds(q_values, stres, fisier="Stres_MDS.pdf"):
    """Curba stresului MDS în funcție de numărul de componente."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(q_values, stres, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Număr componente MDS")
    ax.set_ylabel("Stres Kruskal-1")
    ax.set_title("Stresul MDS în funcție de numărul de componente (Elbow)")
    ax.grid(True, alpha=0.3)
    _savefig(fisier)


def plot_shepard(D_orig, D_redus, fisier="Shepard_MDS.pdf"):
    """Diagrama Shepard MDS: distanțe originale vs distanțe în spațiul redus."""
    D_orig = np.asarray(D_orig)
    D_redus = np.asarray(D_redus)
    iu = np.triu_indices_from(D_orig, k=1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(D_orig[iu], D_redus[iu], s=12, alpha=0.5, color="steelblue")
    lim = max(D_orig[iu].max(), D_redus[iu].max())
    ax.plot([0, lim], [0, lim], "r--", alpha=0.7, label="y = x")
    ax.set_xlabel("Distanță originală")
    ax.set_ylabel("Distanță în spațiu MDS")
    ax.set_title("Diagrama Shepard")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fisier)


def plot_reconstructie_pca(eroare_per_q, fisier="Reconstructie_PCA.pdf"):
    """Eroarea de reconstrucție PCA în funcție de q."""
    fig, ax = plt.subplots(figsize=(10, 6))
    q_values, erori = zip(*eroare_per_q)
    ax.plot(q_values, erori, "o-", color="darkorange", linewidth=2)
    ax.set_xlabel("Număr componente q")
    ax.set_ylabel("Eroare reconstrucție (Frobenius)")
    ax.set_title("Eroare reconstrucție PCA")
    ax.grid(True, alpha=0.3)
    _savefig(fisier)


def plot_heatmap_distante(D, etichete, fisier="Heatmap_distante.pdf", titlu="Matrice distanțe"):
    """Heatmap distanțe între pictori (MDS)."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(D, xticklabels=etichete, yticklabels=etichete, cmap="viridis",
                square=True, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(titlu)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    _savefig(fisier)


def plot_bar(valori, etichete, fisier, titlu="", x_label="", y_label="", rotatie=45):
    """Bar chart generic (comunalități, kurtosis ICA, timpi execuție)."""
    fig, ax = plt.subplots(figsize=(max(8, len(valori) * 0.4), 6))
    ax.bar(range(len(valori)), valori, color="steelblue")
    ax.set_xticks(range(len(valori)))
    ax.set_xticklabels(etichete, rotation=rotatie, ha="right", fontsize=8)
    ax.set_title(titlu)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fisier)


def plot_scatter_grid(panouri, fisier, suptitlu=""):
    """
    Grid de scatter-uri pentru comparație cross-metode.
    `panouri` = listă de tuple (scoruri_2d, metadata_df, by, titlu_panou).
    """
    n = len(panouri)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = np.array([axes])
    for i, (scoruri, meta, by, titlu) in enumerate(panouri):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        categorii = sorted(meta[by].dropna().unique().tolist())
        culori = generare_culori(len(categorii))
        for j, cat in enumerate(categorii):
            mask = (meta[by] == cat).values
            ax.scatter(scoruri[mask, 0], scoruri[mask, 1], color=culori[j],
                       alpha=0.6, s=10, edgecolors="none")
        ax.set_title(titlu, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")
    if suptitlu:
        fig.suptitle(suptitlu, fontsize=12)
    plt.tight_layout()
    _savefig(fisier)
