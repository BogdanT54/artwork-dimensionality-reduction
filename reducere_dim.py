"""
Logica celor 7 metode de reducere a dimensionalității, semnătură similară cu
clusterizare.py al profesorului: aplica_<metoda>(df, x, metadata, ...).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF, FastICA, KernelPCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

import functii


@dataclass
class RezultatReducere:
    nume: str
    scoruri: np.ndarray
    extra: dict = field(default_factory=dict)


def aplica_pca(df, x, metadata, n_max=20):
    """PCA full + criterii multiple de selecție a numărului de componente."""
    n_max = min(n_max, x.shape[1], x.shape[0])
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    pca = PCA(n_components=n_max)
    scoruri = pca.fit_transform(x_std)

    varianta = pca.explained_variance_
    varianta_ratio = pca.explained_variance_ratio_
    varianta_cum = np.cumsum(varianta_ratio)

    # eigenvalues pe matricea de corelație standardizată ≈ varianta
    n_kaiser = functii.kaiser_n_componente(varianta)
    n_80 = int(np.searchsorted(varianta_cum, 0.80) + 1)
    n_elbow = functii.elbow_index(varianta_ratio) + 1

    # corelații features-componente: r_xc = loading * sqrt(varianta) / sqrt(var_feature)
    loadings = pca.components_.T  # (n_features, n_comp)
    corelatii = loadings * np.sqrt(varianta)  # standardizat => var_feature = 1

    return RezultatReducere(
        nume="PCA",
        scoruri=scoruri,
        extra={
            "varianta": varianta,
            "varianta_ratio": varianta_ratio,
            "varianta_cum": varianta_cum,
            "loadings": loadings,
            "corelatii": corelatii,
            "n_kaiser": n_kaiser,
            "n_80": n_80,
            "n_elbow": n_elbow,
            "scaler": scaler,
            "pca_model": pca,
        },
    )


def aplica_fa(df, x, metadata, n_factori=None):
    """
    Analiză factorială cu rotație Varimax (sklearn) + test Bartlett + KMO (implementare manuală).
    """
    from sklearn.decomposition import FactorAnalysis

    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    bartlett_kmo = functii.bartlett_kmo(x_std)

    # determinăm numărul de factori (eigenvalues > 1 din matricea de corelație, criteriul Kaiser)
    if n_factori is None:
        R = np.corrcoef(x_std, rowvar=False)
        R = np.nan_to_num(R, nan=0.0)
        ev = np.linalg.eigvalsh(R)[::-1]
        n_factori = max(2, functii.kaiser_n_componente(ev))
        n_factori = min(n_factori, 20)  # plafonăm pentru tractabilitate

    fa = FactorAnalysis(n_components=n_factori, rotation="varimax", random_state=42)
    scoruri = fa.fit_transform(x_std)
    # sklearn FactorAnalysis: components_ are (n_components, n_features) → transpun pentru loadings
    loadings = fa.components_.T                            # (n_features, n_factori)
    comunalitati = functii.calcul_comunalitati(loadings)

    # varianța explicată per factor pe datele standardizate (var(X)=1 per feature)
    var_factor = (loadings ** 2).sum(axis=0)
    var_total_orig = x_std.shape[1]  # = nr features (var=1 fiecare)
    prop_var = var_factor / var_total_orig
    cum_var = np.cumsum(prop_var)
    varianta = (var_factor, prop_var, cum_var)

    return RezultatReducere(
        nume="FA",
        scoruri=scoruri,
        extra={
            "loadings": loadings,
            "comunalitati": comunalitati,
            "varianta": varianta,
            "bartlett_kmo": bartlett_kmo,
            "n_factori": n_factori,
        },
    )


def aplica_nmf(df, x, metadata, q_list=(5, 10, 15, 20, 30, 50)):
    """NMF cu Elbow pe eroare reconstrucție. X trebuie să fie ≥ 0 (VGG16 fc2 = ReLU)."""
    x = np.asarray(x)
    if x.min() < 0:
        # safety: clip la 0 dacă există rounding/normalizare ce a produs negative
        x = np.clip(x, 0, None)

    erori = []
    rezultate = {}
    for q in q_list:
        model = NMF(n_components=q, init="nndsvd", max_iter=1000, random_state=42)
        W = model.fit_transform(x)
        H = model.components_
        reconstr = W @ H
        eroare = float(np.linalg.norm(x - reconstr, "fro"))
        erori.append(eroare)
        rezultate[q] = {"W": W, "H": H, "eroare": eroare, "model": model}

    idx_elbow = functii.elbow_index(erori)
    q_optim = q_list[idx_elbow]
    return RezultatReducere(
        nume="NMF",
        scoruri=rezultate[q_optim]["W"],
        extra={
            "q_list": list(q_list),
            "erori": erori,
            "q_optim": q_optim,
            "rezultate_per_q": rezultate,
            "H_optim": rezultate[q_optim]["H"],
        },
    )


def aplica_ica(df, x, metadata, k):
    """FastICA cu n_components = k (din Kaiser PCA). Calculează entropie + kurtosis."""
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    ica = FastICA(n_components=k, random_state=42, max_iter=1000, whiten="unit-variance")
    scoruri = ica.fit_transform(x_std)
    df_ent = functii.calcul_entropie_ica(scoruri)
    return RezultatReducere(
        nume="ICA",
        scoruri=scoruri,
        extra={
            "mixing": ica.mixing_,
            "components": ica.components_,
            "entropie_kurtosis": df_ent,
            "k": k,
        },
    )


def aplica_kpca(df, x, metadata, kernel="rbf", n_components=20, gamma=None):
    """Kernel PCA cu kernel-ul ales (RBF implicit)."""
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma,
                     fit_inverse_transform=False, random_state=42)
    scoruri = kpca.fit_transform(x_std)

    # PCA liniar de referință (pentru comparație side-by-side)
    pca_ref = PCA(n_components=n_components)
    scoruri_pca = pca_ref.fit_transform(x_std)

    gamma_efectiv = gamma if gamma is not None else 1.0 / x_std.shape[1]
    return RezultatReducere(
        nume="KPCA",
        scoruri=scoruri,
        extra={
            "kernel": kernel,
            "gamma": gamma,
            "gamma_efectiv": gamma_efectiv,
            "scoruri_pca_referinta": scoruri_pca,
            "eigenvalues": getattr(kpca, "eigenvalues_", None),
            "model": kpca,
        },
    )


def aplica_mds(df, x, metadata, coloana_grup="artist", q_list=(2, 3, 4, 5, 6, 8, 10)):
    """
    MDS pe centroide per pictor (50×50 matrice de distanțe).
    Reduce complexitatea de la O(n²) la O(p²) unde p = nr pictori.
    """
    from scipy.spatial.distance import pdist, squareform

    df_meta = metadata.reset_index(drop=True)
    x_df = pd.DataFrame(x).reset_index(drop=True)
    x_df[coloana_grup] = df_meta[coloana_grup].values
    centroide_df = x_df.groupby(coloana_grup).mean()
    etichete = centroide_df.index.tolist()
    centroide = centroide_df.values

    D_orig = squareform(pdist(centroide, metric="euclidean"))

    stres = []
    scoruri_per_q = {}
    for q in q_list:
        mds = MDS(n_components=q, dissimilarity="precomputed", random_state=42,
                  n_init=4, max_iter=300, normalized_stress="auto", n_jobs=1)
        scoruri = mds.fit_transform(D_orig)
        D_redus = squareform(pdist(scoruri, metric="euclidean"))
        s = functii.calcul_stres_mds(D_orig, D_redus)
        stres.append(s)
        scoruri_per_q[q] = scoruri

    idx_elbow = functii.elbow_index(stres)
    q_optim = q_list[idx_elbow]

    return RezultatReducere(
        nume="MDS",
        scoruri=scoruri_per_q[q_optim],
        extra={
            "q_list": list(q_list),
            "stres": stres,
            "q_optim": q_optim,
            "etichete": etichete,
            "D_orig": D_orig,
            "scoruri_per_q": scoruri_per_q,
        },
    )


def aplica_tsne(df, x, metadata, perplexity_list=(5, 30, 50, 100), n_pca=50, perp_3d=50):
    """t-SNE 2D pe primele n_pca componente PCA (pipeline standard)."""
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    n_pca = min(n_pca, x_std.shape[1], x_std.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    x_pca = pca.fit_transform(x_std)

    rezultate = {}
    for perp in perplexity_list:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init="pca", max_iter=1000, learning_rate="auto")
        scoruri = tsne.fit_transform(x_pca)
        rezultate[perp] = {
            "scoruri": scoruri,
            "kl_divergence": float(tsne.kl_divergence_),
        }

    # 3D t-SNE pentru perp_3d (folosit în 07_main_tsne.py)
    scoruri_3d = None
    if perp_3d is not None:
        tsne_3d = TSNE(n_components=3, perplexity=perp_3d, random_state=42,
                       init="pca", max_iter=1000, learning_rate="auto")
        scoruri_3d = tsne_3d.fit_transform(x_pca)

    return RezultatReducere(
        nume="tSNE",
        scoruri=rezultate[perplexity_list[0]]["scoruri"],
        extra={
            "perplexity_list": list(perplexity_list),
            "rezultate_per_perp": rezultate,
            "n_pca_intermediar": n_pca,
            "scoruri_3d": scoruri_3d,
            "perp_3d": perp_3d,
        },
    )
