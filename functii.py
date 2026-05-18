"""Funcții de calcul și utilitare pentru proiectul de reducere a dimensionalității."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis

DATA_OUT = Path("data_out")
DATA_IN = Path("data_in") / "best_artworks"


def goleste_data_out(tokens=None):
    """
    Asigură existența data_out/. Dacă `tokens` e None: șterge tot (mod profesor).
    Dacă `tokens` e lista de substring-uri: șterge doar fișierele care conțin oricare token —
    astfel fiecare main_X.py poate șterge doar propriile outputuri, fără să afecteze celelalte.
    """
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    for fisier in DATA_OUT.iterdir():
        if not fisier.is_file():
            continue
        if tokens is None or any(t in fisier.name for t in tokens):
            fisier.unlink()


def salvare_ndarray(arr, nume_fisier, header=None, index=None):
    """Salvează un ndarray ca CSV în data_out/."""
    df = pd.DataFrame(arr)
    if header is not None:
        df.columns = header
    if index is not None:
        df.index = index
    df.to_csv(DATA_OUT / nume_fisier)
    return df


def tabelare_varianta(varianta):
    """Construiește tabelul varianță explicată / cumulativă (ca la profesor)."""
    var_ratio = np.array(varianta)
    cum = np.cumsum(var_ratio)
    df = pd.DataFrame({
        "Varianta_explicata": var_ratio,
        "Procent": var_ratio / var_ratio.sum() * 100,
        "Procent_cumulat": cum / cum[-1] * 100,
    })
    df.index = [f"Comp{i+1}" for i in range(len(var_ratio))]
    return df


def nan_replace_df(df):
    """Înlocuiește NaN cu media coloanei (numerice) sau modul (text)."""
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def preprocesare_imagine(path, target_size=(224, 224)):
    """Încarcă o imagine și o pregătește pentru VGG16 (resize + preprocess_input)."""
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input

    img = image.load_img(path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr[0]


def _afiseaza_arhitectura_vgg16(n_imagini, batch_size):
    """Afișează diagrama arhitecturii VGG16 și statisticile sesiunii de extracție."""
    n_batches = (n_imagini + batch_size - 1) // batch_size
    W = 61
    sep = "+" + "-" * (W - 2) + "+"

    def rand(text):
        padding = W - 2 - len(text)
        return "| " + text + " " * (padding - 1) + "|"

    linii = [
        "",
        sep,
        rand("  VGG16  —  extragere vectori fc2 (4096-dim, ReLU >= 0)"),
        sep,
        rand("  Input:  224 x 224 x 3  (ImageNet preprocess)"),
        rand(""),
        rand("  Block 1 :  Conv3-64  -> Conv3-64  -> MaxPool"),
        rand("  Block 2 :  Conv3-128 -> Conv3-128 -> MaxPool"),
        rand("  Block 3 :  Conv3-256 -> Conv3-256 -> Conv3-256 -> MaxPool"),
        rand("  Block 4 :  Conv3-512 -> Conv3-512 -> Conv3-512 -> MaxPool"),
        rand("  Block 5 :  Conv3-512 -> Conv3-512 -> Conv3-512 -> MaxPool"),
        rand(""),
        rand("  Flatten  ->  FC-4096 (ReLU)  ->  [FC-4096]  ->  FC-1000"),
        rand("                                         ^"),
        rand("                                    iesire fc2"),
        sep,
        rand(f"  {n_imagini} imagini  |  batch {batch_size}  |  {n_batches} batch-uri totale"),
        sep,
        "",
    ]
    for linie in linii:
        print(linie)


def extragere_cnn_vgg16(image_paths, batch_size=32):
    """
    Extrage vectori 4096-dim din stratul fc2 al VGG16 (activare ReLU → valori ≥ 0).

    Returnează ndarray (n_imagini, 4096) și lista de paths efectiv procesate.
    """
    import time
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image
    from tqdm import tqdm

    _afiseaza_arhitectura_vgg16(len(image_paths), batch_size)

    base = VGG16(weights="imagenet", include_top=True)
    model = Model(inputs=base.input, outputs=base.get_layer("fc2").output)
    print()

    n_total = len(image_paths)
    n_batches = (n_total + batch_size - 1) // batch_size
    features, paths_ok, imagini_ok, sarite = [], [], 0, 0
    timpi_batch = []

    bara = tqdm(
        range(0, n_total, batch_size),
        total=n_batches,
        desc="  VGG16 fc2",
        unit="batch",
        dynamic_ncols=True,
    )
    for start in bara:
        batch_paths = image_paths[start : start + batch_size]
        pictor = Path(batch_paths[0]).parent.name.replace("_", " ") if batch_paths else "?"

        t0 = time.time()
        arr_list, ok_list = [], []
        for p in batch_paths:
            try:
                img = image.load_img(p, target_size=(224, 224))
                arr_list.append(image.img_to_array(img))
                ok_list.append(p)
            except Exception as exc:
                sarite += 1
                tqdm.write(f"  [skip] {Path(p).name}: {exc}")

        if arr_list:
            batch_np = preprocess_input(np.stack(arr_list, axis=0))
            feats = model.predict(batch_np, verbose=0)
            features.append(feats)
            paths_ok.extend(ok_list)
            imagini_ok += len(ok_list)

        durata = time.time() - t0
        timpi_batch.append(durata)

        bara.set_postfix(
            pictor=pictor[:22],
            imagini=f"{imagini_ok}/{n_total}",
            s_batch=f"{durata:.1f}s",
            sarite=sarite,
        )

    bara.close()
    medie = float(np.mean(timpi_batch)) if timpi_batch else 0.0
    print(
        f"\n  [OK] {imagini_ok} imagini procesate"
        f"  |  {sarite} sarite"
        f"  |  medie {medie:.2f}s/batch\n"
    )
    return np.vstack(features), paths_ok


def calcul_stres_mds(D_orig, D_redus):
    """Stresul Kruskal-1 între matricea de distanțe originală și cea redusă."""
    D_orig = np.asarray(D_orig)
    D_redus = np.asarray(D_redus)
    num = np.sum((D_orig - D_redus) ** 2)
    den = np.sum(D_orig ** 2)
    return float(np.sqrt(num / den))


def calcul_entropie_ica(scoruri):
    """
    Entropia + kurtosis pentru fiecare componentă independentă.
    Entropia mare = distribuție apropiată de gaussian; kurtosis mare = non-gaussianitate.
    """
    n_comp = scoruri.shape[1]
    rezultat = []
    for k in range(n_comp):
        s = scoruri[:, k]
        hist, _ = np.histogram(s, bins=50, density=True)
        hist = hist + 1e-12
        ent = scipy_entropy(hist)
        kurt = scipy_kurtosis(s, fisher=True)
        rezultat.append((ent, kurt))
    df = pd.DataFrame(rezultat, columns=["Entropie", "Kurtosis"])
    df.index = [f"IC{i+1}" for i in range(n_comp)]
    return df


def calcul_comunalitati(loadings):
    """Comunalitățile FA = suma pătratelor loadings-urilor pe fiecare variabilă (rând)."""
    return np.sum(loadings ** 2, axis=1)


def top_picturi_componenta(scoruri, k=5):
    """
    Returnează dict {componenta: (top_max_idx, top_min_idx)} cu primele k indici
    cu cel mai mare și cel mai mic scor pe fiecare componentă.
    """
    n_comp = scoruri.shape[1]
    out = {}
    for c in range(n_comp):
        col = scoruri[:, c]
        top_max = np.argsort(col)[-k:][::-1]
        top_min = np.argsort(col)[:k]
        out[c] = (top_max, top_min)
    return out


def bartlett_kmo(X):
    """
    Testul Bartlett de sferocitate + indicele KMO (Kaiser-Meyer-Olkin) pentru
    validitatea analizei factoriale. Implementare manuală (independentă de factor_analyzer).

    Returnează dict cu chi2, p_value, kmo_total, kmo_per_var.
    """
    from scipy.stats import chi2 as chi2_dist

    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)

    # Bartlett: chi2 = -(n-1 - (2p+5)/6) * ln|R|
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        logdet = -np.inf
    chi2_val = -(n - 1 - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) / 2
    p_value = float(1 - chi2_dist.cdf(chi2_val, df))

    # KMO: necesită inversa lui R (anti-imagine)
    try:
        R_inv = np.linalg.pinv(R)
        D = np.sqrt(np.diag(R_inv))
        partial = -R_inv / np.outer(D, D)
        np.fill_diagonal(partial, 1.0)
        # KMO formula
        num_total = (R ** 2).sum() - p
        partial_sq = (partial ** 2).sum() - p
        kmo_total = num_total / (num_total + partial_sq)
        # KMO per variabilă
        kmo_per_var = np.zeros(p)
        for j in range(p):
            mask = np.arange(p) != j
            r_sq = (R[j, mask] ** 2).sum()
            p_sq = (partial[j, mask] ** 2).sum()
            kmo_per_var[j] = r_sq / (r_sq + p_sq) if (r_sq + p_sq) > 0 else 0
    except Exception:
        kmo_total = float("nan")
        kmo_per_var = np.full(p, float("nan"))

    return {
        "chi2": float(chi2_val),
        "p_value": p_value,
        "kmo_total": float(kmo_total),
        "kmo_per_var": kmo_per_var,
    }


def kaiser_n_componente(eigenvalues, prag=1.0):
    """Numărul de componente cu eigenvalue > prag (criteriul Kaiser)."""
    return int(np.sum(np.asarray(eigenvalues) > prag))


def elbow_index(values):
    """
    Heuristică simplă pentru cot: punctul cu distanța maximă față de dreapta dintre
    primul și ultimul element al curbei (kneedle simplificat).
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3:
        return 0
    p1 = np.array([0, v[0]])
    p2 = np.array([n - 1, v[-1]])
    line_vec = p2 - p1
    line_norm = line_vec / np.linalg.norm(line_vec)
    distante = []
    for i in range(n):
        p = np.array([i, v[i]])
        vec = p - p1
        proj = np.dot(vec, line_norm) * line_norm
        ortho = vec - proj
        distante.append(np.linalg.norm(ortho))
    return int(np.argmax(distante))


def deriva_epoca(years_str):
    """Convertește un string '1853 - 1890' într-un bucket 'sec. XIX'."""
    if not isinstance(years_str, str):
        return "necunoscut"
    cifre = [int(x) for x in years_str.replace("–", "-").split("-") if x.strip().isdigit()]
    if not cifre:
        return "necunoscut"
    anul = cifre[0]
    secol = (anul // 100) + 1
    mapare = {14: "sec. XIV", 15: "sec. XV", 16: "sec. XVI", 17: "sec. XVII",
              18: "sec. XVIII", 19: "sec. XIX", 20: "sec. XX", 21: "sec. XXI"}
    return mapare.get(secol, f"sec. {secol}")


def deriva_stil(genre_str):
    """Simplifică un string genre cu virgule la primul stil dominant."""
    if not isinstance(genre_str, str):
        return "necunoscut"
    primul = genre_str.split(",")[0].strip()
    return primul if primul else "necunoscut"
