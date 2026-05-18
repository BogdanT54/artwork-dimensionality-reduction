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


LAYERE_VIZ = ("block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3")


def _diagrama_visualkeras(model, output_path):
    """Salveaza o singura data diagrama 3D-style a VGG16 cu visualkeras."""
    try:
        import visualkeras
        visualkeras.layered_view(
            model,
            to_file=str(output_path),
            legend=True,
            spacing=25,
            scale_xy=1.6,
            max_z=160,
            draw_volume=True,
        )
        print(f"[viz] diagrama arhitectura salvata: {output_path}")
        return True
    except Exception as exc:
        print(f"[viz] visualkeras indisponibil ({exc}) — sar peste diagrama 3D")
        return False


def _setup_tensorboard(root="logs/vgg16_extraction"):
    """Initializeaza un FileWriter TensorBoard cu timestamp. Returneaza (writer, log_dir)."""
    try:
        import datetime
        import tensorflow as tf
        log_dir = Path(root) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = tf.summary.create_file_writer(str(log_dir))
        print(f"[viz] TensorBoard logs: {log_dir}")
        print(f"[viz] porneste: tensorboard --logdir {root} --port 6006")
        return writer, log_dir
    except Exception as exc:
        print(f"[viz] TensorBoard indisponibil ({exc})")
        return None, None


def _salveaza_feature_maps_png(raw_img, activari, batch_idx, pictor, durata, output_path):
    """Salveaza PNG cu input + 8 feature maps per layer din LAYERE_VIZ (5 layere)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(activari) + 1
    n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
    if n_rows == 1:
        axes = np.array([axes])

    for col in range(n_cols):
        axes[0, col].axis("off")
    axes[0, 0].imshow(np.clip(raw_img, 0, 255).astype(np.uint8))
    axes[0, 0].set_title("Input (224x224)", fontsize=9)
    axes[0, 3].text(
        0.5, 0.5,
        f"Batch #{batch_idx}\nPictor: {pictor}\nDurata: {durata:.2f}s",
        ha="center", va="center", fontsize=11,
        transform=axes[0, 3].transAxes,
    )

    for row_idx, (nume_strat, act) in enumerate(activari.items(), start=1):
        sample = act[0]
        n_canale = min(n_cols, sample.shape[-1])
        norme = sample.mean(axis=(0, 1))
        top_idx = np.argsort(-norme)[:n_canale]
        for col in range(n_cols):
            ax = axes[row_idx, col]
            if col < n_canale:
                ch = top_idx[col]
                ax.imshow(sample[:, :, ch], cmap="viridis")
                if col == 0:
                    ax.set_title(
                        f"{nume_strat}  shape={tuple(sample.shape)}",
                        fontsize=9, loc="left",
                    )
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        f"VGG16 — feature maps live (top 8 canale per layer)",
        fontsize=13, y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _log_tensorboard(writer, raw_batch, activari, batch_idx, durata, throughput, sarite):
    """Scrie imagini + feature maps + histograme + scalari in TensorBoard."""
    if writer is None:
        return
    import tensorflow as tf
    with writer.as_default():
        if raw_batch:
            imgs = np.stack(raw_batch[: min(4, len(raw_batch))], axis=0) / 255.0
            tf.summary.image("input/imagini_originale", imgs, max_outputs=4, step=batch_idx)
        for nume_strat, act in activari.items():
            n_canale = min(8, act.shape[-1])
            sample = act[0:1, :, :, :n_canale]
            sample = tf.transpose(sample, [3, 1, 2, 0])
            tf.summary.image(
                f"feature_maps/{nume_strat}",
                sample, max_outputs=n_canale, step=batch_idx,
            )
            tf.summary.histogram(f"activari/{nume_strat}", act, step=batch_idx)
        tf.summary.scalar("perf/timp_batch_s", durata, step=batch_idx)
        tf.summary.scalar("perf/throughput_img_per_s", throughput, step=batch_idx)
        tf.summary.scalar("perf/sarite_cumulat", sarite, step=batch_idx)
    writer.flush()


def extragere_cnn_vgg16(image_paths, batch_size=32):
    """
    Extrage vectori 4096-dim din stratul fc2 al VGG16 cu afișaj live multi-panou.

    Forward pass-ul rulează în mod eager strat-cu-strat (Conv2D → Pool → … → fc2),
    iar dashboard-ul `rich` actualizează în timp real: bara de batch-uri, bara
    de strat curent, statistici (pictor, throughput, params/strat, ETA, etapă)
    și log-ul ultimelor batch-uri finalizate.

    În paralel salvează vizualizări neurale:
      • data_out/VGG16_arhitectura.png  — diagrama 3D statica (visualkeras)
      • data_out/VGG16_feature_maps_live.png — feature maps reale, refresh/batch
      • logs/vgg16_extraction/<ts>/ — log TensorBoard (deschide cu `tensorboard --logdir logs`)
    """
    import time
    from collections import deque

    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.preprocessing import image

    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    base = VGG16(weights="imagenet", include_top=True)
    layers_lant = []
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        layers_lant.append(layer)
        if layer.name == "fc2":
            break

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    diagrama_path = DATA_OUT / "VGG16_arhitectura.png"
    feature_maps_path = DATA_OUT / "VGG16_feature_maps_live.png"
    _diagrama_visualkeras(base, diagrama_path)
    tb_writer, tb_log_dir = _setup_tensorboard()

    n_total = len(image_paths)
    n_batches = (n_total + batch_size - 1) // batch_size
    n_straturi = len(layers_lant)

    features, paths_ok = [], []
    imagini_ok, sarite = 0, 0
    timpi_batch = []
    log_recent = deque(maxlen=6)

    progres_batch = Progress(
        TextColumn("[bold cyan]Batch"),
        BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
        TextColumn("[cyan]{task.completed:>3}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    )
    task_batch = progres_batch.add_task("", total=n_batches)

    progres_strat = Progress(
        TextColumn("[bold magenta]Strat"),
        BarColumn(bar_width=40, complete_style="magenta", finished_style="green"),
        TextColumn("[magenta]{task.completed:>2}/{task.total}"),
        TextColumn("[white]{task.description}"),
    )
    task_strat = progres_strat.add_task("idle", total=n_straturi)

    stare = {
        "pictor": "-",
        "etapa": "asteptare",
        "layer_nume": "-",
        "layer_tip": "-",
        "shape": "-",
        "params_strat": 0,
        "throughput": 0.0,
        "start_total": time.time(),
    }

    def panou_arhitectura():
        return Text.from_markup(
            "[dim]Input[/dim] [cyan]224×224×3[/cyan]  →  "
            "[yellow]B1[/yellow][dim]:conv64×2·pool[/dim]  →  "
            "[yellow]B2[/yellow][dim]:conv128×2·pool[/dim]  →  "
            "[yellow]B3[/yellow][dim]:conv256×3·pool[/dim]  →  "
            "[yellow]B4[/yellow][dim]:conv512×3·pool[/dim]  →  "
            "[yellow]B5[/yellow][dim]:conv512×3·pool[/dim]  →  "
            "[green]Flatten[/green]  →  [green]FC4096[/green]  →  "
            "[bold green on black] fc2 [/bold green on black]"
        )

    def panou_stat():
        tabel = Table.grid(padding=(0, 2))
        tabel.add_column(style="bold dim", justify="right")
        tabel.add_column(style="bold white")
        tabel.add_row("Pictor curent:", f"[cyan]{stare['pictor']}[/cyan]")
        tabel.add_row("Etapă:", stare["etapa"])
        tabel.add_row(
            "Strat activ:",
            f"[bold]{stare['layer_nume']}[/bold]  "
            f"[dim]({stare['layer_tip']})[/dim]",
        )
        tabel.add_row("Output shape:", f"[yellow]{stare['shape']}[/yellow]")
        tabel.add_row("Params strat:", f"{stare['params_strat']:,}")
        tabel.add_row("Imagini OK:", f"{imagini_ok} / {n_total}")
        tabel.add_row("Sărite:", str(sarite))
        tabel.add_row("Throughput:", f"{stare['throughput']:.2f} img/s")
        medie = (sum(timpi_batch) / len(timpi_batch)) if timpi_batch else 0.0
        tabel.add_row("Medie / batch:", f"{medie:.2f} s")
        scurs = time.time() - stare["start_total"]
        tabel.add_row(
            "Total scurs:",
            f"{int(scurs // 60):02d}:{int(scurs % 60):02d}",
        )
        return tabel

    def panou_viz():
        tabel = Table.grid(padding=(0, 2))
        tabel.add_column(style="bold dim", justify="right")
        tabel.add_column(style="bold white")
        tabel.add_row("Diagrama 3D:", f"[blue]{diagrama_path}[/blue]")
        tabel.add_row(
            "Feature maps live:",
            f"[blue]{feature_maps_path}[/blue]  [dim](refresh / batch)[/dim]",
        )
        if tb_log_dir is not None:
            tabel.add_row(
                "TensorBoard:",
                f"[blue]{tb_log_dir}[/blue]",
            )
            tabel.add_row(
                "Comanda TB:",
                "[green]tensorboard --logdir logs/vgg16_extraction --port 6006[/green]",
            )
        return tabel

    def panou_log():
        tabel = Table.grid(padding=(0, 1))
        if log_recent:
            for linie in log_recent:
                tabel.add_row(linie)
        else:
            tabel.add_row(Text("(niciun batch finalizat încă)", style="dim"))
        return tabel

    def dashboard():
        return Group(
            Panel(
                panou_arhitectura(),
                title="VGG16 — arhitectură",
                border_style="cyan",
                padding=(0, 1),
            ),
            Panel(
                progres_batch,
                title=f"Progres total ({n_total} imagini)",
                border_style="green",
                padding=(0, 1),
            ),
            Panel(
                progres_strat,
                title=f"Forward pass — strat activ ({n_straturi} layere până la fc2)",
                border_style="magenta",
                padding=(0, 1),
            ),
            Panel(
                panou_stat(),
                title="Statistici live",
                border_style="yellow",
                padding=(0, 1),
            ),
            Panel(
                panou_viz(),
                title="Vizualizari neurale",
                border_style="blue",
                padding=(0, 1),
            ),
            Panel(
                panou_log(),
                title="Ultimele batch-uri",
                border_style="dim",
                padding=(0, 1),
            ),
        )

    console = Console()
    with Live(dashboard(), console=console, refresh_per_second=12) as live:
        for batch_idx, start in enumerate(range(0, n_total, batch_size)):
            batch_paths = image_paths[start : start + batch_size]
            stare["pictor"] = (
                Path(batch_paths[0]).parent.name.replace("_", " ")
                if batch_paths
                else "?"
            )

            stare["etapa"] = "[cyan]încărcare + preprocesare[/cyan]"
            live.update(dashboard())

            t0 = time.time()
            arr_list, ok_list, raw_list = [], [], []
            for p in batch_paths:
                try:
                    img = image.load_img(p, target_size=(224, 224))
                    arr = image.img_to_array(img)
                    raw_list.append(arr)
                    arr_list.append(arr)
                    ok_list.append(p)
                except Exception as exc:
                    sarite += 1
                    log_recent.append(
                        Text.from_markup(f"[red][skip][/red] {Path(p).name}: {exc}")
                    )

            activari_batch = {}
            if arr_list:
                x = tf.constant(
                    preprocess_input(np.stack(arr_list, axis=0)), dtype=tf.float32
                )
                stare["etapa"] = "[magenta]forward pass (eager, strat cu strat)[/magenta]"
                progres_strat.reset(task_strat, total=n_straturi)

                for li, layer in enumerate(layers_lant):
                    stare["layer_nume"] = layer.name
                    stare["layer_tip"] = type(layer).__name__
                    x = layer(x)
                    if layer.name in LAYERE_VIZ:
                        activari_batch[layer.name] = x.numpy().copy()
                    stare["shape"] = str(tuple(x.shape))
                    stare["params_strat"] = int(layer.count_params())
                    progres_strat.update(
                        task_strat,
                        completed=li + 1,
                        description=f"[bold]{layer.name}[/bold]  "
                        f"[dim]→ {tuple(x.shape)}[/dim]",
                    )
                    live.update(dashboard())

                feats = x.numpy()
                features.append(feats)
                paths_ok.extend(ok_list)
                imagini_ok += len(ok_list)

            durata = time.time() - t0
            timpi_batch.append(durata)
            stare["throughput"] = (len(ok_list) / durata) if durata > 0 else 0.0

            if activari_batch and raw_list:
                stare["etapa"] = "[green]salvare vizualizari[/green]"
                live.update(dashboard())
                try:
                    _salveaza_feature_maps_png(
                        raw_list[0], activari_batch, batch_idx + 1,
                        stare["pictor"], durata, feature_maps_path,
                    )
                except Exception as exc:
                    log_recent.append(
                        Text.from_markup(f"[yellow][viz][/yellow] PNG: {exc}")
                    )
                try:
                    _log_tensorboard(
                        tb_writer, raw_list, activari_batch, batch_idx + 1,
                        durata, stare["throughput"], sarite,
                    )
                except Exception as exc:
                    log_recent.append(
                        Text.from_markup(f"[yellow][viz][/yellow] TB: {exc}")
                    )
            log_recent.append(
                Text.from_markup(
                    f"[dim]#{batch_idx + 1:03d}[/dim]  "
                    f"[cyan]{stare['pictor'][:22]:<22}[/cyan]  "
                    f"[bold]{durata:5.1f}s[/bold]  "
                    f"[green]OK[/green] {len(ok_list)} img"
                )
            )

            progres_batch.update(task_batch, advance=1)
            live.update(dashboard())

    medie = float(np.mean(timpi_batch)) if timpi_batch else 0.0
    print(
        f"\n  [OK] {imagini_ok} imagini procesate"
        f"  |  {sarite} sărite"
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
