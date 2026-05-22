"""
Pas 0b: Fine-tunează VGG16 pe setul de picturi (clasificare artiști, 50 clase).
Suprascrie features_cnn.csv cu vectori fc2 extrași din modelul fine-tunat.

Rulat DUPĂ 00_main_vectorizare.py (care descarcă datele).

Abordare two-phase cu best practices 2024:
  Faza 1 — backbone înghețat, antrenăm doar capul de clasificare (Adam, LR=1e-3)
  Faza 2 — dezghețăm DOAR block5 + FC (cel mai fin control), AdamW cu cosine decay

Îmbunătățiri față de VGG16 baseline:
  - Augmentare puternică: RandomRotation, RandomZoom, RandomTranslation + flip/brightness
  - Dropout crescut la 0.5 pe head
  - AdamW (weight_decay=1e-4) = L2 implicit pe toți parametrii antrenați
  - Class weights: sqrt(max_count / count) — robust la dezechilibrul VGG16 artworks
  - Label smoothing 0.1 via CategoricalCrossentropy (reduce overconfidence)
  - Cosine decay în faza 2 (mai stabil decât ReduceLROnPlateau)
  - Dezgheță NUMAI block5 (nu block4) — reduce semnificativ overfitting-ul

Durată estimată: 1-2h GPU P100.
"""
import gc
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import functii

DATA_IN      = functii.DATA_IN
IMAGES       = DATA_IN / "images"
ARTISTS_CSV  = DATA_IN / "artists.csv"
FEATURES_CSV = DATA_IN / "features_cnn.csv"
MODEL_PATH   = DATA_IN / "vgg16_finetuned.keras"

BATCH_TRAIN   = 16
BATCH_EXTRACT = 32
VAL_SPLIT     = 0.15
EPOCHS_HEAD   = 15   # faza 1 (mai mult — backbone înghețat, nu riscăm overfitting)
EPOCHS_FT     = 25   # faza 2 (cu early stopping + cosine decay)
LABEL_SMOOTH  = 0.1
WEIGHT_DECAY  = 1e-4

# Faza 2: dezghețăm NUMAI block5 + FC (nu block4 — reduce overfitting)
STRATURI_DEZGHETATE = {
    "block5_conv1", "block5_conv2", "block5_conv3",
    "fc1", "fc2",
    "dropout_head", "artist_pred",
}


# ─── augmentare ───────────────────────────────────────────────────────────────

def _augmenteaza(img, label):
    """Augmentare agresivă pentru dataset mic/dezechilibrat."""
    import tensorflow as tf
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.20)
    img = tf.image.random_saturation(img, lower=0.70, upper=1.30)
    img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
    img = tf.image.random_hue(img, max_delta=0.05)
    # RandomRotation și RandomZoom prin keras layers
    img = tf.expand_dims(img, 0)
    img = tf.keras.layers.RandomRotation(0.08)(img, training=True)
    img = tf.keras.layers.RandomZoom((-0.10, 0.10))(img, training=True)
    img = tf.keras.layers.RandomTranslation(0.05, 0.05)(img, training=True)
    img = tf.squeeze(img, 0)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def _preprocess_vgg(img, label):
    import tensorflow as tf
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, label


def _to_onehot(img, label, n_classes):
    import tensorflow as tf
    return img, tf.one_hot(tf.cast(label, tf.int32), n_classes)


# ─── date ────────────────────────────────────────────────────────────────────

def _calcul_class_weights(images_dir):
    """
    Calculează class_weight dict cu sqrt inverse frequency.
    sqrt(max_count / count) e mai robust decât inverse pur pentru dezechilibre mari.
    """
    extensii = {".jpg", ".jpeg", ".png", ".bmp"}
    class_names = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    counts = np.array([
        sum(1 for f in (images_dir / cls).iterdir() if f.suffix.lower() in extensii)
        for cls in class_names
    ], dtype=float)
    max_count = counts.max()
    weights = np.sqrt(max_count / np.clip(counts, 1, None))
    weights = weights / weights.mean()  # normalizăm la media 1
    return {i: float(w) for i, w in enumerate(weights)}, class_names


def _pregateste_dataset(images_dir, val_split, batch_size, n_classes):
    """Construiește tf.data.Dataset cu augmentare + one-hot labels."""
    import tensorflow as tf

    kw = dict(
        directory=str(images_dir),
        image_size=(224, 224),
        batch_size=batch_size,
        validation_split=val_split,
        seed=42,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training",  **kw)
    val_ds   = tf.keras.utils.image_dataset_from_directory(subset="validation", **kw)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds
                .map(_augmenteaza,                       num_parallel_calls=AUTOTUNE)
                .map(_preprocess_vgg,                    num_parallel_calls=AUTOTUNE)
                .map(lambda x, y: _to_onehot(x, y, n_classes), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds = (val_ds
              .map(_preprocess_vgg,                      num_parallel_calls=AUTOTUNE)
              .map(lambda x, y: _to_onehot(x, y, n_classes), num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))

    return train_ds, val_ds


# ─── model ───────────────────────────────────────────────────────────────────

def _construieste_model(n_classes):
    """VGG16 ImageNet + dropout 0.5 + cap clasificare."""
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.layers import Dropout, Dense
    from tensorflow.keras.models import Model

    base = VGG16(weights="imagenet", include_top=True)
    x = base.get_layer("fc2").output
    x = Dropout(0.5, name="dropout_head")(x)
    output = Dense(n_classes, activation="softmax", name="artist_pred")(x)
    return Model(inputs=base.input, outputs=output, name="vgg16_artisti")


# ─── antrenare ───────────────────────────────────────────────────────────────

def _antrenare(model, train_ds, val_ds, class_weight_dict, steps_per_epoch):
    import tensorflow as tf

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1,
    )

    # ── Faza 1: backbone înghețat, antrenăm capul ──────────────────────────
    print("\n" + "═" * 60)
    print("  Faza 1 / 2 — antrenare cap clasificare (backbone înghețat)")
    print("═" * 60)

    for layer in model.layers:
        layer.trainable = layer.name in {"dropout_head", "artist_pred"}

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=WEIGHT_DECAY),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    model.summary(line_length=80,
                  print_fn=lambda s: print(s) if "Trainable" in s or "Non-trainable" in s else None)

    t0 = time.time()
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )
    best_val1 = max(hist1.history["val_accuracy"])
    print(f"  Faza 1 finalizată în {(time.time()-t0)/60:.1f} min  "
          f"| best val_acc = {best_val1:.4f}")

    # ── Faza 2: dezghețăm NUMAI block5 + FC, cosine decay ─────────────────
    print("\n" + "═" * 60)
    print("  Faza 2 / 2 — fine-tuning block5+FC (AdamW + cosine decay)")
    print("═" * 60)

    for layer in model.layers:
        layer.trainable = layer.name in STRATURI_DEZGHETATE

    # Cosine decay: LR pornește de la 1e-5, scade la ~0 în EPOCHS_FT epoci
    total_steps = EPOCHS_FT * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=total_steps,
        alpha=1e-7,
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH), monitor="val_accuracy",
        save_best_only=True, verbose=1,
    )
    # Faza 2 nu mai are ReduceLROnPlateau — cosine decay gestionează singur LR-ul
    callbacks_ft = [early_stop, checkpoint_cb]

    t0 = time.time()
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FT,
        class_weight=class_weight_dict,
        callbacks=callbacks_ft,
        verbose=1,
    )
    best_val2 = max(hist2.history["val_accuracy"])
    print(f"  Faza 2 finalizată în {(time.time()-t0)/60:.1f} min  "
          f"| best val_acc = {best_val2:.4f}")

    return hist1.history, hist2.history


# ─── grafic antrenare ─────────────────────────────────────────────────────────

def _salveaza_grafic_antrenare(hist1, hist2):
    ep1 = len(hist1["accuracy"])
    ep2 = len(hist2["accuracy"])
    x1 = np.arange(1, ep1 + 1)
    x2 = np.arange(ep1 + 1, ep1 + ep2 + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, titlu, ylab in [
        (ax_acc,  "accuracy", "Acuratețe antrenare / validare",  "Acuratețe"),
        (ax_loss, "loss",     "Loss antrenare / validare (CE)",  "Loss"),
    ]:
        ax.plot(x1, hist1[key],          "b-o",  ms=4, label="train faza 1")
        ax.plot(x1, hist1[f"val_{key}"], "b--s", ms=4, label="val faza 1")
        ax.plot(x2, hist2[key],          "r-o",  ms=4, label="train faza 2")
        ax.plot(x2, hist2[f"val_{key}"], "r--s", ms=4, label="val faza 2")
        ax.axvline(ep1 + 0.5, color="gray", ls=":", lw=1.5, label="start fine-tune")
        ax.set_xlabel("Epocă")
        ax.set_ylabel(ylab)
        ax.set_title(titlu)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fine-tuning VGG16 — block5 only, AdamW, cosine decay, label_smooth=0.1", fontsize=12)
    plt.tight_layout()
    functii.DATA_OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(functii.DATA_OUT / "Training_finetune.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[OK] grafic antrenare salvat: data_out/Training_finetune.pdf")


# ─── extragere features ────────────────────────────────────────────────────────

def _extrage_features(model, df_paths):
    """Extrage vectori fc2 din modelul fine-tunat pentru toate imaginile."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing import image as kimage

    feat_model = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    paths = df_paths["path"].tolist()
    n = len(paths)
    n_batches = (n + BATCH_EXTRACT - 1) // BATCH_EXTRACT
    features, paths_ok, sarite = [], [], 0

    print(f"\n[info] extragere fc2 din modelul fine-tunat ({n} imagini)...")
    t0 = time.time()

    for i, start in enumerate(range(0, n, BATCH_EXTRACT)):
        batch = paths[start : start + BATCH_EXTRACT]
        arr_list, ok_list = [], []
        for p in batch:
            try:
                img = kimage.load_img(p, target_size=(224, 224))
                arr_list.append(kimage.img_to_array(img))
                ok_list.append(p)
            except Exception:
                sarite += 1
        if arr_list:
            x = preprocess_input(np.stack(arr_list, axis=0))
            feats = feat_model.predict(x, verbose=0)
            features.append(feats)
            paths_ok.extend(ok_list)
        if (i + 1) % 25 == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"  batch {i+1:>3}/{n_batches}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min", end="\r")
        if i % 10 == 0:
            gc.collect()

    print(f"\n[OK] {len(paths_ok)} imagini extrase, {sarite} sărite  "
          f"({(time.time()-t0)/60:.1f} min total)")
    return np.vstack(features), paths_ok


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if not IMAGES.exists() or not ARTISTS_CSV.exists():
        sys.exit("[eroare] rulează mai întâi 00_main_vectorizare.py (pentru download date)")

    import tensorflow as tf

    print(f"[info] TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[info] GPU disponibil: {gpus if gpus else 'NU — antrenare pe CPU (lent!)'}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Class weights (sqrt inverse frequency — robust la dezechilibru)
    class_weight_dict, class_names = _calcul_class_weights(IMAGES)
    n_classes = len(class_names)
    max_w = max(class_weight_dict.values())
    min_w = min(class_weight_dict.values())
    print(f"[info] {n_classes} clase; class weights: min={min_w:.3f}, max={max_w:.3f} "
          f"(sqrt inverse frequency)")

    # Dataset
    print("\n[info] pregătire dataset cu augmentare extinsă...")
    train_ds, val_ds = _pregateste_dataset(IMAGES, VAL_SPLIT, BATCH_TRAIN, n_classes)
    # Număr de batches pe epocă (pentru CosineDecay)
    steps_per_epoch = sum(1 for _ in train_ds)
    print(f"[info] {n_classes} clase, batch_size={BATCH_TRAIN}, "
          f"val_split={VAL_SPLIT}, steps/epoch={steps_per_epoch}")

    # Model
    model = _construieste_model(n_classes)
    print(f"[info] model: {model.count_params():,} parametri total")

    # Antrenare
    hist1, hist2 = _antrenare(model, train_ds, val_ds, class_weight_dict, steps_per_epoch)
    _salveaza_grafic_antrenare(hist1, hist2)

    if not MODEL_PATH.exists():
        model.save(str(MODEL_PATH))
    print(f"[OK] model salvat: {MODEL_PATH}")

    # Extragere features cu modelul fine-tunat
    df_paths = functii.colecteaza_paths_si_metadata()
    if len(df_paths) == 0:
        sys.exit("[eroare] niciun fișier imagine găsit.")

    features, paths_ok = _extrage_features(model, df_paths)
    print(f"[OK] features shape = {features.shape}")

    if FEATURES_CSV.exists():
        backup = FEATURES_CSV.with_suffix(".csv.imagenet_backup")
        if not backup.exists():
            FEATURES_CSV.rename(backup)
            print(f"[info] backup → {backup.name}")
        else:
            FEATURES_CSV.unlink()

    df_paths_ok = df_paths.set_index("path").loc[paths_ok].reset_index()
    feat_cols = [f"f{i+1}" for i in range(features.shape[1])]
    df_out = pd.concat(
        [df_paths_ok.reset_index(drop=True),
         pd.DataFrame(features, columns=feat_cols)],
        axis=1,
    )
    temp = FEATURES_CSV.with_suffix(".csv.tmp")
    df_out.to_csv(temp, index=False)
    temp.replace(FEATURES_CSV)
    print(f"[OK] salvat {FEATURES_CSV}  shape={df_out.shape}")
    print("\nPoți acum rula 01_main_pca.py, 02_main_fa.py etc. cu features fine-tunate.")


if __name__ == "__main__":
    main()
