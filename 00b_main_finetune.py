"""
Pas 0b: Fine-tunează VGG16 pe setul de picturi (clasificare artiști, 50 clase).
Suprascrie features_cnn.csv cu vectori fc2 extrași din modelul fine-tunat.

Rulat DUPĂ 00_main_vectorizare.py (care descarcă datele).

Abordare two-phase:
  Faza 1 — backbone înghețat, antrenăm doar capul de clasificare (LR mare)
  Faza 2 — dezghețăm block4+block5+fc1+fc2, fine-tuning cu LR mic

Durată estimată: 1-3h GPU / 8-24h CPU.
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

BATCH_TRAIN   = 16   # mai mic decât extracția — sunt și gradienți
BATCH_EXTRACT = 32
VAL_SPLIT     = 0.15
EPOCHS_HEAD   = 12   # faza 1
EPOCHS_FT     = 30   # faza 2 (cu early stopping)

# Straturi dezghețate în faza 2 (block4, block5, fc1, fc2)
STRATURI_DEZGHETATE = {
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3",
    "fc1", "fc2",
}


# ─── date ────────────────────────────────────────────────────────────────────

def _pregateste_dataset(images_dir, val_split, batch_size, augment_train=True):
    """Construiește tf.data.Dataset din structura de directoare artist/imagine."""
    import tensorflow as tf

    def _augmenteaza(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_saturation(img, lower=0.75, upper=1.25)
        img = tf.image.random_contrast(img, lower=0.80, upper=1.20)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    def _preprocess_vgg(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return img, label

    kw = dict(
        directory=str(images_dir),
        image_size=(224, 224),
        batch_size=batch_size,
        validation_split=val_split,
        seed=42,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training",  **kw)
    val_ds   = tf.keras.utils.image_dataset_from_directory(subset="validation", **kw)
    class_names = train_ds.class_names

    if augment_train:
        train_ds = train_ds.map(_augmenteaza,    num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(_preprocess_vgg, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(_preprocess_vgg,   num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# ─── model ───────────────────────────────────────────────────────────────────

def _construieste_model(n_classes):
    """VGG16 (include_top=True) + dropout + cap clasificare artiști."""
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.layers import Dropout, Dense
    from tensorflow.keras.models import Model

    base = VGG16(weights="imagenet", include_top=True)

    # Înlocuiește ultimul strat (predictions/1000) cu capul nostru
    x = base.get_layer("fc2").output
    x = Dropout(0.4, name="dropout_head")(x)
    output = Dense(n_classes, activation="softmax", name="artist_pred")(x)
    model = Model(inputs=base.input, outputs=output, name="vgg16_artisti")
    return model


# ─── callback feature maps live ──────────────────────────────────────────────

def _construieste_callback_feature_maps(sample_img_path, output_path, freq_batches=25):
    """
    Callback Keras: pe parcursul antrenării, la fiecare `freq_batches` batch-uri,
    extrage feature maps (LAYERE_VIZ) pentru o imagine de referință, salvează
    PNG-ul și îl actualizează inline în notebook (Kaggle/Jupyter).
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as kimage
    from tensorflow.keras.applications.vgg16 import preprocess_input

    img = kimage.load_img(sample_img_path, target_size=(224, 224))
    raw = kimage.img_to_array(img)
    x_in = preprocess_input(np.expand_dims(raw.copy(), axis=0))
    sample_name = Path(sample_img_path).parent.name.replace("_", " ")

    class _CB(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.feat_model = None
            self.names = []
            self.display = functii._KaggleImageDisplay()
            self.global_step = 0
            self.t0 = time.time()

        def _build(self):
            outputs, names = [], []
            for nume in functii.LAYERE_VIZ:
                try:
                    outputs.append(self.model.get_layer(nume).output)
                    names.append(nume)
                except ValueError:
                    pass
            if not outputs:
                return False
            self.feat_model = tf.keras.Model(
                inputs=self.model.input, outputs=outputs, name="feat_viz",
            )
            self.names = names
            return True

        def on_train_batch_end(self, batch, logs=None):
            self.global_step += 1
            if self.global_step % freq_batches != 0:
                return
            if self.feat_model is None and not self._build():
                return
            try:
                outs = self.feat_model.predict(x_in, verbose=0)
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]
                activari = {n: a for n, a in zip(self.names, outs)}
                durata = time.time() - self.t0
                functii._salveaza_feature_maps_png(
                    raw, activari, self.global_step,
                    sample_name, durata, output_path,
                )
                self.display.update(output_path)
            except Exception as exc:
                print(f"[viz] feature maps step {self.global_step}: {exc}")

    return _CB()


# ─── antrenare ───────────────────────────────────────────────────────────────

def _antrenare(model, train_ds, val_ds, sample_img_path=None):
    import tensorflow as tf

    callbacks_baza = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
    ]

    if sample_img_path is not None:
        feature_maps_path = functii.DATA_OUT / "VGG16_finetune_feature_maps_live.png"
        callbacks_baza.append(
            _construieste_callback_feature_maps(
                sample_img_path, feature_maps_path, freq_batches=25,
            )
        )
        print(f"[viz] feature maps live → {feature_maps_path} (refresh la 25 batch-uri)")

    # ── Faza 1: backbone înghețat, antrenăm doar capul ──────────────────────
    print("\n" + "═" * 60)
    print("  Faza 1 / 2 — antrenare cap clasificare (backbone înghețat)")
    print("═" * 60)
    for layer in model.layers:
        layer.trainable = layer.name in {"dropout_head", "artist_pred"}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=80, print_fn=lambda s: print(s) if "Trainable" in s or "Non-trainable" in s else None)

    t0 = time.time()
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD, callbacks=callbacks_baza, verbose=1,
    )
    print(f"  Faza 1 finalizată în {(time.time()-t0)/60:.1f} min  "
          f"| val_acc final = {hist1.history['val_accuracy'][-1]:.3f}")

    # ── Faza 2: dezghețăm block4+block5+FC, fine-tuning cu LR mic ───────────
    print("\n" + "═" * 60)
    print("  Faza 2 / 2 — fine-tuning block4+block5+FC (LR=1e-5)")
    print("═" * 60)
    for layer in model.layers:
        if layer.name in STRATURI_DEZGHETATE or layer.name in {"dropout_head", "artist_pred"}:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_ft = callbacks_baza + [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    t0 = time.time()
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FT, callbacks=callbacks_ft, verbose=1,
    )
    print(f"  Faza 2 finalizată în {(time.time()-t0)/60:.1f} min  "
          f"| val_acc final = {hist2.history['val_accuracy'][-1]:.3f}")

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
        ax.plot(x1, hist1[key],           "b-o",  ms=4, label="train faza 1")
        ax.plot(x1, hist1[f"val_{key}"],  "b--s", ms=4, label="val faza 1")
        ax.plot(x2, hist2[key],           "r-o",  ms=4, label="train faza 2")
        ax.plot(x2, hist2[f"val_{key}"],  "r--s", ms=4, label="val faza 2")
        ax.axvline(ep1 + 0.5, color="gray", ls=":", lw=1.5, label="start fine-tune")
        ax.set_xlabel("Epocă")
        ax.set_ylabel(ylab)
        ax.set_title(titlu)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fine-tuning VGG16 pe Best Artworks (50 artiști)", fontsize=13)
    plt.tight_layout()
    functii.DATA_OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(functii.DATA_OUT / "Training_finetune.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[OK] grafic antrenare salvat: data_out/Training_finetune.pdf")


# ─── extragere features cu modelul fine-tunat ────────────────────────────────

def _extrage_features(model, df_paths):
    """Extrage vectori fc2 din modelul fine-tunat pentru toate imaginile."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing import image as kimage

    feat_model = Model(
        inputs=model.input,
        outputs=model.get_layer("fc2").output,
        name="feat_extractor",
    )

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

    # Date
    print("\n[info] pregătire dataset...")
    train_ds, val_ds, class_names = _pregateste_dataset(IMAGES, VAL_SPLIT, BATCH_TRAIN)
    n_classes = len(class_names)
    print(f"[info] {n_classes} clase (artiști), "
          f"batch_size={BATCH_TRAIN}, val_split={VAL_SPLIT}")

    # Model
    model = _construieste_model(n_classes)
    total_p = model.count_params()
    print(f"[info] model construit: {total_p:,} parametri total")

    # O imagine de referință pentru preview-ul feature maps în timpul antrenării
    sample_img_path = None
    for d in sorted(IMAGES.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                sample_img_path = p
                break
        if sample_img_path is not None:
            break

    # Antrenare
    hist1, hist2 = _antrenare(model, train_ds, val_ds, sample_img_path=sample_img_path)
    _salveaza_grafic_antrenare(hist1, hist2)

    # Salvare model (dacă ModelCheckpoint nu a salvat deja cel mai bun)
    if not MODEL_PATH.exists():
        model.save(str(MODEL_PATH))
    print(f"[OK] model salvat: {MODEL_PATH}")

    # Extragere features cu modelul fine-tunat
    df_paths = functii.colecteaza_paths_si_metadata()
    if len(df_paths) == 0:
        sys.exit("[eroare] niciun fișier imagine găsit.")

    features, paths_ok = _extrage_features(model, df_paths)
    print(f"[OK] features shape = {features.shape}")

    # Salvare features_cnn.csv (suprascrie cel din ImageNet dacă există)
    if FEATURES_CSV.exists():
        backup = FEATURES_CSV.with_suffix(".csv.imagenet_backup")
        FEATURES_CSV.rename(backup)
        print(f"[info] backup ImageNet features → {backup.name}")

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
