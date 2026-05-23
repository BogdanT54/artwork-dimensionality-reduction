"""
Pas 0: descarcă datasetul Best Artworks, extrage vectori DINOv2-base (CLS token, 768-dim)
și salvează features_cnn.csv. Rulat o singură dată (sau după re-fine-tuning).

DINOv2 (Meta, 2023) — ViT-B/14 antrenat self-supervised pe 142M imagini.
Produce embeddings CLS de 768 dimensiuni, superioare VGG16 pe sarcini de stil vizual.
ATENȚIE: features DINOv2 pot fi negative (fără ReLU final) → NMF va aplica MinMaxScaler automat.
"""
import os
import sys
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import functii

DATA_IN = functii.DATA_IN
IMAGES = DATA_IN / "images"
ARTISTS_CSV = DATA_IN / "artists.csv"
FEATURES_CSV = DATA_IN / "features_cnn.csv"
FINETUNED_MODEL = DATA_IN / "dinov2_finetuned.pt"

BATCH_SIZE = 32
IMG_SIZE = 224  # DINOv2-base acceptă 224×224


# ─── download + unicode (identic cu VGG16 branch) ────────────────────────────

def _configureaza_kaggle_config_dir():
    if os.environ.get("KAGGLE_CONFIG_DIR") and \
       (Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json").exists():
        return
    candidati = [Path.home() / ".kaggle", Path.cwd() / ".kaggle"]
    for c in candidati:
        if (c / "kaggle.json").exists():
            os.environ["KAGGLE_CONFIG_DIR"] = str(c)
            print(f"[info] kaggle.json găsit la {c}")
            return
    sys.exit("[eroare] kaggle.json nu a fost găsit.")


def _repara_mojibake(nume, target_set):
    nfc = unicodedata.normalize("NFC", nume)
    if nfc in target_set:
        return nfc
    encodings = ["cp437", "cp850", "cp1252", "latin-1"]
    seen = {nume}
    queue = [nume]
    while queue and len(seen) < 30:
        current = queue.pop(0)
        for enc in encodings:
            try:
                reparat = current.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if reparat in seen:
                continue
            seen.add(reparat)
            nfc = unicodedata.normalize("NFC", reparat)
            if nfc in target_set:
                return nfc
            queue.append(reparat)
    return None


def _normalizeaza_foldere_unicode():
    if not IMAGES.exists():
        return
    target_set = set()
    if ARTISTS_CSV.exists():
        df_art = pd.read_csv(ARTISTS_CSV)
        for nume in df_art["name"]:
            target_set.add(unicodedata.normalize("NFC", nume.replace(" ", "_")))

    redenumite_nfd = 0
    redenumite_mojibake = 0
    for d in sorted(IMAGES.iterdir()):
        if not d.is_dir():
            continue
        original = d.name
        nfc = unicodedata.normalize("NFC", original)
        target = None
        motiv = ""
        if nfc != original and nfc in target_set:
            target, motiv = nfc, "NFD→NFC"
        elif nfc not in target_set and target_set:
            reparat = _repara_mojibake(original, target_set)
            if reparat is not None:
                target, motiv = reparat, "mojibake"

        if target is None or target == original:
            continue
        dest = d.parent / target
        if dest.exists():
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
                else:
                    shutil.rmtree(str(d))
                print(f"[unicode] șters folder corupt: {original!r}")
            except Exception as exc:
                print(f"[unicode] nu pot șterge {original!r}: {exc}")
            continue
        d.rename(dest)
        print(f"[unicode] {motiv}: {original!r} → {target!r}")
        if motiv == "NFD→NFC":
            redenumite_nfd += 1
        else:
            redenumite_mojibake += 1

    if redenumite_nfd or redenumite_mojibake:
        print(f"[unicode] reparate: {redenumite_nfd} NFD→NFC, {redenumite_mojibake} mojibake")
    else:
        print("[unicode] toate folderele sunt OK — nicio reparare necesară")


def _verifica_foldere_dupa_descarcare(df_art):
    if not IMAGES.exists():
        return
    foldere_disc = {unicodedata.normalize("NFC", d.name) for d in IMAGES.iterdir() if d.is_dir()}
    gasiti, lipsa = [], []
    for _, row in df_art.iterrows():
        folder_nfc = unicodedata.normalize("NFC", row["folder"])
        if folder_nfc in foldere_disc:
            gasiti.append(row["name"])
        else:
            lipsa.append((row["name"], folder_nfc))

    print(f"\n{'─'*55}")
    print(f"  Verificare foldere: {len(gasiti)}/{len(df_art)} pictori găsiți pe disc")
    print(f"{'─'*55}")
    if lipsa:
        for nume, folder in sorted(lipsa):
            print(f"    ✗  {nume:<30}  (folder: {folder})")
    else:
        print(f"  [OK] toți cei {len(gasiti)} pictori au folder pe disc")
    print(f"{'─'*55}\n")


def descarca_kaggle():
    if IMAGES.exists() and ARTISTS_CSV.exists():
        print(f"[OK] dataset deja prezent la {DATA_IN}")
    else:
        DATA_IN.mkdir(parents=True, exist_ok=True)
        _configureaza_kaggle_config_dir()
        print("[info] descărcare Best Artworks of All Time de pe Kaggle ...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            sys.exit("[eroare] pachetul 'kaggle' nu este instalat.")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("ikarus777/best-artworks-of-all-time",
                                   path=str(DATA_IN), unzip=True, quiet=False)
        nested = IMAGES / "images"
        if nested.exists() and any(nested.iterdir()):
            for sub in nested.iterdir():
                shutil.move(str(sub), str(IMAGES))
            nested.rmdir()
            print("[info] layout corectat: images/images/ → images/")

    _normalizeaza_foldere_unicode()
    if ARTISTS_CSV.exists():
        df_art = pd.read_csv(ARTISTS_CSV)
        df_art["folder"] = df_art["name"].str.replace(" ", "_", regex=False)
        _verifica_foldere_dupa_descarcare(df_art)


# ─── extragere DINOv2 ────────────────────────────────────────────────────────

def _extrage_dinov2(image_paths, model_path=None):
    """
    Extrage embeddings CLS token din DINOv2-base pentru lista de imagini.
    Dacă model_path există, încarcă modelul fine-tunat; altfel folosește frozen backbone.

    Output: numpy array (N, 768), lista paths ok
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Dinov2Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    if model_path is not None and Path(model_path).exists():
        print(f"[info] încărcare model DINOv2 fine-tunat: {model_path}")
        checkpoint = torch.load(str(model_path), map_location=device)
        backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        # Încărcăm greutățile backbone-ului din checkpoint (fără capul de clasificare)
        backbone_state = {k.replace("dinov2.", ""): v
                          for k, v in checkpoint["model_state"].items()
                          if k.startswith("dinov2.")}
        backbone.load_state_dict(backbone_state, strict=False)
        print("[info] greutăți fine-tunate încărcate în backbone")
    else:
        print("[info] model fine-tunat negăsit — se folosesc greutăți DINOv2 frozen (self-supervised LVD-142M)")
        backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")

    backbone = backbone.to(device)
    backbone.eval()

    n_total = len(image_paths)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    features = []
    paths_ok = []
    sarite = 0

    print(f"[info] {n_total} imagini, {n_batches} batch-uri")
    t0 = __import__("time").time()

    for i, start in enumerate(range(0, n_total, BATCH_SIZE)):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        imgs = []
        ok = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                imgs.append(img)
                ok.append(p)
            except Exception:
                sarite += 1

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = backbone(**inputs)
            # CLS token: outputs.last_hidden_state[:, 0, :]
            cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        features.append(cls_tokens)
        paths_ok.extend(ok)

        if (i + 1) % 25 == 0 or (i + 1) == n_batches:
            elapsed = __import__("time").time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"  batch {i+1:>3}/{n_batches}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min", end="\r")

    total_time = __import__("time").time() - t0
    print(f"\n  [OK] {len(paths_ok)} imagini procesate  |  {sarite} sărite  "
          f"|  {device}  |  medie {total_time/n_batches:.2f}s/batch")

    return np.vstack(features), paths_ok


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    descarca_kaggle()

    model_path = FINETUNED_MODEL if FINETUNED_MODEL.exists() else None
    if model_path:
        print(f"[info] model fine-tuned detectat: {model_path}")
        print("[info] features_cnn.csv va fi RE-GENERAT cu greutăți fine-tuned.")
        if FEATURES_CSV.exists():
            FEATURES_CSV.unlink()
            print("[info] features_cnn.csv vechi șters — regenerare cu model fine-tuned.")
    else:
        print("[info] model fine-tuned negăsit — se folosesc greutăți DINOv2 frozen.")

    if FEATURES_CSV.exists():
        size_mb = FEATURES_CSV.stat().st_size / 1e6
        print(f"[OK] {FEATURES_CSV} deja există ({size_mb:.1f} MB) — sar peste extragere.")
        return

    df_paths = functii.colecteaza_paths_si_metadata()
    if len(df_paths) == 0:
        sys.exit("[eroare] niciun fișier imagine găsit.")

    print("[info] extragere DINOv2-base CLS token (768-dim) ...")
    features, paths_ok = _extrage_dinov2(df_paths["path"].tolist(), model_path=model_path)
    print(f"[OK] features shape = {features.shape}")
    print(f"[info] features DINOv2: min={features.min():.3f}, max={features.max():.3f} "
          f"(pot fi negative — NMF va aplica MinMaxScaler automat)")

    df_paths_ok = df_paths.set_index("path").loc[paths_ok].reset_index()
    feat_cols = [f"f{i+1}" for i in range(features.shape[1])]
    df_out = pd.concat([df_paths_ok.reset_index(drop=True),
                        pd.DataFrame(features, columns=feat_cols).reset_index(drop=True)], axis=1)
    DATA_IN.mkdir(parents=True, exist_ok=True)
    temp_csv = FEATURES_CSV.with_suffix(".csv.tmp")
    df_out.to_csv(temp_csv, index=False)
    temp_csv.replace(FEATURES_CSV)
    print(f"[OK] salvat {FEATURES_CSV} cu shape {df_out.shape}")


if __name__ == "__main__":
    main()
