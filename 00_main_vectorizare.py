"""
Pas 0: descarcă datasetul Best Artworks (dacă lipsește), extrage vectori VGG16 fc2
și salvează features_cnn.csv. Rulat o singură dată.
"""
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import functii

DATA_IN = functii.DATA_IN
IMAGES = DATA_IN / "images"
ARTISTS_CSV = DATA_IN / "artists.csv"
FEATURES_CSV = DATA_IN / "features_cnn.csv"


def _configureaza_kaggle_config_dir():
    """
    Caută kaggle.json în ordine: $KAGGLE_CONFIG_DIR → ~/.kaggle → ./.kaggle (workspace).
    Setează KAGGLE_CONFIG_DIR înainte de import-ul kaggle (care citește la import-time).
    """
    if os.environ.get("KAGGLE_CONFIG_DIR") and \
       (Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json").exists():
        return
    candidati = [Path.home() / ".kaggle", Path.cwd() / ".kaggle"]
    for c in candidati:
        if (c / "kaggle.json").exists():
            os.environ["KAGGLE_CONFIG_DIR"] = str(c)
            print(f"[info] kaggle.json găsit la {c}")
            return
    sys.exit("[eroare] kaggle.json nu a fost găsit. Pune-l în ~/.kaggle/kaggle.json "
             "sau în ./.kaggle/kaggle.json din workspace (chmod 600).")


def descarca_kaggle():
    """Descarcă datasetul de pe Kaggle dacă nu există local — via API Python (nu CLI)."""
    if IMAGES.exists() and ARTISTS_CSV.exists():
        print(f"[OK] dataset deja prezent la {DATA_IN}")
        return
    DATA_IN.mkdir(parents=True, exist_ok=True)
    _configureaza_kaggle_config_dir()
    print("[info] descărcare Best Artworks of All Time de pe Kaggle ...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("[eroare] pachetul 'kaggle' nu este instalat. Rulează: pip install kaggle")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"[eroare] autentificare Kaggle eșuată: {e}\n"
                 f"Verifică username + key în kaggle.json.")
    try:
        api.dataset_download_files("ikarus777/best-artworks-of-all-time",
                                    path=str(DATA_IN), unzip=True, quiet=False)
    except Exception as e:
        sys.exit(f"[eroare] descărcare dataset eșuată: {e}")

    # Normalize layout: Kaggle archive may extract to images/images/<artist>/...
    nested = IMAGES / "images"
    if nested.exists() and any(nested.iterdir()):
        for sub in nested.iterdir():
            shutil.move(str(sub), str(IMAGES))
        nested.rmdir()


def colecteaza_paths_si_metadata():
    """Returnează DataFrame cu coloanele: path, artist, stil, epoca, gen."""
    if not ARTISTS_CSV.exists():
        sys.exit(f"[eroare] {ARTISTS_CSV} lipsește.")
    df_art = pd.read_csv(ARTISTS_CSV)
    print(f"[info] {len(df_art)} pictori în artists.csv")

    # name în artists.csv: "Vincent van Gogh"; foldere: "Vincent_van_Gogh"
    df_art["folder"] = df_art["name"].str.replace(" ", "_", regex=False)
    df_art["epoca"] = df_art["years"].apply(functii.deriva_epoca)
    df_art["stil"] = df_art["genre"].apply(functii.deriva_stil)

    rows = []
    extensii = {".jpg", ".jpeg", ".png", ".bmp"}
    for _, row in df_art.iterrows():
        folder = IMAGES / row["folder"]
        if not folder.exists():
            print(f"[warn] folderul lipsește: {folder}")
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in extensii:
                rows.append({
                    "path": str(p),
                    "artist": row["name"],
                    "stil": row["stil"],
                    "epoca": row["epoca"],
                    "gen": row.get("nationality", "necunoscut"),
                })
    df_paths = pd.DataFrame(rows)
    print(f"[info] {len(df_paths)} imagini găsite")
    return df_paths


def main():
    descarca_kaggle()
    df_paths = colecteaza_paths_si_metadata()
    if len(df_paths) == 0:
        sys.exit("[eroare] niciun fișier imagine găsit.")

    print("[info] extragere VGG16 fc2 ...")
    features, paths_ok = functii.extragere_cnn_vgg16(df_paths["path"].tolist(), batch_size=32)
    print(f"[OK] features shape = {features.shape}")

    df_paths_ok = df_paths.set_index("path").loc[paths_ok].reset_index()
    feat_cols = [f"f{i+1}" for i in range(features.shape[1])]
    df_feat = pd.DataFrame(features, columns=feat_cols)
    df_out = pd.concat([df_paths_ok.reset_index(drop=True),
                        df_feat.reset_index(drop=True)], axis=1)
    DATA_IN.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(FEATURES_CSV, index=False)
    print(f"[OK] salvat {FEATURES_CSV} cu shape {df_out.shape}")


if __name__ == "__main__":
    main()
