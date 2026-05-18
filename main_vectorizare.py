"""
Pas 0: descarcă datasetul Best Artworks (dacă lipsește), extrage vectori VGG16 fc2
și salvează features_cnn.csv. Rulat o singură dată.
"""
import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import functii

DATA_IN = functii.DATA_IN
IMAGES = DATA_IN / "images"
ARTISTS_CSV = DATA_IN / "artists.csv"
FEATURES_CSV = DATA_IN / "features_cnn.csv"


def descarca_kaggle():
    """Descarcă datasetul de pe Kaggle dacă nu există local."""
    if IMAGES.exists() and ARTISTS_CSV.exists():
        print(f"[OK] dataset deja prezent la {DATA_IN}")
        return
    DATA_IN.mkdir(parents=True, exist_ok=True)
    print("[info] descărcare Best Artworks of All Time de pe Kaggle ...")
    cmd = ["kaggle", "datasets", "download",
           "-d", "ikarus777/best-artworks-of-all-time",
           "-p", str(DATA_IN), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("[eroare] kaggle CLI nu este instalat. Rulează: pip install kaggle")
    except subprocess.CalledProcessError as e:
        sys.exit(f"[eroare] descărcare Kaggle eșuată: {e}\n"
                 f"Verifică ~/.kaggle/kaggle.json (chmod 600).")

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
