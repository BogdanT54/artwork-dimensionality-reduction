"""
Pas 0: descarcă datasetul Best Artworks (dacă lipsește), extrage vectori VGG16 fc2
și salvează features_cnn.csv. Rulat o singură dată.
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


def _repara_mojibake(nume, target_set):
    """
    Încearcă să repare un nume de folder corupt de encoding-uri greșite.

    Exemple reale întâlnite cu Albrecht Dürer:
      • 'Albrecht_DuΓòá├¬rer'   (dublu mojibake cp437 ↔ utf-8)
      • 'Albrecht_Du╠êrer'      (NFD UTF-8 reinterpretat ca cp437)
      • 'Albrecht_Dürer'  (NFD pur — combining diaeresis)

    Strategia: încercăm chains de encode(cp437/cp850/cp1252/latin-1) → decode(utf-8)
    de până la 3 niveluri, plus normalizare NFC, și verificăm dacă rezultatul
    se potrivește cu un nume așteptat din `artists.csv`.
    """
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
    """
    Repară numele de foldere pe disc:
      1. NFD → NFC (combining diacritic → caracter compus)
      2. mojibake (cp437/cp850/cp1252/latin-1 ↔ utf-8) — vezi exemplele Albrecht Dürer

    Folosește `artists.csv` ca referință pentru numele așteptate, ca să poată
    repara corect chiar și după mai multe nivele de corupere.
    """
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
            target = nfc
            motiv = "NFD→NFC"
        elif nfc not in target_set and target_set:
            reparat = _repara_mojibake(original, target_set)
            if reparat is not None:
                target = reparat
                motiv = "mojibake"

        if target is None or target == original:
            continue

        dest = d.parent / target
        if dest.exists():
            # Destinația există deja (corectă) — folderul corupt e gol sau duplicat, ștergem
            import shutil
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
                    print(f"[unicode] șters folder gol corupt: {original!r}")
                else:
                    shutil.rmtree(str(d))
                    print(f"[unicode] șters folder corupt (cu conținut mutat deja): {original!r}")
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
    """Afișează un raport complet: care pictori au folder pe disc și care lipsesc."""
    if not IMAGES.exists():
        print("[warn] directorul images/ nu există încă")
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
        print(f"  [LIPSESC {len(lipsa)} pictori]")
        for nume, folder in sorted(lipsa):
            print(f"    ✗  {nume:<30}  (folder asteptat: {folder})")
        print()
        # Show what's actually on disk for debugging
        print(f"  Foldere existente pe disc ({len(foldere_disc)} total):")
        for f in sorted(foldere_disc):
            print(f"    •  {f}")
    else:
        print(f"  [OK] toți cei {len(gasiti)} pictori au folder pe disc")

    extra = foldere_disc - {unicodedata.normalize("NFC", r["folder"]) for _, r in df_art.iterrows()}
    if extra:
        print(f"\n  Foldere extra pe disc (nu sunt în artists.csv): {sorted(extra)}")

    print(f"{'─'*55}\n")


def descarca_kaggle():
    """Descarcă datasetul de pe Kaggle dacă nu există local — via API Python (nu CLI)."""
    if IMAGES.exists() and ARTISTS_CSV.exists():
        print(f"[OK] dataset deja prezent la {DATA_IN}")
    else:
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
            print(f"[info] layout corectat: images/images/ → images/")

    # Always run Unicode normalization and verification — catches both fresh and
    # existing downloads that have NFD folder names on disk.
    _normalizeaza_foldere_unicode()
    if ARTISTS_CSV.exists():
        df_art = pd.read_csv(ARTISTS_CSV)
        df_art["folder"] = df_art["name"].str.replace(" ", "_", regex=False)
        _verifica_foldere_dupa_descarcare(df_art)


def colecteaza_paths_si_metadata():
    return functii.colecteaza_paths_si_metadata()


def main():
    descarca_kaggle()

    if FEATURES_CSV.exists():
        size_mb = FEATURES_CSV.stat().st_size / 1e6
        print(f"[OK] {FEATURES_CSV} deja există ({size_mb:.1f} MB) — sar peste extragere.")
        print(f"[info] sterge manual fisierul daca vrei sa re-extragi: rm {FEATURES_CSV}")
        return

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

    # Scrie atomic: temp file -> rename. Un crash mid-write nu poate corupe CSV-ul.
    temp_csv = FEATURES_CSV.with_suffix(".csv.tmp")
    df_out.to_csv(temp_csv, index=False)
    temp_csv.replace(FEATURES_CSV)
    print(f"[OK] salvat {FEATURES_CSV} cu shape {df_out.shape}")


if __name__ == "__main__":
    main()
