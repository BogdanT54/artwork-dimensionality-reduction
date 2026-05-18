# Reducerea Dimensionalității pe Picturi (Best Artworks of All Time)

Proiect pentru cursul **Tehnici de Învățare Automată și Aplicații**, tema **C. Reducerea dimensionalității**.

Aplică șapte metode obligatorii (PCA, Analiză Factorială, NMF, ICA, Kernel PCA, MDS, t-SNE) pe vectori semantici extrași cu VGG16 din ~8.446 picturi ale celor 50 cei mai influenți pictori (Kaggle: `ikarus777/best-artworks-of-all-time`).

## Structură

```
artwork-dimensionality-reduction/
├── data_in/best_artworks/   # dataset descărcat de Kaggle + features_cnn.csv generat
├── data_out/                # toate PDF + CSV (golit la fiecare rulare main)
├── grafice.py               # funcții de vizualizare
├── functii.py               # helper-e de calcul
├── reducere_dim.py          # logica celor 7 metode
└── main_*.py                # un main per metodă, rulate manual
```

## Setup

1. Instalare dependențe:
   ```bash
   pip install -r requirements.txt
   ```

2. Configurare Kaggle API (o singură dată):
   - cont gratuit pe https://www.kaggle.com
   - Account → API → Create New Token → descarcă `kaggle.json`
   - mută-l în `~/.kaggle/kaggle.json` și `chmod 600 ~/.kaggle/kaggle.json`

3. Pas 0 — generare vectori CNN (~15 min cu GPU, ~1h CPU):
   ```bash
   python main_vectorizare.py
   ```
   Produce `data_in/best_artworks/features_cnn.csv` (~8446 × 4101 coloane).

## Rulare metode

Fiecare main se rulează independent și produce PDF + CSV în `data_out/`:

```bash
python main_pca.py        # PCA + eigenpicturi + corelograma
python main_fa.py         # Analiză Factorială + Bartlett + KMO + Varimax
python main_nmf.py        # NMF + Elbow eroare + componente ca picturi
python main_ica.py        # FastICA + entropie + kurtosis
python main_kpca.py       # Kernel PCA RBF vs PCA liniar
python main_mds.py        # MDS pe centroide pictori + Shepard + heatmap
python main_tsne.py       # t-SNE cu 4 valori de perplexity
python main_comparatie.py # scatter grid + timpi + silhouette per metoda
```

## Notă tehnică

VGG16 stratul `fc2` are activare ReLU → vectorii sunt ≥ 0, deci NMF funcționează direct fără MinMaxScaler.
Pentru eigenpicturi: în spațiu CNN 4096-dim nu se pot reconstrui imagini pixel-wise, deci afișăm TOP 5 picturi reale cu scor maxim/minim pe fiecare componentă — interpretare semantică directă.
