# Reducerea Dimensionalității pe Picturi (Best Artworks of All Time)

Proiect pentru cursul **Tehnici de Învățare Automată și Aplicații**, tema **C. Reducerea dimensionalității**.

Aplică șapte metode obligatorii (PCA, Analiză Factorială, NMF, ICA, Kernel PCA, MDS, t-SNE) pe vectori semantici extrași cu VGG16 din ~8.446 picturi ale celor 50 cei mai influenți pictori (Kaggle: `ikarus777/best-artworks-of-all-time`).

---

## Structura fișierelor

```
artwork-dimensionality-reduction/
│
├── kaggle.json.template        ← copiază în ~/.kaggle/kaggle.json și completează
│
├── data_in/
│   └── best_artworks/
│       ├── images/             ← descărcat de Kaggle (organizat pe pictor)
│       ├── artists.csv         ← metadata 50 pictori (gen, naționalitate, ani)
│       └── features_cnn.csv    ← GENERAT de 00_main_vectorizare.py
│
├── data_out/                   ← toate PDF-urile și CSV-urile (golit selectiv la fiecare main)
│
├── functii.py                  ← funcții calcul: VGG16 extractor, Bartlett, KMO, stres MDS...
├── grafice.py                  ← funcții vizualizare: scatter, elbow, corelograma, eigenpicturi...
├── reducere_dim.py             ← logica celor 7 metode (aplica_pca, aplica_fa, ...)
│
├── 00_main_vectorizare.py      ← Pas 0 (o singură dată): imagini → features_cnn.csv
├── 01_main_pca.py              ← Pas 1: PCA
├── 02_main_fa.py               ← Pas 2: Analiză Factorială
├── 03_main_nmf.py              ← Pas 3: NMF
├── 04_main_ica.py              ← Pas 4: ICA
├── 05_main_kpca.py             ← Pas 5: Kernel PCA
├── 06_main_mds.py              ← Pas 6: MDS
├── 07_main_tsne.py             ← Pas 7: t-SNE
└── 08_main_comparatie.py       ← Pas 8: comparație cross-metode
```

---

## Setup

### 1. Instalare dependențe

```bash
pip install -r requirements.txt
```

### 2. Configurare Kaggle API

```bash
# Copiază template-ul și completează username + key
cp kaggle.json.template ~/.kaggle/kaggle.json
# Editează fișierul — înlocuiește cele două câmpuri cu datele tale Kaggle
# (Account → Settings → API → Create New Token)
chmod 600 ~/.kaggle/kaggle.json
# Verificare
kaggle datasets list -s "best artworks" | head -3
```

### 3. Generare vectori CNN (Pas 0 — rulat o singură dată)

```bash
python 00_main_vectorizare.py
```

Descarcă automat dataset-ul (dacă lipsește), extrage vectori VGG16 fc2 (4096-dim) și salvează:
`data_in/best_artworks/features_cnn.csv` — ~8.446 rânduri × 4.101 coloane (5 metadata + 4096 features).

Durată estimată: ~15 min cu GPU, ~1h cu CPU.

---

## Rulare metode (Pașii 1–8)

Fiecare script se rulează independent după ce `features_cnn.csv` există.
Outputurile se acumulează în `data_out/` — rularea unui main nu șterge outputurile altora.

```bash
python 01_main_pca.py
python 02_main_fa.py
python 03_main_nmf.py
python 04_main_ica.py
python 05_main_kpca.py
python 06_main_mds.py
python 07_main_tsne.py
python 08_main_comparatie.py   # necesită outputurile pașilor 1–7
```

---

## Fișiere generate în `data_out/`

### PCA (`01_main_pca.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Varianta_PCA.csv` | CSV | Varianță explicată per componentă + procent cumulativ |
| `Scoruri_PCA.csv` | CSV | Scorurile celor 8.446 picturi pe primele 50 componente + metadata |
| `r_xc_PCA.csv` | CSV | Corelații features-componente (matrice loadings ponderată) |
| `Selectie_PCA.csv` | CSV | Nr. componente recomandate per criteriu (Kaiser, 80%, Elbow) |
| `Varianta_PCA.pdf` | PDF | Grafic varianță explicată cumulativă + prag 80% |
| `Elbow_PCA.pdf` | PDF | Scree plot PCA cu marcaj Elbow |
| `Corelograma_PCA.pdf` | PDF | Heatmap corelații features × componente (top 50 features) |
| `Cercul_PCA.pdf` | PDF | Cercul corelațiilor (biplot) |
| `Eigenpicturi_PCA.pdf` | PDF | Top 5 picturi cu scor max/min pe primele 6 componente |
| `Scatter_PCA_artist.pdf` | PDF | Scatter 2D comp1/comp2, colorat pe pictor |
| `Scatter_PCA_stil.pdf` | PDF | Scatter 2D comp1/comp2, colorat pe stil artistic |
| `Scatter_PCA_epoca.pdf` | PDF | Scatter 2D comp1/comp2, colorat pe epocă |
| `Scatter_PCA_gen.pdf` | PDF | Scatter 2D comp1/comp2, colorat pe naționalitate |

### Analiză Factorială (`02_main_fa.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Bartlett_KMO_FA.csv` | CSV | Chi², p-value Bartlett + KMO total + nr. factori selecționați |
| `Varianta_FA.csv` | CSV | Varianță per factor + proporție + cumulativ |
| `Comunalitati_FA.csv` | CSV | Comunalitățile (h²) per feature după Varimax |
| `Incarcare_FA.csv` | CSV | Matricea loadings (features × factori) după rotație Varimax |
| `Scoruri_FA.csv` | CSV | Scorurile factoriale ale celor 8.446 picturi + metadata |
| `Comunalitati_FA.pdf` | PDF | Bar chart comunalități (top 40 features) |
| `Loadings_FA.pdf` | PDF | Heatmap loadings features × factori |
| `Scatter_FA_artist.pdf` | PDF | Scatter 2D F1/F2, colorat pe pictor |
| `Scatter_FA_stil.pdf` | PDF | Scatter 2D F1/F2, colorat pe stil |
| `Scatter_FA_epoca.pdf` | PDF | Scatter 2D F1/F2, colorat pe epocă |
| `Scatter_FA_gen.pdf` | PDF | Scatter 2D F1/F2, colorat pe naționalitate |

### NMF (`03_main_nmf.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Erori_NMF.csv` | CSV | Eroare Frobenius per q ∈ {5, 10, 15, 20, 30, 50} |
| `W_NMF.csv` | CSV | Matricea W (scoruri instanțe) pentru q optim + metadata |
| `H_NMF.csv` | CSV | Matricea H (componente-features) pentru q optim |
| `Elbow_NMF.pdf` | PDF | Curba erorii Frobenius + punct Elbow |
| `Componente_NMF.pdf` | PDF | Top 5 picturi cu activare maximă pe fiecare componentă NMF |
| `Scatter_NMF_artist.pdf` | PDF | Scatter 2D C1/C2, colorat pe pictor |
| `Scatter_NMF_stil.pdf` | PDF | Scatter 2D C1/C2, colorat pe stil |
| `Scatter_NMF_epoca.pdf` | PDF | Scatter 2D C1/C2, colorat pe epocă |
| `Scatter_NMF_gen.pdf` | PDF | Scatter 2D C1/C2, colorat pe naționalitate |

### ICA (`04_main_ica.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Scoruri_ICA.csv` | CSV | Componentele independente ale celor 8.446 picturi + metadata |
| `Entropie_Kurtosis_ICA.csv` | CSV | Entropie diferențială + kurtosis per componentă independentă |
| `Kurtosis_ICA.pdf` | PDF | Bar chart kurtosis — cu cât mai mare, cu atât mai non-gaussian |
| `Entropie_ICA.pdf` | PDF | Bar chart entropie diferențială per IC |
| `Top_picturi_IC1.pdf` | PDF | Top 5 picturi cu activare absolută maximă pe IC1 |
| `Top_picturi_IC2.pdf` | PDF | Top 5 picturi cu activare absolută maximă pe IC2 |
| `Top_picturi_IC3.pdf` | PDF | Top 5 picturi cu activare absolută maximă pe IC3 |
| `Top_picturi_IC4.pdf` | PDF | Top 5 picturi cu activare absolută maximă pe IC4 |
| `Scatter_ICA_artist.pdf` | PDF | Scatter 2D IC1/IC2, colorat pe pictor |
| `Scatter_ICA_stil.pdf` | PDF | Scatter 2D IC1/IC2, colorat pe stil |
| `Scatter_ICA_epoca.pdf` | PDF | Scatter 2D IC1/IC2, colorat pe epocă |
| `Scatter_ICA_gen.pdf` | PDF | Scatter 2D IC1/IC2, colorat pe naționalitate |

### Kernel PCA (`05_main_kpca.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Scoruri_KPCA.csv` | CSV | Scoruri Kernel PCA (RBF) pe sub-eșantion + metadata |
| `Scoruri_PCA_referinta_KPCA.csv` | CSV | Scoruri PCA liniar pe același sub-eșantion (referință) |
| `Comparatie_KPCA_PCA_artist.pdf` | PDF | Side-by-side PCA liniar vs Kernel PCA, colorat pe pictor |
| `Comparatie_KPCA_PCA_stil.pdf` | PDF | Side-by-side, colorat pe stil |
| `Comparatie_KPCA_PCA_epoca.pdf` | PDF | Side-by-side, colorat pe epocă |
| `Comparatie_KPCA_PCA_gen.pdf` | PDF | Side-by-side, colorat pe naționalitate |

### MDS (`06_main_mds.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Distante_MDS.csv` | CSV | Matrice distanțe euclidiene 50×50 între centroizii pictorilor |
| `Scoruri_MDS.csv` | CSV | Scoruri MDS pentru q optim (câte un rând per pictor) |
| `Stres_MDS.csv` | CSV | Stresul Kruskal-1 per număr de componente q |
| `Stres_MDS.pdf` | PDF | Curba stresului MDS — Elbow pentru alegerea lui q |
| `Shepard_MDS.pdf` | PDF | Diagrama Shepard: distanțe originale vs distanțe în spațiu MDS |
| `Heatmap_Distante_MDS.pdf` | PDF | Heatmap distanțe euclidiene 50×50 pictori |
| `Scatter_MDS_pictori.pdf` | PDF | Scatter 2D etichetat cu numele fiecărui pictor |

### t-SNE (`07_main_tsne.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Scoruri_tSNE_perp5.csv` | CSV | Embedding 2D cu perplexity=5 + metadata |
| `Scoruri_tSNE_perp30.csv` | CSV | Embedding 2D cu perplexity=30 + metadata |
| `Scoruri_tSNE_perp50.csv` | CSV | Embedding 2D cu perplexity=50 + metadata |
| `Scoruri_tSNE_perp100.csv` | CSV | Embedding 2D cu perplexity=100 + metadata |
| `KL_tSNE.csv` | CSV | Divergența KL finală per valoare de perplexity |
| `Grid_tSNE_artist.pdf` | PDF | Grid 2×2 scatter (4 perplexity), colorat pe pictor |
| `Grid_tSNE_stil.pdf` | PDF | Grid 2×2, colorat pe stil artistic |
| `Grid_tSNE_epoca.pdf` | PDF | Grid 2×2, colorat pe epocă |
| `Grid_tSNE_gen.pdf` | PDF | Grid 2×2, colorat pe naționalitate |

### Comparație cross-metode (`08_main_comparatie.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Silhouette_comparatie.csv` | CSV | Silhouette score per metodă pe label 'artist' (separabilitate 2D) |
| `Timpi_executie.csv` | CSV | Timpi de execuție (secunde) pe sub-eșantion de 1.000 instanțe |
| `Comparatie_metode_scatter.pdf` | PDF | Grid scatter cu toate metodele alăturate, colorat pe pictor |
| `Silhouette_comparatie.pdf` | PDF | Bar chart silhouette per metodă |

---

## Note tehnice

- **VGG16 fc2 + NMF**: stratul fc2 are activare ReLU → valori ≥ 0 → NMF funcționează direct, fără MinMaxScaler.
- **Eigenpicturi PCA**: în spațiu CNN nu se pot reconstrui imagini pixel-wise. Se afișează TOP 5 picturi reale cu scor maxim/minim pe fiecare componentă — interpretare semantică directă.
- **MDS pe centroizi**: MDS pe 8.446 instanțe ar necesita ~280 GB RAM (O(n²)). Se agregă la centroizii celor 50 pictori, deci MDS interpretează distanțe stilistice între pictori.
- **t-SNE pe PCA-50**: t-SNE pe 4.096 features brute e lent și zgomotos. Pipeline standard: PCA la 50 componente, apoi t-SNE.
- **KPCA pe sub-eșantion**: kernel matrix O(n²) — se folosesc max 2.000 instanțe stratificat per pictor.
