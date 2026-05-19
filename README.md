# Reducerea Dimensionalității pe Picturi — Best Artworks of All Time

Proiect pentru cursul **Tehnici de Învățare Automată și Aplicații**, tema **C. Reducerea dimensionalității**.

Aplică șapte metode obligatorii (PCA, Analiză Factorială, NMF, ICA, Kernel PCA, MDS, t-SNE) pe vectori semantici extrași cu VGG16 din ~8.446 picturi ale celor 50 cei mai influenți pictori (Kaggle: `ikarus777/best-artworks-of-all-time`).

---

## Cele două variante de pipeline

```
Varianta A — ImageNet (rapid, ~15 min)
  00_main_vectorizare.py  →  features_cnn.csv  →  01..08_main_*.py

Varianta B — Fine-tunat pe picturi (recomandat, ~1-3h GPU)
  00_main_vectorizare.py  →  00b_main_finetune.py  →  features_cnn.csv  →  01..08_main_*.py
```

**De ce contează?** VGG16 antrenat pe ImageNet știe să distingă pisici de mașini, nu stilul lui Van Gogh de cel al lui Rembrandt. Fine-tuning-ul pe cei 50 de artiști produce vectori fc2 discriminativi pentru sarcina reală — clusterele din PCA/t-SNE devin separabile.

---

## Structura fișierelor

```
artwork-dimensionality-reduction/
│
├── kaggle.json.template            ← copiază în ~/.kaggle/kaggle.json și completează
│
├── data_in/
│   └── best_artworks/
│       ├── images/                 ← descărcat de Kaggle (organizat pe pictor)
│       ├── artists.csv             ← metadata 50 pictori (gen, naționalitate, ani)
│       ├── features_cnn.csv        ← GENERAT de pas 0 (sau 0b)
│       ├── features_cnn.csv.imagenet_backup  ← backup ImageNet (creat de 00b)
│       └── vgg16_finetuned.keras   ← model fine-tunat (creat de 00b)
│
├── data_out/                       ← toate PDF-urile și CSV-urile
│
├── functii.py                      ← calcule: VGG16 extractor, Bartlett, KMO, stres MDS...
├── grafice.py                      ← vizualizare: scatter cu elipse, elbow, corelograma...
├── reducere_dim.py                 ← logica celor 7 metode (aplica_pca, aplica_fa, ...)
│
├── 00_main_vectorizare.py          ← Pas 0: descarcă date + extrage features ImageNet VGG16
├── 00b_main_finetune.py            ← Pas 0b (opțional): fine-tunează VGG16 pe artiști
├── 01_main_pca.py                  ← Pas 1: PCA
├── 02_main_fa.py                   ← Pas 2: Analiză Factorială
├── 03_main_nmf.py                  ← Pas 3: NMF
├── 04_main_ica.py                  ← Pas 4: ICA
├── 05_main_kpca.py                 ← Pas 5: Kernel PCA
├── 06_main_mds.py                  ← Pas 6: MDS
├── 07_main_tsne.py                 ← Pas 7: t-SNE
└── 08_main_comparatie.py           ← Pas 8: comparație cross-metode
```

---

## Setup

### 1. Instalare dependențe

```bash
pip install -r requirements.txt
```

### 2. Configurare Kaggle API

```bash
cp kaggle.json.template ~/.kaggle/kaggle.json
# Editează fișierul — înlocuiește cele două câmpuri cu datele tale Kaggle
# (kaggle.com → Account → Settings → API → Create New Token)
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Eliberare memorie înainte de rulare (recomandat în Codespaces)

```bash
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
pkill -f tensorboard 2>/dev/null; true
free -h   # verificare RAM disponibil
```

---

## Pas 0 — Descărcare date și extragere features ImageNet

```bash
python 00_main_vectorizare.py
```

La prima rulare: descarcă automat dataset-ul de pe Kaggle (~2.7 GB), normalizează numele folderelor (fix Unicode NFD→NFC), verifică că toți cei 50 de artiști sunt prezenți pe disc, apoi extrage vectori VGG16 fc2 (4096-dim) cu afișaj live multi-panou.

Salvează: `data_in/best_artworks/features_cnn.csv` — ~8.446 rânduri × 4.101 coloane.

La rulări ulterioare: sare peste extragere dacă `features_cnn.csv` există deja.

**Durată estimată:** ~15 min GPU / ~1h CPU.

---

## Pas 0b — Fine-tuning VGG16 pe setul de picturi *(recomandat)*

```bash
python 00b_main_finetune.py
```

Antrenează VGG16 pe clasificarea celor 50 de artiști, în două faze:

| Faza | Ce se antrenează | LR | Epoci max |
|------|------------------|----|-----------|
| 1 — cap clasificare | Strat nou `Dense(50)` — backbone înghețat | 1e-3 | 12 |
| 2 — fine-tuning | block4 + block5 + fc1 + fc2 + cap | 1e-5 | 30 |

**Augmentare date** (faza de antrenare): flip orizontal aleator, variații de luminozitate, saturație, contrast — reduce overfitting-ul pe ~169 imagini/artist.

**Early stopping** cu `patience=5` în ambele faze; cel mai bun model (pe `val_accuracy`) este restaurat automat.

La final:
- `data_in/best_artworks/vgg16_finetuned.keras` — model salvat
- `features_cnn.csv` — **suprascris** cu vectori fc2 din modelul fine-tunat
- `features_cnn.csv.imagenet_backup` — backup al versiunii ImageNet (pentru revenire)
- `data_out/Training_finetune.pdf` — curbe accuracy + loss (ambele faze)

**Revenire la features ImageNet:**
```bash
cp data_in/best_artworks/features_cnn.csv.imagenet_backup data_in/best_artworks/features_cnn.csv
```

**Durată estimată:** ~1-3h GPU / ~8-24h CPU.

---

## Pașii 1–8 — Metode de reducere a dimensionalității

Fiecare script se rulează independent după ce `features_cnn.csv` există. Outputurile se acumulează în `data_out/` — rularea unui main nu șterge outputurile celorlalți.

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

### Fine-tuning (`00b_main_finetune.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Training_finetune.pdf` | PDF | Curbe acuratețe + loss (faza 1 cap / faza 2 fine-tune) |

### PCA (`01_main_pca.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Varianta_PCA.csv` | CSV | Varianță explicată per componentă + procent cumulativ |
| `Scoruri_PCA.csv` | CSV | Scoruri pe 150 componente + metadata (pictor, stil, epocă, gen) |
| `r_xc_PCA.csv` | CSV | Corelații features-componente (matrice loadings ponderată) |
| `Selectie_PCA.csv` | CSV | Nr. componente recomandate per criteriu (Kaiser, 80%, Elbow) |
| `Varianta_PCA.pdf` | PDF | Varianță cumulativă + prag 80% + markere Kaiser/Elbow |
| `Elbow_PCA.pdf` | PDF | Scree plot PCA cu marcaj Elbow |
| `Corelograma_PCA.pdf` | PDF | Heatmap corelații features × componente (top 50 features) |
| `Cercul_PCA.pdf` | PDF | Cercul corelațiilor (biplot) |
| `Eigenpicturi_PCA.pdf` | PDF | Top 5 picturi cu scor max/min pe primele 6 componente |
| `Scatter_PCA_artist.pdf` | PDF | Scatter 2D comp1/comp2 cu elipse de confidență + etichete centroid |
| `Scatter_PCA_stil.pdf` | PDF | Scatter 2D colorat pe stil artistic |
| `Scatter_PCA_epoca.pdf` | PDF | Scatter 2D colorat pe epocă |
| `Scatter_PCA_gen.pdf` | PDF | Scatter 2D colorat pe naționalitate |

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
| `Scatter_FA_*.pdf` | PDF | Scatter 2D F1/F2 × 4 coloraje |

### NMF (`03_main_nmf.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Erori_NMF.csv` | CSV | Eroare Frobenius per q ∈ {5, 10, 15, 20, 30, 50} |
| `W_NMF.csv` | CSV | Matricea W (scoruri instanțe) pentru q optim + metadata |
| `H_NMF.csv` | CSV | Matricea H (componente-features) pentru q optim |
| `Elbow_NMF.pdf` | PDF | Curba erorii Frobenius + punct Elbow |
| `Componente_NMF.pdf` | PDF | Top 5 picturi cu activare maximă pe fiecare componentă NMF |
| `Scatter_NMF_*.pdf` | PDF | Scatter 2D C1/C2 × 4 coloraje |

### ICA (`04_main_ica.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Scoruri_ICA.csv` | CSV | Componentele independente ale celor 8.446 picturi + metadata |
| `Entropie_Kurtosis_ICA.csv` | CSV | Entropie diferențială + kurtosis per componentă independentă |
| `Kurtosis_ICA.pdf` | PDF | Bar chart kurtosis — cu cât mai mare, cu atât mai non-gaussian |
| `Entropie_ICA.pdf` | PDF | Bar chart entropie diferențială per IC |
| `Top_picturi_IC*.pdf` | PDF | Top 5 picturi cu activare absolută maximă pe primele 4 IC |
| `Scatter_ICA_*.pdf` | PDF | Scatter 2D IC1/IC2 × 4 coloraje |

### Kernel PCA (`05_main_kpca.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Scoruri_KPCA.csv` | CSV | Scoruri Kernel PCA (RBF) pe sub-eșantion + metadata |
| `Scoruri_PCA_referinta_KPCA.csv` | CSV | Scoruri PCA liniar pe același sub-eșantion (referință) |
| `Comparatie_KPCA_PCA_*.pdf` | PDF | Side-by-side PCA liniar vs Kernel PCA × 4 coloraje |

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
| `Scoruri_tSNE_perp*.csv` | CSV | Embedding 2D × 4 valori de perplexity (5, 30, 50, 100) + metadata |
| `KL_tSNE.csv` | CSV | Divergența KL finală per valoare de perplexity |
| `Grid_tSNE_*.pdf` | PDF | Grid 2×2 scatter (4 perplexity) × 4 coloraje |

### Comparație cross-metode (`08_main_comparatie.py`)

| Fișier | Tip | Conținut |
|---|---|---|
| `Silhouette_comparatie.csv` | CSV | Silhouette score per metodă pe label 'artist' (separabilitate 2D) |
| `Timpi_executie.csv` | CSV | Timpi de execuție (secunde) pe sub-eșantion de 1.000 instanțe |
| `Comparatie_metode_scatter.pdf` | PDF | Grid scatter cu toate metodele alăturate, colorat pe pictor |
| `Silhouette_comparatie.pdf` | PDF | Bar chart silhouette per metodă |

---

## Note tehnice

### Extragere features
- **Extragerea rulează o singură dată** în batch-uri (nu epochs) — este inference pur, fără antrenare. Fiecare imagine trece prin rețea exact o dată pentru a obține vectorul fc2.
- **Afișaj live** în timpul extragerii: dashboard `rich` cu bara de progress, strat curent din forward pass, throughput și statistici. Feature maps PNG salvate la fiecare 5 batch-uri în `data_out/VGG16_feature_maps_live.png`.
- **Checkpoint** la fiecare 50 de batch-uri: `data_out/features_partial.npy` + `data_out/paths_partial.txt` — în caz de crash, extragerea poate fi reluată manual.
- **Scriere atomică**: `features_cnn.csv` este scris mai întâi ca `.csv.tmp`, apoi redenumit — un crash mid-write nu poate corupe fișierul existent.

### Fine-tuning
- **De ce nu epochs la extragere, dar epochs la fine-tuning?** Extragerea e inference (greutăți fixe → output determinist → o singură trecere ajunge). Fine-tuning-ul actualizează greutățile prin backpropagare → necesită mai multe treceri (epochs) până la convergență.
- **Augmentare**: flip orizontal, variații de luminozitate/saturație/contrast aplicate înainte de `preprocess_input` VGG16. Reduc overfitting-ul pe ~169 imagini/artist.
- **Two-phase**: faza 1 la LR mare (1e-3) adaptează capul nou fără a strica features pre-antrenate; faza 2 la LR mic (1e-5) ajustează fin straturile superioare fără a distruge features ImageNet utile.
- **fc2 rămâne 4096-dim** — dimensiunea features și formatul `features_cnn.csv` sunt identice între varianta ImageNet și cea fine-tunată. Pașii 1–8 funcționează cu ambele fără modificări.

### Scatter plots (PCA, FA, NMF, ICA, t-SNE)
- **Elipse de confidență** per categorie (bazate pe eigendecompoziția matricei de covarianță a clusterului, n_std≈1.7).
- **Etichete la centroid**: pentru artist (50 categorii) se afișează prenumele/cognomenul pe fond alb; pentru stil/epocă/gen se afișează eticheta completă.
- **% varianță pe axe**: `Comp1 (5.2% varianță)`, `Comp2 (3.1% varianță)` — explică de ce PCA 2D nu produce separare perfectă.

### PCA pe features CNN
- **80% varianță necesită multe componente** (>100) — specific features CNN distribuite uniform pe multe direcții, nu ca datele tabulare clasice. Nu indică o problemă; este o proprietate a spațiului de embeddings.
- Graficul `Varianta_PCA.pdf` arată unde se atinge efectiv 80% (n_max=150 componente calculate) și marchează criteriile Kaiser și Elbow.

### Alte metode
- **VGG16 fc2 + NMF**: stratul fc2 are activare ReLU → valori ≥ 0 → NMF funcționează direct, fără MinMaxScaler.
- **Eigenpicturi PCA**: în spațiu CNN nu se pot reconstrui imagini pixel-wise. Se afișează TOP 5 picturi reale cu scor maxim/minim pe fiecare componentă — interpretare semantică directă.
- **MDS pe centroizi**: MDS pe 8.446 instanțe ar necesita ~280 GB RAM (O(n²) matrice distanțe). Se agregă la centroizii celor 50 pictori → MDS interpretează distanțe stilistice între pictori.
- **t-SNE pe PCA-50**: t-SNE pe 4.096 features brute e lent și zgomotos. Pipeline standard: PCA la 50 componente, apoi t-SNE.
- **KPCA pe sub-eșantion**: kernel matrix O(n²) — se folosesc max 2.000 instanțe stratificat per pictor.

### Unicode și compatibilitate OS
Kaggle extrage uneori arhivele cu encoding NFD pe Linux (caracterele compuse precum `ü` sunt stocate ca `u` + combining diacritic în loc de un singur codepoint). La fiecare pornire, scriptul `00_main_vectorizare.py` normalizează automat toate folderele NFD → NFC și afișează un raport complet cu care artiști sunt găsiți pe disc.
