"""
Pas 0b: Fine-tunează DINOv2-base pe setul de picturi (clasificare artiști, 50 clase).
Suprascrie features_cnn.csv cu embeddings CLS (768-dim) din modelul fine-tunat.

Rulat DUPĂ 00_main_vectorizare.py (care descarcă datele).

Abordare two-phase cu best practices 2024 pentru ViT/DINOv2:
  Faza 1 — Linear probe: backbone înghețat, antrenăm NUMAI capul liniar (5 epoci)
  Faza 2 — Fine-tuning: dezghețăm ultimele 3 encoder blocks + head (AdamW, cosine+warmup)

Best practices aplicate (bazate pe DINOv2 fine-tuning literature 2024):
  - Linear probe mai întâi → inițializare solidă a capului
  - Dezgheță NUMAI ultimele 3 din cei 12 transformer blocks (DINOv2-base)
  - AdamW cu weight_decay=0.1 (critic pentru ViT — previne overfitting)
  - Cosine LR schedule cu warmup (3 epoci) — standard pentru ViT fine-tuning
  - LR diferit: head 1e-3, backbone 1e-5 (10x mai mic pt conv features preantrenate)
  - Class weights sqrt(max/count) — robust pentru dezechilibru 50 clase
  - Label smoothing 0.1 — reduce overconfidence pe dataset mic
  - Augmentare specifică arte: ColorJitter puternic + RandomGrayscale + augmentare geometrică
  - Mixed precision (fp16) pe GPU — 2× viteză, același rezultat
  - Gradient clipping (max_norm=1.0) — stabilitate antrenare ViT

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
MODEL_PATH   = DATA_IN / "dinov2_finetuned.pt"

BATCH_TRAIN   = 32   # DINOv2-base e mai mic decât VGG16 → putem crește batch
BATCH_EXTRACT = 32
VAL_SPLIT     = 0.15
EPOCHS_PROBE  = 5    # faza 1: linear probe
EPOCHS_FT     = 20   # faza 2: fine-tuning (cu early stopping)
WARMUP_EPOCHS = 3    # warmup pentru cosine schedule
LABEL_SMOOTH  = 0.1
WEIGHT_DECAY  = 0.10  # ViT: weight decay mai mare (0.1 vs 1e-4 pentru CNN)
LR_HEAD       = 1e-3  # learning rate pentru clasificator
LR_BACKBONE   = 1e-5  # learning rate pentru ultimele 3 blocks (10× mai mic)
N_BLOCKS_FT   = 3     # număr de transformer blocks dezghețate (din 12 total)
IMG_SIZE      = 224

# Vizualizare feature maps în timpul antrenării (analog cu VGG16)
VIS_EVERY_N_BATCHES = 25                # cât de des să salvăm un PNG (la fiecare N batch-uri global)
VIS_BLOCKS          = (0, 2, 5, 8, 11)  # care din cele 12 transformer blocks să afișăm
VIS_TOP_CHANNELS    = 8                 # top canale (după varianță) per block
VIS_SUBDIR          = "finetune_progress"


# ─── augmentare (torchvision) ─────────────────────────────────────────────────

def _construieste_augmentare(train=True):
    """
    Augmentare orientată spre artă: ColorJitter mai puternic decât standard,
    RandomGrayscale (stiluri alb-negru), augmentare geometrică moderată.
    """
    import torchvision.transforms as T

    if train:
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ─── dataset PyTorch ─────────────────────────────────────────────────────────

def _construieste_dataset(images_dir, val_split, batch_size):
    """Construiește DataLoader-uri PyTorch din structura director artist/imagine."""
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
    from PIL import Image

    extensii = {".jpg", ".jpeg", ".png", ".bmp"}
    class_names = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    class ArtworkDataset(Dataset):
        def __init__(self, transform=None):
            self.samples = []
            for cls_name in class_names:
                cls_dir = images_dir / cls_name
                label = class_to_idx[cls_name]
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in extensii:
                        self.samples.append((str(p), label))
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            if self.transform:
                img = self.transform(img)
            return img, label

    full_ds_train = ArtworkDataset(transform=_construieste_augmentare(train=True))
    full_ds_val   = ArtworkDataset(transform=_construieste_augmentare(train=False))

    n = len(full_ds_train)
    rng = np.random.RandomState(42)
    idx_all = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx   = idx_all[:n_val].tolist()
    train_idx = idx_all[n_val:].tolist()

    train_ds = Subset(full_ds_train, train_idx)
    val_ds   = Subset(full_ds_val,   val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names


def _calcul_class_weights(images_dir, class_names, device):
    import torch
    extensii = {".jpg", ".jpeg", ".png", ".bmp"}
    counts = np.array([
        sum(1 for f in (images_dir / cls).iterdir() if f.suffix.lower() in extensii)
        for cls in class_names
    ], dtype=float)
    max_count = counts.max()
    weights = np.sqrt(max_count / np.clip(counts, 1, None))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─── model DINOv2 ────────────────────────────────────────────────────────────

class DINOv2Classifier(object):
    """Wrapper ușor pentru DINOv2 + linear head."""

    def __init__(self, n_classes, device):
        import torch
        import torch.nn as nn
        from transformers import Dinov2Model

        self.device = device
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
        hidden_size = self.backbone.config.hidden_size  # 768 pentru dinov2-base

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, n_classes),
        ).to(device)

        # Înghețăm tot backbone-ul implicit
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        import torch
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.head(cls_token)

    def dezgheata_faza2(self):
        """Dezgheță ultimele N_BLOCKS_FT transformer blocks."""
        encoder_layers = self.backbone.encoder.layer
        n_total = len(encoder_layers)
        for i, layer in enumerate(encoder_layers):
            if i >= n_total - N_BLOCKS_FT:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"[info] dezghețate ultimele {N_BLOCKS_FT}/{n_total} encoder blocks")

    def parametri_faza1(self):
        return list(self.head.parameters())

    def parametri_faza2(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": list(self.head.parameters()), "lr": LR_HEAD},
        ]

    def state_dict_complet(self):
        return {
            "backbone": self.backbone.state_dict(),
            "head": self.head.state_dict(),
        }


# ─── LR schedule cu warmup ───────────────────────────────────────────────────

def _cosine_cu_warmup(optimizer, warmup_steps, total_steps):
    """Cosine decay cu warmup liniar — standard pentru ViT fine-tuning."""
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─── vizualizare feature maps live (analog cu VGG16) ─────────────────────────

def _salveaza_feature_maps_dinov2(model, img_tensor, label, class_names,
                                   batch_idx, elapsed_s, out_dir):
    """
    Salvează un PNG cu activările patch-urilor după 5 transformer blocks,
    plus imaginea de input. Echivalent VGG16: feature maps per layer.

    DINOv2 (ViT-B/14) procesează imaginea ca 16×16=256 patch-uri + CLS token.
    Pentru fiecare block selectat, extragem hidden_state-ul patch-urilor,
    îl reshape-uim în (16, 16, 768) și afișăm primele 8 canale cu varianță maximă.
    """
    import torch
    import torchvision.transforms.functional as TF

    backbone_was_training = model.backbone.training
    model.backbone.eval()

    n_patches_side = IMG_SIZE // 14  # 16 pentru 224×224

    img_device = img_tensor.unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model.backbone(pixel_values=img_device, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # 13 tensori: embedding + 12 blocks

    # Un-normalize input pentru afișare (revers ImageNet stats)
    inv_mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(3, 1, 1)
    inv_std  = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(3, 1, 1)
    img_unnorm = img_tensor.cpu() * inv_std + inv_mean
    img_unnorm = torch.clamp(img_unnorm, 0, 1)
    img_display = TF.to_pil_image(img_unnorm)

    pictor_nume = class_names[label].replace("_", " ")

    n_rows = len(VIS_BLOCKS) + 1  # +1 pentru rândul cu input + info
    n_cols = VIS_TOP_CHANNELS
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))

    # Rândul 0: input în stânga, info în mijloc
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Input (224×224)", fontsize=8, loc="left")
    axes[0, 0].axis("off")
    info_col = n_cols // 2
    axes[0, info_col].text(
        0.5, 0.5,
        f"Batch #{batch_idx}\nPictor: {pictor_nume}\nDurata: {elapsed_s:.2f}s",
        ha="center", va="center", fontsize=10, transform=axes[0, info_col].transAxes,
    )
    for j in range(n_cols):
        if j != 0:
            axes[0, j].axis("off")

    # Rândurile 1..N: feature maps per block
    for row_idx, block_idx in enumerate(VIS_BLOCKS, start=1):
        # hidden_states[block_idx+1] = output DUPĂ block_idx (index 0 = embeddings inițiale)
        hs = hidden_states[block_idx + 1][0]          # (257, 768) = CLS + 256 patches
        patch_tokens = hs[1:, :]                       # (256, 768) — fără CLS
        patch_maps = patch_tokens.reshape(
            n_patches_side, n_patches_side, -1
        ).cpu().float().numpy()                        # (16, 16, 768)

        # Top 8 canale după varianță (cele mai informative pe această imagine)
        channel_var = patch_maps.var(axis=(0, 1))
        top_ch = np.argsort(-channel_var)[:VIS_TOP_CHANNELS]

        for col, ch_idx in enumerate(top_ch):
            heatmap = patch_maps[:, :, ch_idx]
            axes[row_idx, col].imshow(heatmap, cmap="viridis", interpolation="nearest")
            axes[row_idx, col].axis("off")
            if col == 0:
                axes[row_idx, col].set_title(
                    f"block{block_idx}  shape=({n_patches_side}, {n_patches_side}, 768)",
                    fontsize=8, loc="left",
                )

    fig.suptitle(
        "DINOv2 — feature maps live (top 8 canale per block, patch tokens după self-attention)",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = out_dir / f"batch_{batch_idx:06d}.png"
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)

    if backbone_was_training:
        model.backbone.train()


# ─── antrenare ───────────────────────────────────────────────────────────────

def _antreneaza_o_epoca(model, loader, optimizer, scheduler, loss_fn, scaler, device,
                        class_names=None, vis_dir=None, t_start=None, batch_counter=None):
    import torch
    model.backbone.train()
    model.head.train()
    total_loss, total_correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model.forward(imgs)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(model.backbone.parameters()) + list(model.head.parameters())
             if p.requires_grad],
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)

        # Vizualizare feature maps live (analog cu VGG16)
        if batch_counter is not None and vis_dir is not None:
            batch_counter[0] += 1
            if batch_counter[0] % VIS_EVERY_N_BATCHES == 0:
                elapsed = time.time() - t_start if t_start is not None else 0.0
                try:
                    _salveaza_feature_maps_dinov2(
                        model, imgs[0].detach(), int(labels[0].item()),
                        class_names, batch_counter[0], elapsed, vis_dir,
                    )
                except Exception as exc:
                    print(f"  [warn] vizualizare eșuată la batch {batch_counter[0]}: {exc}")

    return total_loss / n, total_correct / n


def _evalueaza(model, loader, loss_fn, device):
    import torch
    model.backbone.eval()
    model.head.eval()
    total_loss, total_correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.forward(imgs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)

    return total_loss / n, total_correct / n


def _antrenare_completa(model, train_loader, val_loader, class_weights, class_names, device):
    import torch
    import torch.nn as nn

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    steps_per_epoch = len(train_loader)

    # Director pentru PNG-uri cu feature maps live
    vis_dir = functii.DATA_OUT / VIS_SUBDIR
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] feature maps live → {vis_dir}/  (la fiecare {VIS_EVERY_N_BATCHES} batch-uri)")

    # Counter global de batch-uri (persistă între faze) și moment start
    batch_counter = [0]
    t_start_training = time.time()

    hist1 = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    hist2 = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # ── Faza 1: Linear probe ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Faza 1 / 2 — Linear probe (backbone înghețat, LR=1e-3)")
    print("═" * 60)

    opt1 = torch.optim.AdamW(model.parametri_faza1(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    t0 = time.time()
    best_val_acc = 0.0

    for ep in range(1, EPOCHS_PROBE + 1):
        tr_loss, tr_acc = _antreneaza_o_epoca(
            model, train_loader, opt1, None, loss_fn, scaler, device,
            class_names=class_names, vis_dir=vis_dir,
            t_start=t_start_training, batch_counter=batch_counter,
        )
        val_loss, val_acc = _evalueaza(model, val_loader, loss_fn, device)
        hist1["loss"].append(tr_loss)
        hist1["acc"].append(tr_acc)
        hist1["val_loss"].append(val_loss)
        hist1["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"  EP {ep:02d}/{EPOCHS_PROBE}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"  Faza 1 finalizată în {(time.time()-t0)/60:.1f} min  "
          f"| best val_acc = {best_val_acc:.4f}")

    # ── Faza 2: Fine-tuning ultimele 3 blocks ──────────────────────────
    print("\n" + "═" * 60)
    print(f"  Faza 2 / 2 — Fine-tuning ultimele {N_BLOCKS_FT} blocks "
          f"(AdamW + cosine+warmup)")
    print("═" * 60)

    model.dezgheata_faza2()
    total_steps = EPOCHS_FT * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    opt2 = torch.optim.AdamW(model.parametri_faza2(), weight_decay=WEIGHT_DECAY)
    scheduler = _cosine_cu_warmup(opt2, warmup_steps, total_steps)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    PATIENCE = 5
    t0 = time.time()

    for ep in range(1, EPOCHS_FT + 1):
        tr_loss, tr_acc = _antreneaza_o_epoca(
            model, train_loader, opt2, scheduler, loss_fn, scaler, device,
            class_names=class_names, vis_dir=vis_dir,
            t_start=t_start_training, batch_counter=batch_counter,
        )
        val_loss, val_acc = _evalueaza(model, val_loader, loss_fn, device)
        hist2["loss"].append(tr_loss)
        hist2["acc"].append(tr_acc)
        hist2["val_loss"].append(val_loss)
        hist2["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        mark = " ← best" if improved else ""
        print(f"  EP {ep:02d}/{EPOCHS_FT}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{mark}")

        if improved:
            best_val_acc = val_acc
            best_state = model.state_dict_complet()
            # Salvează checkpoint
            torch.save({
                "epoch": ep,
                "model_state": {**{"dinov2." + k: v for k, v in best_state["backbone"].items()},
                                **{"head." + k: v for k, v in best_state["head"].items()}},
                "val_acc": best_val_acc,
            }, str(MODEL_PATH))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [early stop] nicio îmbunătățire în {PATIENCE} epoci consecutive")
                break

    print(f"  Faza 2 finalizată în {(time.time()-t0)/60:.1f} min  "
          f"| best val_acc = {best_val_acc:.4f}")

    return hist1, hist2


# ─── grafic antrenare ─────────────────────────────────────────────────────────

def _salveaza_grafic_antrenare(hist1, hist2):
    ep1 = len(hist1["acc"])
    ep2 = len(hist2["acc"])
    x1 = np.arange(1, ep1 + 1)
    x2 = np.arange(ep1 + 1, ep1 + ep2 + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key_t, key_v, titlu, ylab in [
        (ax_acc,  "acc",  "val_acc",  "Acuratețe antrenare / validare",  "Acuratețe"),
        (ax_loss, "loss", "val_loss", "Loss antrenare / validare (CE+LS)", "Loss"),
    ]:
        ax.plot(x1, hist1[key_t], "b-o",  ms=4, label="train faza 1")
        ax.plot(x1, hist1[key_v], "b--s", ms=4, label="val faza 1")
        ax.plot(x2, hist2[key_t], "r-o",  ms=4, label="train faza 2")
        ax.plot(x2, hist2[key_v], "r--s", ms=4, label="val faza 2")
        ax.axvline(ep1 + 0.5, color="gray", ls=":", lw=1.5, label="start fine-tune")
        ax.set_xlabel("Epocă")
        ax.set_ylabel(ylab)
        ax.set_title(titlu)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"DINOv2-base fine-tuning — last {N_BLOCKS_FT} blocks, AdamW wd=0.1, "
        f"cosine+warmup, label_smooth={LABEL_SMOOTH}",
        fontsize=11,
    )
    plt.tight_layout()
    functii.DATA_OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(functii.DATA_OUT / "Training_finetune.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[OK] grafic antrenare salvat: data_out/Training_finetune.pdf")


# ─── extragere features cu modelul fine-tunat ────────────────────────────────

def _extrage_features(backbone, images_dir_paths, device):
    """Extrage CLS token din backbone fine-tunat pentru toate imaginile din df_paths."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    backbone.eval()

    paths = images_dir_paths
    n = len(paths)
    n_batches = (n + BATCH_EXTRACT - 1) // BATCH_EXTRACT
    features = []
    paths_ok = []
    sarite = 0

    print(f"\n[info] extragere CLS token DINOv2 fine-tunat ({n} imagini)...")
    t0 = time.time()

    for i, start in enumerate(range(0, n, BATCH_EXTRACT)):
        batch = paths[start : start + BATCH_EXTRACT]
        imgs, ok = [], []
        for p in batch:
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
            cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        features.append(cls_tokens)
        paths_ok.extend(ok)

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

    try:
        import torch
        from transformers import Dinov2Model
    except ImportError:
        sys.exit("[eroare] torch sau transformers nu sunt instalate. "
                 "Rulează: pip install torch torchvision transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] PyTorch {torch.__version__}, device: {device}")
    if device.type == "cuda":
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}, "
              f"memorie: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    # Dataset + class weights
    print("\n[info] pregătire dataset cu augmentare extinsă...")
    train_loader, val_loader, class_names = _construieste_dataset(IMAGES, VAL_SPLIT, BATCH_TRAIN)
    n_classes = len(class_names)
    class_weights = _calcul_class_weights(IMAGES, class_names, device)
    max_w = float(class_weights.max())
    min_w = float(class_weights.min())
    print(f"[info] {n_classes} clase, batch={BATCH_TRAIN}, val_split={VAL_SPLIT}")
    print(f"[info] class weights: min={min_w:.3f}, max={max_w:.3f} (sqrt inverse frequency)")

    # Model
    model = DINOv2Classifier(n_classes, device)
    n_params = sum(p.numel() for p in list(model.backbone.parameters()) + list(model.head.parameters()))
    print(f"[info] DINOv2-base: {n_params:,} parametri total")

    # Antrenare
    hist1, hist2 = _antrenare_completa(
        model, train_loader, val_loader, class_weights, class_names, device
    )
    _salveaza_grafic_antrenare(hist1, hist2)
    print(f"[OK] model salvat: {MODEL_PATH}")

    # Extragere features din backbone fine-tunat
    df_paths = functii.colecteaza_paths_si_metadata()
    if len(df_paths) == 0:
        sys.exit("[eroare] niciun fișier imagine găsit.")

    features, paths_ok = _extrage_features(model.backbone, df_paths["path"].tolist(), device)
    print(f"[OK] features shape = {features.shape}")
    print(f"[info] DINOv2 CLS: min={features.min():.3f}, max={features.max():.3f} "
          f"(NMF va aplica MinMaxScaler automat dacă există valori negative)")

    if FEATURES_CSV.exists():
        backup = FEATURES_CSV.with_suffix(".csv.frozen_backup")
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
    print("\nPoți acum rula 01_main_pca.py, 02_main_fa.py etc. cu features DINOv2 fine-tunate.")


if __name__ == "__main__":
    main()
