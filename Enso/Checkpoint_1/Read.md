# Checkpoint 1 — Basic GCN Baseline

## Overview

Checkpoint 1 is the **first working end-to-end pipeline** for
GNN-based ENSO forecasting. It serves as a baseline before
introducing the paper's core innovations. The goal was to verify
that the full pipeline — data loading → graph construction →
GNN training → evaluation — works correctly.

**Key result: Correlation Skill Score of 0.9762**
*(Note: This is not directly comparable to the paper — see limitations below)*

---

## What This Checkpoint Does

- Downloads raw GODAS ocean temperature NetCDF files from NOAA PSL
- Extracts SST anomaly and heat content anomaly features
- Builds a simple grid-based graph over ocean grid points
- Trains a 2-layer GCN to predict ENSO state
- Evaluates with Pearson correlation skill score

---

## Data

### Source
Raw GODAS `pottmp` (potential temperature) NetCDF files downloaded
directly from NOAA PSL:
> https://downloads.psl.noaa.gov/Datasets/godas/

### Files downloaded
`pottmp.1981.nc` through `pottmp.2000.nc` — 20 yearly files (~2GB total)

> Note: `pottmp.1980.nc` was found to be corrupted and excluded

### Feature extraction

| Feature | Extraction method | Physical meaning |
|---|---|---|
| SST anomaly | `pottmp` at depth index 0 (~5m) | Sea surface temperature |
| Heat content anomaly | Mean of `pottmp` over depth indices 0–9 (0–300m) | Subsurface ocean heat |

Both features had the **seasonal cycle removed** by subtracting the
monthly climatological mean.

### Data split

| Split | Period | Samples |
|---|---|---|
| Training | 1981–1996 | 192 months (80%) |
| Test | 1997–2000 | 48 months (20%) |

### Preprocessing
- Grid subsampled every 10th lat/lon point to fit GPU memory
- Features normalized to range `[-1, 1]`
- NaN values (land/missing ocean) replaced with 0

---

## Graph Construction

Each spatial grid point = one graph node

- **Nodes:** ~1,500 (after subsampling)
- **Edges:** Fixed grid connectivity — each node connected to
  immediate horizontal and vertical neighbours only
- **Node features:** `[SST anomaly, heat content anomaly]` — 2 features per node

---

## Model Architecture
```python
GraphinoLite(
  conv1: GCNConv(2 → 32) + ReLU + Dropout(0.2)
  conv2: GCNConv(32 → 32) + ReLU
  pool:  GlobalMeanPool
  fc:    Linear(32 → 1)
)
```

| Parameter | Value |
|---|---|
| Total parameters | ~13,000 |
| Input features per node | 2 |
| Hidden units | 32 |
| GCN layers | 2 |
| Pooling | Global mean pool |

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=0.0005) |
| Loss | MSE |
| Epochs | 50 |
| Batch size | 4 |
| Gradient clipping | max norm = 1.0 |
| Target variable | Proxy ONI (spatial mean SST anomaly) |
| Platform | Kaggle T4 GPU |

---

## Results

| Metric | Value |
|---|---|
| **Correlation Skill Score** | **0.9762** |
| Test MSE | 0.0280 |
| Final Train Loss | 0.0047 |
| Final Test Loss | 0.0280 |
| Lead time | 0 months (current state) |

---

## Why the High Correlation is Misleading

The 0.9762 correlation **is not directly comparable** to the
Graphino paper's 0.80 for three important reasons:

1. **Lead time:** We predict the *current* ocean state (0-month lead).
   The paper forecasts 3–23 months into the future — a much harder task
2. **Test data:** Our test set (GODAS 1997–2000) comes from the same
   dataset as training (GODAS 1981–1996). The paper tests on completely
   unseen data from a different source
3. **Target variable:** We use a proxy spatial mean SST rather than
   the real Niño 3.4 ONI index

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Fixed grid edges | Cannot learn ocean teleconnections |
| 1-month input | Misses slow ENSO buildup patterns |
| GODAS training only | 16× less data than the paper |
| Proxy ONI target | Less accurate than real ONI index |
| 0-month lead | Not a real forecast task |
| Subsampled grid | Lower spatial resolution |

All of these are addressed in **Checkpoint 2**.

---

## Files

| File | Description |
|---|---|
| `checkpoint1_notebook.ipynb` | Full Kaggle training notebook |
| `checkpoint1_loss.png` | Training and test loss curves |
| `checkpoint1_predictions.png` | Predicted vs actual ONI |

---

## References

- Cachay et al. (2021) — *The World as a Graph* —
  [arXiv:2104.05089](https://arxiv.org/abs/2104.05089)
- GODAS dataset —
  [NOAA PSL](https://psl.noaa.gov/data/gridded/data.godas.html)


