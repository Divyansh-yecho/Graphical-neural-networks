# Checkpoint 2 — Proper Graphino Replication

## Overview

Checkpoint 2 is a proper replication of the **Graphino** paper
(*"The World as a Graph: Improving El Niño Forecasts with Graph Neural Networks"*,
Cachay et al. 2021). This checkpoint addresses all major limitations of
Checkpoint 1 by adopting the paper's actual datasets, input format, target
variable, and evaluation protocol.

**Key result: Correlation Skill Score of 0.7820 at 3-month lead time**
— matching the paper's reported ~0.80.

---

## What Changed from Checkpoint 1

| Aspect | Checkpoint 1 | Checkpoint 2 |
|---|---|---|
| Training data | GODAS 1981–1996 (189 months) | CMIP5 + SODA (3,061 samples) |
| Input per sample | 1-month snapshot (2 features) | 36-month window (72 features) |
| Target variable | Proxy SST mean | Real ONI index |
| Test data | Last 20% of training set | GODAS 1980–2015 (completely unseen) |
| Graph edges | Fixed grid connections | Learnable node embeddings |
| Lead time | 0 months | 3 months |

---

## Datasets

All datasets from Ham et al. (2019) — downloaded from:
> https://github.com/jeonghwan723/DL_ENSO

| Dataset | Period | Samples | Role |
|---|---|---|---|
| CMIP5 | 1861–2001 | 2,961 | Training (climate simulations) |
| SODA | 1871–1970 | 100 | Training (reanalysis) |
| GODAS | 1980–2015 | 36 | Test only (real observations) |

### Input format
Each sample = **36 months × 2 variables × 24 lat × 72 lon**

| Variable | Description |
|---|---|
| `sst` / `sst1` | Sea surface temperature anomaly |
| `t300` | Mean temperature anomaly over top 300m |

After flattening: **1,728 graph nodes × 72 features each**

### Labels
Real ONI index (3-month moving average of Niño 3.4 SST anomalies)
at 12 possible lead times (1–12 months). We use **lead time = 3 months**.

---

## Model Architecture — LearnableGraphino
```python
LearnableGraphino(
  node_embed:  nn.Parameter(1728, 16)   # learnable embeddings
  conv1:       GCNConv(72 → 64)
  conv2:       GCNConv(64 → 64)
  conv3:       GCNConv(64 → 64)
  fc1:         Linear(64 → 32)
  fc2:         Linear(32 → 1)
)
Total parameters: 42,753
```

### How the graph is built
- Each of 1,728 ocean grid points = one graph node
- Each node has a learnable 16-dim embedding vector
- Edges = **top-8 most similar nodes** by cosine similarity
- Since embeddings update via backpropagation, the graph
  topology evolves during training to reflect learned teleconnections

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Loss | MSE |
| Epochs | 50 |
| Batch size | 32 (train), 8 (test) |
| Gradient clipping | max norm = 1.0 |
| Dropout | 0.2 after GCN layers 1 and 2 |
| Platform | Kaggle T4 GPU |

---

## Results

| Metric | Value | Paper target |
|---|---|---|
| **Correlation Skill Score** | **0.7820** | **~0.80** |
| Test MSE | 0.4713 | — |
| Train Loss | 0.1235 | — |
| Test Loss | 0.5001 | — |
| Training samples | 3,061 | ~3,000+ |
| Test samples | 36 (GODAS — unseen) | 432 |
| Lead time | 3 months | 3 months |

---

## Analysis of Results

### Skill vs Lead Time
The model achieves **skillful forecasting (correlation > 0.50) up to
5 months ahead**:

| Lead time | Correlation |
|---|---|
| 1 month | 0.69 |
| 2 months | 0.73 |
| 3 months | **0.78** (peak) |
| 4 months | 0.76 |
| 5 months | 0.59 |
| 6 months | 0.44 ← drops below skill threshold |

The sharp drop at 6 months is consistent with the known
**Spring Predictability Barrier** in ENSO forecasting.

### Residuals
- No systematic bias — mean error close to zero
- Largest errors occur at strong La Niña events
  (samples ~16 and ~35 in the test set)
- Model slightly overestimates warming and underestimates cooling
  — consistent with CMIP5's known El Niño bias

### Feature Variance
- SST anomaly shows consistently higher variance than t300
- Both features show a ~12-month periodic variance pattern
  reflecting the seasonal cycle of ENSO variability
- Variance rises toward the most recent month in the 36-month
  window — the current ocean state is the most informative

### Learned Graph Structure
- The cosine similarity heatmap shows a sparse learned graph
- Equatorial Pacific nodes cluster together (high similarity)
- Some anti-correlated node pairs visible — physically meaningful
  teleconnection structure

---

## Known Limitations

1. **Domain gap** — CMIP5 simulations have different ENSO statistics
   than real observations, causing the large train/test loss gap
2. **No two-stage training** — paper pre-trains CMIP5 then fine-tunes
   SODA separately; we combined them
3. **Simplified edge learning** — node embedding similarity vs paper's
   dedicated structure learner network
4. **No ensemble** — paper averages 4 models; we use 1
5. **Small test set** — 36 vs paper's 432 samples
6. **Smaller model** — 64 hidden units vs paper's 250

---

## Files

| File | Description |
|---|---|
| `checkpoint2_notebook.ipynb` | Full Kaggle training notebook |
| `checkpoint2_loss.png` | Training and test loss curves |
| `checkpoint2_predictions.png` | Predicted vs actual ONI |
| `checkpoint2_residuals.png` | Prediction error analysis |
| `checkpoint2_skill_vs_lead.png` | Skill score vs lead time |
| `checkpoint2_similarity.png` | Learned node similarity heatmap |
| `checkpoint2_feature_variance.png` | Feature variance by month |

---

## References

- Cachay et al. (2021) — *The World as a Graph* — [arXiv:2104.05089](https://arxiv.org/abs/2104.05089)
- Ham et al. (2019) — *Deep learning for multi-year ENSO forecasts* — [Nature 573](https://doi.org/10.1038/s41586-019-1559-7)
- Data source: [github.com/jeonghwan723/DL_ENSO](https://github.com/jeonghwan723/DL_ENSO)






