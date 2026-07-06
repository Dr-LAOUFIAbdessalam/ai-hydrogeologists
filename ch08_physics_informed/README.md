# Chapter 8 — Physics-Informed and Surrogate Modelling

This chapter contains TWO worked examples, deliberately contrasting:

## Exercise 1: Synthetic Theis inversion (PINN) — an honest FAILURE
Explicitly synthetic, benchmarked against a known analytical solution
(Theis, 1935). Inverse recovery of Transmissivity (T) and Storativity (S)
from sparse synthetic drawdown observations did NOT succeed (T off by
~90%, S off by several hundred percent), despite a physics-loss weight
four orders of magnitude larger than a first naive attempt and separate,
much slower learning rates for (T, S) versus the network.

**Diagnostic:** initializing (T, S) exactly at their true values and
observing training showed the truth is not a stable point of this joint
training scheme — the optimizer drifts away even from a perfect start,
a genuine identifiability/optimization pathology, not a tuning mistake.
The resulting drawdown-field RMSE (~0.45 m against a mean drawdown of
~1.8 m) looks superficially reasonable despite the badly wrong physical
parameters. Reported as obtained, not tuned until a preferred number
appeared, per the book's commitment to honesty about method limitations
(Section 8.6).

## Exercise 2: REAL Hennaya MODFLOW surrogate — a feature-importance trap, resolved
Uses the actual calibrated MODFLOW model of the Hennaya plain published in
Laoufi, A.; Boudjema, A.; Guettaia, S.; Derdour, A.; Almaliki, A.H. (2024).
"Integrated Simulation of Groundwater Flow and Nitrate Transport in an
Alluvial Aquifer Using MODFLOW and MT3D." *Sustainability*, 16, 10777.
https://doi.org/10.3390/su162310777

**Design mirrors the published paper's own validation exactly:** trained
on 1981 (steady-state calibration target) + 2022 (transient calibration
target), validated on 2012 — held out from training entirely, exactly as
in the paper's own independent validation (their reported R2 = 0.978).

**Step 1 result:** Linear Regression reaches R2 = 0.973 and Random Forest
R2 = 0.966 on the held-out 2012 campaign — remarkably close to the
published model's own validation performance.

**Step 2, a trap:** Random Forest feature importance shows the calibrated
K/S/porosity fields contribute almost nothing (combined importance =
0.002) while elevation-related features dominate (combined importance =
0.839) — suggesting the surrogate found a statistical shortcut through
land-surface elevation rather than learning the K/S-dependent physics.

**Step 3, resolution via ablation:** refitting with K/S/porosity ALONE
(no position or elevation at all) still achieves **R2 = 0.944 (Random
Forest)**, though only **R2 = 0.525 (Linear Regression)** — confirming K/S
DO carry substantial real, non-linear signal. The near-zero importance in
the full model is a **multicollinearity artifact**: elevation and the
calibrated K/S zones are correlated (r = -0.42 between surface elevation
and log-conductivity), plausibly because both follow the same underlying
geological zonation of the alluvial plain. Impurity-based tree importance
can arbitrarily assign credit to one of two correlated features, starving
the other — motivating the move to SHAP in Chapter 9 rather than relying
on raw `feature_importances_` alone.

Together, these two exercises illustrate complementary lessons from
Sections 8.1 and 8.6: physics-informed and surrogate models are not
automatically immune to producing misleading results, whether through
open failure (Exercise 1) or through a good score that requires careful
diagnosis to interpret correctly (Exercise 2).

## Contents
- `scripts/01_pinn_theis_inversion.py` — Exercise 1 (synthetic Theis PINN)
- `notebooks/ch08_pinn_theis.ipynb` — Exercise 1 Colab notebook (self-contained)
- `data/raw/geomertry.TXT`, `Conductivity.TXT`, `Storage.TXT` — calibrated
  MODFLOW parameter grids (Layer 1 = alluvial aquifer, Layer 2 =
  impermeable bedrock base), 800 grid cells each
- `data/raw/Simulated_head_1981/2012/2022.TXT` — MODFLOW head outputs
  (HNOFLO = 1e30 flags inactive/dry cells, filtered out; 418 active cells)
- `scripts/02_hennaya_surrogate.py` — Exercise 2 script
- `notebooks/ch08b_hennaya_surrogate.ipynb` — Exercise 2 Colab notebook

## How to run
Open either notebook via Google Colab → File → Open notebook → GitHub,
then Run all. Exercise 1 trains from scratch (a few minutes); Exercise 2
is fast (classical ML on 836 training rows, seconds).
