"""
Chapter 4 case study - Step 2
Since well field labels are not physically consistent across campaigns,
comparison is done at the surface level: interpolate each campaign onto a
common grid (Inverse Distance Weighting), then difference the surfaces.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.path import Path
from scipy.spatial import ConvexHull

heads = pd.read_csv("/home/claude/hennaya/heads_clean.csv")

# Flag and drop the one physically implausible record identified in Step 1
heads = heads[heads["well_id"] != "HYN-2022-033"].copy()

def idw_grid(x, y, z, grid_x, grid_y, power=2, k=8):
    tree = cKDTree(np.column_stack([x, y]))
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    dist, idx = tree.query(pts, k=min(k, len(x)))
    dist = np.where(dist == 0, 1e-6, dist)
    w = 1.0 / dist**power
    z_interp = np.sum(w * z[idx], axis=1) / np.sum(w, axis=1)
    return z_interp.reshape(gx.shape), gx, gy

# common grid covering the union of all campaigns
xmin, xmax = heads["x"].min() - 200, heads["x"].max() + 200
ymin, ymax = heads["y"].min() - 200, heads["y"].max() + 200
res = 60  # metres
gx1d = np.arange(xmin, xmax, res)
gy1d = np.arange(ymin, ymax, res)

surfaces = {}
hulls = {}
for year in [1981, 2012, 2022]:
    sub = heads[heads["campaign_year"] == year]
    x, y, z = sub["x"].values, sub["y"].values, sub["head_m"].values
    Z, GX, GY = idw_grid(x, y, z, gx1d, gy1d)
    surfaces[year] = Z
    hull = ConvexHull(np.column_stack([x, y]))
    hulls[year] = Path(np.column_stack([x, y])[hull.vertices])

# mask: only keep grid cells inside the convex hull of ALL three campaigns
# (avoid extrapolating into unsampled areas)
gx, gy = np.meshgrid(gx1d, gy1d)
pts = np.column_stack([gx.ravel(), gy.ravel()])
mask = np.ones(len(pts), dtype=bool)
for year in [1981, 2012, 2022]:
    mask &= hulls[year].contains_points(pts)
mask = mask.reshape(gx.shape)

for year in surfaces:
    surfaces[year] = np.where(mask, surfaces[year], np.nan)

# ---- difference maps and annualised rates ----
diff_81_12 = surfaces[2012] - surfaces[1981]
rate_81_12 = diff_81_12 / (2012 - 1981)   # m/year

diff_12_22 = surfaces[2022] - surfaces[2012]
rate_12_22 = diff_12_22 / (2022 - 2012)   # m/year

for name, arr in [("1981->2012 total change (m)", diff_81_12),
                   ("1981->2012 rate (m/yr)", rate_81_12),
                   ("2012->2022 total change (m)", diff_12_22),
                   ("2012->2022 rate (m/yr)", rate_12_22)]:
    valid = arr[~np.isnan(arr)]
    print(f"{name}: mean={np.nanmean(arr):.3f}  min={np.nanmin(arr):.3f}  "
          f"max={np.nanmax(arr):.3f}  n_cells={valid.size}")

pct_declining_1 = 100 * np.sum(rate_81_12 < 0) / np.sum(~np.isnan(rate_81_12))
pct_declining_2 = 100 * np.sum(rate_12_22 < 0) / np.sum(~np.isnan(rate_12_22))
print(f"\n% of overlap area with declining head, 1981-2012: {pct_declining_1:.1f}%")
print(f"% of overlap area with declining head, 2012-2022: {pct_declining_2:.1f}%")

# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, year in zip(axes[0], [1981, 2012, 2022]):
    sub = heads[heads["campaign_year"] == year]
    im = ax.pcolormesh(gx, gy, surfaces[year], cmap="viridis", shading="auto")
    ax.scatter(sub["x"], sub["y"], c="white", s=10, edgecolor="k", linewidth=0.3)
    ax.set_title(f"Piezometric head, {year} (n={len(sub)})")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Head (m)", shrink=0.8)

im1 = axes[1][0].pcolormesh(gx, gy, rate_81_12, cmap="RdBu", vmin=-2, vmax=2, shading="auto")
axes[1][0].set_title("Rate of change 1981-2012 (m/yr)")
plt.colorbar(im1, ax=axes[1][0], shrink=0.8)

im2 = axes[1][1].pcolormesh(gx, gy, rate_12_22, cmap="RdBu", vmin=-2, vmax=2, shading="auto")
axes[1][1].set_title("Rate of change 2012-2022 (m/yr)")
plt.colorbar(im2, ax=axes[1][1], shrink=0.8)

accel = rate_12_22 - rate_81_12
im3 = axes[1][2].pcolormesh(gx, gy, accel, cmap="RdBu", vmin=-2, vmax=2, shading="auto")
axes[1][2].set_title("Acceleration (rate2 - rate1, m/yr)")
plt.colorbar(im3, ax=axes[1][2], shrink=0.8)

for ax in axes[1]:
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/hennaya_head_trend_maps.png", dpi=150)
print("\nSaved figure: hennaya_head_trend_maps.png")
