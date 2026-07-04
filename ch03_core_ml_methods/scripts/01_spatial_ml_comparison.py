"""
Chapter 3 worked example (adapted): comparing a Random Forest against a
linear regression baseline for predicting depth to water table from
spatial and physiographic covariates, using SPATIAL cross-validation
(Section 3.5.2) instead of a random split (Section 3.5.1) to avoid the
optimistic bias caused by spatial autocorrelation among nearby wells.

NOTE ON SCOPE: the original book outline (Section 3.6) called for a
temporal comparison of RF/XGBoost/LSTM on a continuous monthly time
series. The Hennaya dataset consists of three independent snapshots
(1981, 2012, 2022), not a continuous series, so a temporal model
comparison is not meaningful here (see Chapter 4 discussion). This
worked example instead demonstrates the same core methodological
lesson - avoiding naive splits with spatially/temporally correlated
hydrogeological data - using a spatial prediction task for which the
data are actually suited.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("/home/claude/hennaya/hennaya_heads_ml_ready.csv")

# Drop the one physically implausible record flagged in Chapter 2
df = df[df["well_id"] != "HYN-2022-033"].copy()
# Drop rows with missing soil texture (built-up land, SoilGrids has no
# prediction there) - for THIS exercise only; not dropped elsewhere.
df = df.dropna(subset=["clay_pct", "sand_pct", "silt_pct"]).copy()

target = "depth_to_water_approx_m"
num_features = ["elevation_srtm_m", "clay_pct", "sand_pct", "silt_pct", "x", "y"]
cat_feature = "landcover_class"

X_num = df[num_features].values
X_cat = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(df[[cat_feature]])
X = np.hstack([X_num, X_cat])
y = df[target].values

print(f"n = {len(df)} wells, {X.shape[1]} features "
      f"({len(num_features)} numeric + {X_cat.shape[1]} one-hot land cover)")
print(f"Target ({target}): mean={y.mean():.1f} m, std={y.std():.1f} m, "
      f"range=[{y.min():.1f}, {y.max():.1f}] m")

# --- Spatial blocks for cross-validation (Section 3.5.2) -----------------
# k-means on well coordinates creates spatially compact folds so that
# training and test wells are never immediate neighbours.
N_BLOCKS = 6
blocks = KMeans(n_clusters=N_BLOCKS, random_state=42, n_init=10).fit_predict(df[["x", "y"]].values)
df["spatial_block"] = blocks
print("\nWells per spatial block:")
print(df["spatial_block"].value_counts().sort_index())

def spatial_cv_scores(model, X, y, blocks):
    rmses, r2s = [], []
    for b in np.unique(blocks):
        train_idx = blocks != b
        test_idx = blocks == b
        if test_idx.sum() < 3:
            continue  # skip tiny blocks, unstable metric
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        r2s.append(r2_score(y[test_idx], pred))
    return np.array(rmses), np.array(r2s)

def random_cv_scores(model, X, y, n_splits=6, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_splits)
    rmses, r2s = [], []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        r2s.append(r2_score(y[test_idx], pred))
    return np.array(rmses), np.array(r2s)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
}

print("\n" + "="*70)
print("RESULTS: naive random split vs spatially-honest split")
print("="*70)
for name, model in models.items():
    rmse_r, r2_r = random_cv_scores(model, X, y)
    rmse_s, r2_s = spatial_cv_scores(model, X, y, blocks)
    print(f"\n{name}:")
    print(f"  Random CV   : RMSE = {rmse_r.mean():.2f} +/- {rmse_r.std():.2f} m | "
          f"R2 = {r2_r.mean():.2f} +/- {r2_r.std():.2f}")
    print(f"  Spatial CV  : RMSE = {rmse_s.mean():.2f} +/- {rmse_s.std():.2f} m | "
          f"R2 = {r2_s.mean():.2f} +/- {r2_s.std():.2f}")
    gap = rmse_s.mean() - rmse_r.mean()
    print(f"  --> Spatial CV RMSE is {gap:+.2f} m relative to random CV "
          f"({'optimistic bias confirmed' if gap > 0 else 'no meaningful gap'})")
