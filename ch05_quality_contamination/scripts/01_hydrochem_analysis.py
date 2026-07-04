"""
Chapter 5 worked example: hydrochemical facies, seasonal comparison, and
exploratory nitrate risk mapping for the 19 Hennaya sampling points
(17 wells + 2 springs, dry and wet seasons, 2022).

METHODOLOGICAL NOTE: with only 19 points, a supervised classifier with a
credible train/test split (as in the original book outline's XGBoost /
AUC=0.84 example) is not statistically defensible. This exercise instead
uses methods that are legitimate at this sample size: unsupervised
clustering, PCA, paired non-parametric seasonal comparison, and descriptive
spatial mapping - all explicitly exploratory, not predictive.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

xl = pd.ExcelFile("wet_et_dry_with_19_puits.xlsx")
raw = xl.parse("Feuil1", header=None)

def extract_block(raw, start_row, n=19, season="dry"):
    cols = ["point", "x", "y", "T", "ph", "tds", "ec", "na", "ca", "mg",
            "cl", "so4", "hco3", "no3_n", "k"]
    block = raw.iloc[start_row:start_row+n, 0:15].copy()
    block.columns = cols
    block["season"] = season
    return block

dry = extract_block(raw, 3, 19, "dry")
wet = extract_block(raw, 26, 19, "wet")
df = pd.concat([dry, wet], ignore_index=True)

for c in ["x","y","T","ph","tds","ec","na","ca","mg","cl","so4","hco3","no3_n","k"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Convert NO3-N (nitrate as nitrogen) to NO3 (as used for the WHO 50 mg/L threshold)
df["no3"] = df["no3_n"] * 4.43

print(f"n = {len(df)} records ({df['point'].nunique()} points x 2 seasons)")
print(df.groupby("season")[["na","ca","mg","cl","so4","hco3","no3"]].describe().T)

# --- mass balance sanity check (confirms mg/L, not meq/L, as established) ---
df["ion_sum_mgL"] = df[["na","ca","mg","cl","so4","hco3"]].sum(axis=1)
ratio = (df["ion_sum_mgL"] / df["tds"]).mean()
print(f"\nSanity check: mean(sum of major ions / TDS) = {ratio:.2f} "
      f"(confirms concentrations are in mg/L, not meq/L)")

df.to_csv("hennaya_hydrochem_tidy.csv", index=False)

# =====================================================================
# 1. Hydrochemical facies via k-means clustering
# =====================================================================
major_ions = ["na", "ca", "mg", "cl", "so4", "hco3"]
X = StandardScaler().fit_transform(df[major_ions])

print("\n--- Cluster count selection (silhouette score) ---")
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"k={k}: silhouette = {sil:.3f}")

best_k = 3  # selected from silhouette scan above
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(df[major_ions].pipe(lambda d: StandardScaler().fit_transform(d)))
df["facies_cluster"] = kmeans.labels_

print(f"\nCluster means (k={best_k}), original units (mg/L):")
print(df.groupby("facies_cluster")[major_ions + ["no3"]].mean().round(1))
print("\nCluster sizes:")
print(df["facies_cluster"].value_counts().sort_index())

# =====================================================================
# 2. PCA on major-ion chemistry
# =====================================================================
pca = PCA(n_components=3)
pcs = pca.fit_transform(X)
print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
print("PC1 loadings:", dict(zip(major_ions, pca.components_[0].round(2))))
print("PC2 loadings:", dict(zip(major_ions, pca.components_[1].round(2))))

# =====================================================================
# 3. Seasonal comparison (paired, non-parametric - n=19 pairs)
# =====================================================================
print("\n--- Paired seasonal comparison (Wilcoxon signed-rank, n=19 pairs) ---")
dry_wide = df[df.season == "dry"].set_index("point")
wet_wide = df[df.season == "wet"].set_index("point")
common_points = dry_wide.index.intersection(wet_wide.index)

for param in ["na", "ca", "mg", "cl", "so4", "hco3", "no3", "ec"]:
    d = dry_wide.loc[common_points, param].values
    w = wet_wide.loc[common_points, param].values
    stat, p = stats.wilcoxon(d, w)
    direction = "dry > wet" if np.median(d - w) > 0 else "wet > dry"
    sig = "*" if p < 0.05 else ""
    print(f"  {param:6s}: median diff = {np.median(d-w):+7.1f}  p={p:.3f} {sig}  ({direction})")

# =====================================================================
# 4. Nitrate: descriptive spatial summary (no classifier - n too small)
# =====================================================================
print("\n--- Nitrate (NO3, mg/L) descriptive summary ---")
who_threshold = 50
for season in ["dry", "wet"]:
    sub = df[df.season == season]
    n_exceed = (sub["no3"] > who_threshold).sum()
    print(f"  {season}: mean={sub['no3'].mean():.1f}, max={sub['no3'].max():.1f}, "
          f"n exceeding WHO 50 mg/L = {n_exceed}/{len(sub)}")

print("\nSaved: hennaya_hydrochem_tidy.csv")
"""
Chapter 5 addendum: Centered Log-Ratio (CLR) transformation of the major-ion
composition, compared against the naive StandardScaler approach used
initially, to check whether accounting for the compositional (closed-sum)
nature of the data changes the facies clustering and PCA conclusions.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

df = pd.read_csv("hennaya_hydrochem_tidy.csv")
major_ions = ["na", "ca", "mg", "cl", "so4", "hco3"]

# =====================================================================
# CLR transform
# =====================================================================
def clr_transform(X):
    """X: array (n_samples, n_parts), strictly positive concentrations."""
    log_X = np.log(X)
    geometric_mean_log = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean_log

X_raw = df[major_ions].values
assert (X_raw > 0).all(), "CLR requires strictly positive values"

X_clr = clr_transform(X_raw)
clr_cols = [f"clr_{ion}" for ion in major_ions]
df[clr_cols] = X_clr

print("CLR-transformed values (first 3 rows):")
print(df[["point", "season"] + clr_cols].head(3).to_string())

# =====================================================================
# Re-run clustering on CLR-transformed data (already log-ratio scaled,
# no further standardisation needed since CLR already centers the data)
# =====================================================================
print("\n--- Silhouette scores: raw StandardScaler vs CLR ---")
X_scaled_raw = StandardScaler().fit_transform(X_raw)

for k in [2, 3, 4, 5]:
    labels_raw = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled_raw)
    labels_clr = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_clr)
    sil_raw = silhouette_score(X_scaled_raw, labels_raw)
    sil_clr = silhouette_score(X_clr, labels_clr)
    print(f"k={k}: raw silhouette = {sil_raw:.3f}  |  CLR silhouette = {sil_clr:.3f}")

# --- Compare cluster assignments at k=3 (used earlier) ---
labels_raw_3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled_raw)
labels_clr_3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_clr)
ari = adjusted_rand_score(labels_raw_3, labels_clr_3)
print(f"\nAgreement between raw-scaled and CLR cluster assignments (k=3): "
      f"Adjusted Rand Index = {ari:.3f}  (1.0 = identical, 0.0 = random)")

df["facies_cluster_clr"] = labels_clr_3
print("\nCLR-based cluster means (back-transformed to mg/L for interpretability):")
print(df.groupby("facies_cluster_clr")[major_ions + ["no3"]].mean().round(1))
print("\nCLR-based cluster sizes:")
print(df["facies_cluster_clr"].value_counts().sort_index())

# =====================================================================
# PCA comparison
# =====================================================================
pca_raw = PCA(n_components=3).fit(X_scaled_raw)
pca_clr = PCA(n_components=3).fit(X_clr)

print(f"\nExplained variance - raw:  {pca_raw.explained_variance_ratio_.round(3)}")
print(f"Explained variance - CLR:  {pca_clr.explained_variance_ratio_.round(3)}")

print("\nPC1 loadings - raw StandardScaler:")
print(dict(zip(major_ions, pca_raw.components_[0].round(2))))
print("PC1 loadings - CLR (log-ratio, interpret as relative dominance):")
print(dict(zip(major_ions, pca_clr.components_[0].round(2))))

df.to_csv("hennaya_hydrochem_tidy.csv", index=False)
print("\nUpdated hennaya_hydrochem_tidy.csv with CLR columns and CLR-based facies")
