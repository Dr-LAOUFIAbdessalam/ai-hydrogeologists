# Chapter 5 — Hydrochemistry and Nitrate Risk, Hennaya Plain

## Scope note
The book outline originally targeted a supervised classifier (XGBoost,
AUC ~0.84) as in Section 5.5. With only 19 sampling points, this is not
statistically defensible. This exercise instead uses methods legitimate at
this sample size: k-means facies clustering, PCA, paired Wilcoxon
signed-rank seasonal comparison, and descriptive spatial/statistical
summary — explicitly exploratory, not predictive.

## Key results
- **All 19 points exceed the WHO 50 mg/L nitrate guideline in both
  seasons** (dry mean 121 mg/L, wet mean 145 mg/L). Stated as a strong
  local signal, not a precise regional statistic (n=19, one year).
- Nitrate is the ONLY major parameter higher in the wet season than the
  dry season (all others show the expected dry-season evapo-concentration
  effect) — consistent with a "first flush" leaching mechanism from
  agricultural soils, matching the cropland-dominated land cover found in
  Chapter 2.
- Three hydrochemical facies identified by k-means; one small cluster
  (n=7 records, points p11/p16/p17/Ain Ouahab) shows markedly elevated
  Na-Cl alongside the highest nitrate.
- Well `p11` shows the highest chloride AND nitrate simultaneously in both
  seasons — a pattern more typical of localized wastewater infiltration
  than diffuse agricultural leaching; flagged for field verification, not
  claimed as proven.

## Data note
The source spreadsheet header states "meq/L" but a mass-balance check
(sum of major ions vs TDS) confirms the values are actually mg/L — a
labelling error in the original file, corrected here.

## Compositional data analysis (CLR) addendum
Major-ion concentrations are compositional data (implicitly constrained by
charge balance), so Euclidean-distance methods on raw/standardized values
risk spurious correlations from the "closure effect". Comparing the naive
StandardScaler approach above against a Centered Log-Ratio (CLR) transform:
- **Adjusted Rand Index between raw and CLR cluster assignments = 0.13**
  — the two approaches produce substantially different groupings, not
  minor variants of the same one.
- CLR explains more PC1 variance (~50% vs ~31%) with a cleaner axis
  dominated by chloride vs magnesium, plausibly reflecting a real
  hydrochemical evolution pathway (ion exchange / differential evaporative
  concentration) rather than the blended Na-Cl-SO4 axis from the naive
  approach.
- **Conclusion: for compositional hydrochemical data, skipping the
  log-ratio transform is not a neutral simplification** — CLR (over ALR,
  which needs an arbitrary reference component, or ILR, whose balances are
  harder to interpret physically) is used as the standard here.

## Contents
- `data/raw/wet_et_dry_with_19_puits.xlsx` — original data
- `data/raw/hennaya_hydrochem_tidy.csv` — reshaped, cleaned, with facies labels
- `scripts/01_hydrochem_analysis.py` — standalone Python script
- `notebooks/ch05_hydrochem_analysis.ipynb` — Colab notebook, reads directly
  from this repository

## How to run
Open `notebooks/ch05_hydrochem_analysis.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.
