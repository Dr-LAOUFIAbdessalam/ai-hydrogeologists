# Chapter 4 — Case Study: Groundwater Level Trends, Hennaya Plain

## Scope note
The book outline originally planned monthly time-series forecasting
(RF/LSTM) as in Sections 4.2-4.3. The Hennaya piezometric record is three
independent snapshots (1981, 2012, 2022), not a continuous series, and well
field labels are not physically consistent across campaigns (confirmed with
the data owner). This case study instead interpolates each campaign onto a
common grid (IDW) restricted to the overlap area sampled in all three
campaigns, then differences the surfaces to obtain spatially distributed
rates of change.

## Key results
- Mean rate is POSITIVE in both periods (1981-2012: +0.19 m/yr;
  2012-2022: +0.11 m/yr) — heads are rising on average.
- BUT the fraction of the area with local decline nearly doubles
  (15.8% -> 29.0%) between the two periods — increasing spatial
  heterogeneity, not uniform depletion or recovery.
- The 5-year precipitation anomaly before each census (Zenata station) is
  +17% (wetter) before 2012 and -16% (drier) before 2022, offering a
  plausible (not proven) explanation for the growing decline zones despite
  the still-positive mean rate — likely reflecting aquifer storage inertia.

## Limitations stated explicitly
- IDW on 30-95 wells per campaign; results near the coverage edges are
  sensitive to the sparse 1981 network (30 wells).
- The precipitation record begins in 1980/81, exactly at the first census,
  so no pre-survey climatic context exists for the 1981 baseline.
- One well (HYN-2022-033 / field label P139) was excluded as physically
  implausible (see Chapter 2 QC).

## Contents
- `data/raw/heads_clean.csv` — cleaned campaigns (Chapter 2 output)
- `scripts/01_trend_analysis.py` — standalone Python script
- `notebooks/ch04_trend_case_study.ipynb` — Colab notebook, reads directly
  from this repository (precipitation data reused from
  `ch02_data_preparation/data/raw/`)

## How to run
Open `notebooks/ch04_trend_case_study.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.
