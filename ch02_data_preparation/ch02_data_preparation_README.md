# Chapter 2 — Data Preparation: Hennaya Plain Piezometric Dataset

## What this chapter does
Complete data preparation workflow (Section 2.5) from raw piezometric
campaigns (1981, 2012, 2022, Hennaya plain, NW Algeria) through quality
control, gap/conflict resolution, and enrichment with climatic, terrain,
and soil covariates — producing a single ML-ready table.

## Key data-quality findings (real, not illustrative)
- **Well field labels are not physically consistent across campaigns**
  (confirmed with the data owner): e.g. "P1" refers to a different
  physical well in 1981, 2012, and 2022. New unambiguous IDs
  (`HYN-<year>-<seq>`) were assigned; original labels retained for reference.
- **Duplicate/conflicting record resolved:** well P37 (1981) had two rows
  with different `Obs. Time` and `HEAD` values; the anomalous single-day
  entry was dropped in favour of the annual-average entry.
- **One physically implausible record flagged:** `HYN-2022-033` (field
  label P139) shows HEAD 58.6 m below its own screen elevation, an outlier
  inconsistent with neighbouring wells — excluded from spatial analyses in
  later chapters, retained here with a QC flag for transparency.
- **Coordinate reference system:** UTM Zone 30N (EPSG:32630), confirmed
  both by direct coordinate-range comparison across datasets and by
  converting a test point to WGS84 (~-1.37°E, 34.98°N — matches the
  Hennaya/Tlemcen area exactly).

## Covariates added, and how
- **Precipitation:** Zenata station, 45 hydrological years (1980/81–2024/25).
  The 1980/81 year is flagged incomplete (only 4 of 12 months on record)
  and excluded from means. Cumulative and anomaly covariates (1/3/5-year
  windows) computed relative to each census year.
- **Elevation:** SRTM 30 m via the free Open Topo Data API (fetched live
  on Colab; the local sandbox used for development has no internet access
  to this host, hence the Colab-dependent design).
- **Land cover:** ESA WorldCover 10 m via Microsoft Planetary Computer
  STAC (chosen over the full Google Earth Engine workflow for simplicity —
  no account setup needed). Result: 61% of wells sit on cropland,
  supporting the agricultural-nitrate narrative developed in Chapter 5.
- **Soil texture (clay/sand/silt):** SoilGrids v2.0 (250 m, ISRIC),
  substituted for the globally available lithological map (GLiM), whose
  only free resolution (0.5°, ~55 km) would have assigned every well the
  identical value across the ~5 km wide Hennaya plain — a resolution
  mismatch documented explicitly rather than silently worked around.
  11 of 15 missing soil-texture values occur over built-up land cover,
  consistent with SoilGrids' known lack of coverage for artificial surfaces.

## A cautionary note on naive comparison across campaigns
Raw per-campaign means (e.g. mean depth-to-water) drift across 1981→2022,
but this partly reflects each campaign sampling a different, only
partially overlapping set of locations — not necessarily a real temporal
trend. The spatially honest trend analysis is carried out separately in
Chapter 4, which interpolates each campaign onto a common grid before
differencing.

## Contents
- `data/raw/head_1981.txt`, `head_2012.txt`, `head_2022.txt` — original
  piezometric campaigns
- `data/raw/Precipitations_mensuelles.xlsx` — Zenata station monthly precipitation
- `data/raw/wells_dem_landcover.csv`, `wells_dem_landcover_soil.csv` —
  fetched covariates (elevation, land cover, soil texture)
- `notebooks/ch02_data_preparation.ipynb` — Colab notebook, reads all raw
  files directly from this repository

## How to run
Open `notebooks/ch02_data_preparation.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.
The DEM/land-cover/soil-fetching cells require live internet access
(available on Colab, not in the development sandbox) and take several
minutes due to free-API rate limits (SoilGrids: 5 requests/minute).
