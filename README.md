# Atmospheric Data Analysis for Precipitation Forecasting
### IISC Bangalore Station | 2010 – 2025

A data analytics project built on 15 years of hourly atmospheric 
observations from the Indian Institute of Science (IISC) weather 
station in Bangalore, India. The project covers the full data 
pipeline — from raw multi-source ingestion and cleaning, through 
feature engineering, to exploratory data analysis — producing a 
structured, analysis-ready dataset for precipitation forecasting.

---

## Project Overview

This project integrates two independent atmospheric data sources 
into a single clean master dataset of 134,715 hourly observations 
spanning 2010 to 2025. The core analytical question is whether 
tropospheric moisture measurements, combined with standard surface 
meteorological variables, carry enough signal to forecast 
precipitation six hours ahead.

The project stops at the data analytics boundary — the final 
deliverable is a thoroughly analysed, well-documented, 
feature-engineered dataset ready for any downstream modelling task.

---

## Data Sources

**Source 1 — NASA POWER (MERRA-2 Reanalysis)**

Hourly surface meteorological data downloaded for IISC Bangalore 
coordinates (13.01N, 77.56E) for the period 2010 to 2025. 
Provides temperature, relative humidity, surface pressure, and 
precipitation at hourly resolution as a single CSV file.

**Source 2 — Nevada Geodetic Laboratory (NGL)**

Tropospheric delay estimates processed from raw satellite 
observation data at the IISC station. Delivered as one ZIP archive 
per year, each containing daily compressed files in SINEX/TRO 
format at 5-minute sampling intervals. Provides Zenith Total Delay 
(ZTD) and Precipitable Water Vapour (PWV).

---

## Dataset

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| Date range            | 2010-01-05 to 2025-12-31                   |
| Total rows            | 134,715 hourly observations                |
| Input features        | 13                                         |
| Target variable       | Precipitation (mm/hr)                      |
| Overall coverage      | 96.1% of the 15-year period                |
| Missing values        | 0 (after gap filling)                      |
| Zero-rain hours       | 65,207 (48.4%)                             |
| Rainy hours           | 69,508 (51.6%)                             |
| Max hourly precip     | 30.61 mm/hr                                |

---

## Features

**From NASA POWER**
- Temperature at 2 metres (C)
- Relative Humidity at 2 metres (%)
- Surface Pressure (kPa)
- Precipitation (mm/hr) — target variable

**From NGL Satellite Data**
- Zenith Total Delay — ZTD (mm)
- Precipitable Water Vapour — PWV (mm)

**Engineered Features**
- Cyclic hour encoding — hour_sin, hour_cos
- Cyclic month encoding — month_sin, month_cos
- Precipitation lag features — Lag 1hr, Lag 3hr, Lag 6hr, Lag 24hr

---

## Data Pipeline

The raw data from both sources arrives in incompatible formats and 
resolutions. The pipeline handles the following steps in order:

1. Parse NGL yearly ZIP archives with double decompression
2. Extract ZTD and PWV from SINEX format files
3. Convert proprietary epoch format to standard datetime
4. Apply physical sanity bounds to filter bad sensor readings
5. Resample 5-minute NGL data to hourly means
6. Load NASA POWER CSV, skip metadata header, fix units
7. Convert precipitation from mm/day to mm/hr
8. Merge both sources on Datetime using inner join
9. Reindex to continuous hourly timeline to expose hidden gaps
10. Fill short gaps by linear interpolation
11. Fill medium gaps by forward fill
12. Fill precipitation gaps with zero
13. Drop rows with gaps exceeding 24 hours
14. Engineer cyclic time features and lag features

---

## Exploratory Data Analysis

The EDA notebook covers the following:

- Target variable distribution — extreme skew toward light rain,
  48.4% zero values, intensity bucket breakdown
- Feature distributions — all six physical variables
- 15-year time series — daily aggregations with monsoon seasons
  highlighted
- Monthly climatology — seasonal patterns in temperature, humidity,
  PWV, and precipitation across all 12 months
- Diurnal cycle — hourly patterns showing afternoon convection peak
- ZTD and PWV analysis — physical relationship with precipitation,
  dry vs rainy hour distributions, boxplot comparison
- Correlation analysis — full Pearson matrix and feature-target
  correlation ranking
- Cyclic encoding rationale — visual demonstration of why raw hour
  and month numbers are incorrect for neural network input
- Lag feature analysis — scatter plots and correlation values
  showing temporal persistence of rainfall
- Data coverage — yearly coverage percentages, month-by-year
  heatmap showing all gaps across 15 years
- Monsoon vs non-monsoon comparison — distribution shifts in key
  variables between seasons
- PWV rise before rain — composited average of 300+ independent
  rain onset events showing moisture buildup 6-12 hours before rain

---

## Key Findings

- PWV is on average 12-15 mm higher in the hours immediately before
  rain compared to dry hours, confirming it as the strongest
  physical predictor in the dataset
- Precipitation lag features carry the highest correlation with the
  target variable, capturing the temporal persistence of weather
  systems
- Rainfall is strongly concentrated in June to September, with July
  and August accounting for the majority of annual totals
- The afternoon hours between 14:00 and 17:00 LST show the highest
  rain frequency, consistent with convective rainfall driven by
  daytime heating
- Surface pressure operates in a very narrow band due to Bangalore's
  fixed elevation of 841 metres, limiting its discriminative power
- The dataset has extreme class imbalance in rainfall intensity —
  the top 1% of rain events account for a disproportionate share
  of total precipitation volume

---

## Repository Structure
