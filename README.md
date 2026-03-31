# ANOMALY-DETECTION
COBBLESTONE WORK
Here’s a draft for your GitHub `README.md`:

# Efficient Data Stream Anomaly Detection

This repository contains the code and resources for my project titled Efficient Data Stream Anomaly Detection. The project aims to detect anomalies in a continuous data stream using various statistical and machine learning models. The approach involves handling seasonal variations, concept drift, and integrating multiple detection models in an ensemble or hybrid framework for improved accuracy.

# Project Overview

A complete project report has been uploaded as a PDF, which includes:
Introduction & Approach: Explanation of the problem, goals, and the detection methodology.
Methods: Detailed description of each detection model, including traditional statistical methods and more advanced machine learning techniques.
Results: Comparative analysis of the models with performance metrics such as Precision, Recall, F1 Score, and ROC AUC.
References: Research and materials consulted during the project.

# Code Structure

Static Models
The following models are implemented as static methods in `staticmethods.py`:
Z-Score
KNN (K-Nearest Neighbors)
IQR (Interquartile Range)
EMVG (Extended Moving Average with Seasonal Variation)
emvg.py: Implements the EMVG model, an extended version of the moving average, adapted to handle seasonal variations in the data.


# Visualization

visual.py: Handles the visualization of results and detection, including plotting data streams and marking anomalies.

# Hybrid Models

The hybrid model uses a combination of statistical and machine learning models to improve performance. The following files are included:
data_stream.py: Simulates or handles the data stream input for anomaly detection.
ensemble.py: Implements an ensemble approach to combine multiple detection models.
holt_winters.py: Uses the Holt-Winters method for seasonal trend handling.
hybridmain.py: Main script to execute the hybrid model.
isolation_forest.py: Implements Isolation Forest, a machine learning model for anomaly detection in streaming data.
visualization.py: Contains advanced visualization tools specific to the hybrid models.
z_score.py: Z-Score detection for the hybrid model.

# Getting Started

# 1. Clone the repository:
git clone https://github.com/varaprasad5238/ANOMALY-DETECTION.git
# 2. Install required dependencies:
pip install -r requirements.txt
# 3. Run individual models or the hybrid model using:
python hybridmain.py

# Results
The project implements various methods for anomaly detection, including:
Z-Score
KNN
IQR
EMVG
Isolation Forest
Holt-Winters
Hybrid Model
Detailed results are available in the report, with metrics such as Precision, Recall, F1 Score, and ROC AUC.

# Paginator — Multi-API Aggregation with ML-based Sorting

`paginator.py` solves the problem of paginating a dynamically ranked dataset that is aggregated from multiple independent external APIs and sorted by an ML model (stored as a pickle file).

## The Problem

Applying `offset`/`limit` per-API **before** global sorting produces incorrect pages because:

* Each external API returns a different number of records with its own ordering.
* Offset cannot be applied per-API — each API has its own independent dataset.
* Final ranking is determined by an ML/pickle model **after** all records are merged.
* Fetching only `offset + limit` records may miss top-ranked records that haven't been fetched yet.

## The Solution

Treat all API responses as one virtual dataset:

1. Fetch **all** records from every external API.
2. Merge into a single list.
3. Apply global ML-model scoring and sort.
4. Slice with `offset` / `limit`.

## Usage

```python
from paginator import AggregatedPaginator

def fetch_industry_a():
    # call external API A → list of dicts
    return requests.get("https://api-a.example.com/data").json()

def fetch_industry_b():
    # call external API B → list of dicts
    return requests.get("https://api-b.example.com/data").json()

paginator = AggregatedPaginator(
    api_fetchers=[fetch_industry_a, fetch_industry_b],
    model_path="ranking_model.pkl",   # scikit-learn model with predict/predict_proba
)

# Fetch all data, score with model, sort descending
paginator.load_and_sort(feature_keys=["revenue", "growth_rate"])

# Page 1
page1 = paginator.paginate(offset=0, limit=10)
# {"total": 38, "offset": 0, "limit": 10, "data": [...]}

# Page 2
page2 = paginator.paginate(offset=10, limit=10)
```

Without a pickle model, pass a `sort_key` lambda instead:

```python
paginator = AggregatedPaginator(api_fetchers=[fetch_industry_a, fetch_industry_b])
paginator.load_and_sort(sort_key=lambda r: r["anomaly_score"], ascending=False)
page = paginator.paginate(offset=0, limit=20)
```

## Running the tests

```bash
pip install pytest numpy
python -m pytest test_paginator.py -v
```
