# Scalable Machine Learning for Global Internet Performance Tiering Using Ookla Network Speed Data

This project implements a scalable machine learning pipeline for analyzing global internet performance using the Ookla Internet Speed Dataset. The goal is to process large-scale fixed and mobile network performance data, identify interpretable internet performance tiers, and compare machine learning models for latency prediction.

## Project Background

Reliable internet connectivity is essential for education, commerce, healthcare, governance, and communication. However, internet quality varies across locations and connection types. This project uses scalable machine learning to analyze large-scale internet speed records and organize them into meaningful performance tiers.

The project focuses on implementation using PySpark, since the dataset is too large for ordinary full-data processing with pandas.

## Dataset

The dataset used is the Ookla Internet Speed Dataset from Kaggle:

https://www.kaggle.com/datasets/dhruvildave/ookla-internet-speed-dataset

The downloaded dataset contains quarterly folders from 2019-Q1 to 2021-Q4. Each folder includes fixed broadband and mobile network performance files.

In this project:

- Full downloaded dataset folder: approximately 37.80 GB
- Parquet performance files used for ML: approximately 10.29 GB
- Number of Parquet files used: 24
- Connection types: fixed and mobile
- Time coverage: 2019-Q1 to 2021-Q4

Main variables include:

- `avg_d_kbps` — average download speed in kbps
- `avg_u_kbps` — average upload speed in kbps
- `avg_lat_ms` — average latency in milliseconds
- `tests` — number of speed tests
- `devices` — number of devices
- `quadkey` and `tile` — geographic tile identifiers

## Methodology

The notebook follows this pipeline:

1. **PySpark Data Ingestion**
   - Reads multiple quarterly Parquet files
   - Combines fixed and mobile datasets
   - Adds `connection_type`, `year`, and `quarter`

2. **Feature Engineering**
   - Converts kbps to Mbps
   - Creates `download_mbps`, `upload_mbps`, and `latency_ms`
   - Creates `download_upload_ratio`
   - Creates `tests_per_device`
   - Applies log transformations to reduce skewness

3. **Scalable ML Preprocessing**
   - Encodes `connection_type` using `StringIndexer`
   - Combines features using `VectorAssembler`
   - Scales features using `StandardScaler`
   - Produces `scaled_features` for PySpark ML models

4. **K-Means Clustering**
   - Uses K-Means with `k = 3`
   - Groups records into internet performance tiers
   - Evaluates clustering using silhouette score

5. **Latency Prediction**
   - Uses `log_latency_ms` as the target variable
   - Trains Linear Regression as a baseline model
   - Trains Random Forest Regression as a stronger non-linear model
   - Compares models using RMSE, MAE, and R²

## Models Used

### K-Means Clustering

K-Means was used because the dataset does not contain predefined labels for high-, moderate-, or low-performing internet conditions.

Results:

- Silhouette score: `0.2394`
- Cluster 1: High-performance tier
- Cluster 0: Stable / moderate-to-strong tier
- Cluster 2: Low-performance tier

### Linear Regression

Linear Regression was used as the baseline model for latency prediction.

Results:

- RMSE: `0.6745`
- MAE: `0.4864`
- R²: `0.3822`

### Random Forest Regression

Random Forest Regression was used as the stronger non-linear model.

Results:

- RMSE: `0.6550`
- MAE: `0.4696`
- R²: `0.4174`

Random Forest outperformed Linear Regression across all evaluation metrics.

## Key Findings

- K-Means clustering produced three interpretable internet performance tiers.
- Cluster 1 had the highest average download and upload speeds and the lowest latency.
- Cluster 2 had the lowest speeds and highest latency.
- Fixed broadband records were more common in stronger-performing clusters.
- Mobile records were more concentrated in the low-performance cluster.
- Random Forest Regression performed better than Linear Regression for latency prediction.
- Feature importance showed that `log_upload_mbps`, `connection_type_index`, and `log_download_mbps` were the strongest predictors of latency.

## How to Run the Notebook

1. Download the dataset from Kaggle.
2. Extract the dataset locally.
3. Open the notebook in Jupyter Notebook or VSCode.
4. Update the dataset path in the notebook:

```python
DATA_ROOT = r"your/local/path/to/ookla-internet-speed-dataset/versions/9"


## Install required libraries:
pip install pyspark pandas numpy matplotlib

## Run the notebook cells in order
Do not skip cells because later sections depend on earlier DataFrames such as:

project_df
featured_df
ml_df
clustered_df

Important Notes

This project uses PySpark for the main data processing and machine learning pipeline. Pandas is only used for small samples or aggregated outputs for visualization.

Avoid running .toPandas(), .cache(), or .count() on the full dataset unless specifically needed, because these operations may cause memory or disk issues on local machines.

K-Means and regression training may take a long time depending on your hardware.

Recommended Setup
Python 3.10 or higher
PySpark
Java installed and configured
At least 16 GB RAM recommended
Large free disk space for Spark temporary files
Laptop or PC plugged in
Sleep mode disabled during long model training
Project Outputs

The notebook produces:

Cluster summaries
Cluster distribution by connection type
Cluster distribution over time
Spatial overview of sampled cluster locations
Linear Regression metrics
Random Forest Regression metrics
Random Forest feature importance
Model comparison table and visualizations
Limitations

The project was implemented in a local Spark environment, so runtime and memory constraints affected some workflow decisions. Initial EDA used a pilot subset, while the final feature engineering and machine learning models used all available Parquet files.

The clustering results should be interpreted as approximate performance tiers, not strict categories. The regression models also do not fully explain latency because the dataset does not include factors such as routing distance, congestion, server location, or local infrastructure conditions.

Authors
Bautista, Justine Benedict
Diao, Michael Anthony
Reyes, Darylle Joshua
Tagama, Nathan Lee


AI Disclosure

Artificial Intelligence was used to assist with grammar correction, sentence restructuring, stylistic refinement, and brainstorming model selection. All final content, implementation, model results, and interpretations were reviewed and approved by the authors.
