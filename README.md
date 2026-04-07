## Notebook 2: Feature Engineering


### Dependencies
- pandas
- numpy
- scikit-learn
- category_encoders
- joblib

### Output Files
- `data/merged_data/train.parquet`
- `data/merged_data/val.parquet`
- `data/merged_data/test.parquet`
- `artifacts/scaler.pkl` — fitted StandardScaler
- `artifacts/encoder.pkl` — dict with keys `ordinal` (OrdinalEncoder) and `target` (TargetEncoder)

### Notes
- Load artifacts with `joblib.load()` 
- 25,065 rows dropped — countries absent from index
- Train/Val/Test split: 70/15/15
- Country column remains as full name (e.g. "Australia") 
- CoL index columns are highly col linear — noted in correlation heatmap