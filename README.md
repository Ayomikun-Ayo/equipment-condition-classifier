# Equipment Condition Classifier

A machine learning model that predicts whether industrial equipment needs inspection, based on live sensor readings.

## Problem

Equipment failures are costly and often preventable if warning signs are caught early. This project uses sensor data (vibration, temperature, oil pressure, sound level) to classify a machine's condition as either **Normal** or **Inspect**, flagging machines that may need maintenance attention before they break down.

## Dataset

- 72 equipment readings, each with 4 sensor features and a labeled condition
- Perfectly balanced: 36 "Normal" / 36 "Inspect"
- Features:
  - `vibration_mm_s` — vibration level
  - `temperature_c` — operating temperature
  - `oil_pressure_bar` — oil pressure
  - `sound_level_db` — sound level
- Target: `condition` (Normal / Inspect)

## Approach

1. Loaded and inspected the data with pandas
2. Split into train/test sets (75/25, stratified to preserve class balance)
3. Trained a **Decision Tree Classifier** (`scikit-learn`)
4. Evaluated on the held-out test set
5. Visualized the tree to inspect the learned decision rules
6. Checked feature importance to see which sensors mattered most
7. Ran predictions on new, unseen sample readings

## Results

- **Accuracy: 94.4%** (17/18 correct on the test set)
- Confusion matrix:
  ```
  [[9 0]
   [1 8]]
  ```
  One "Normal" machine was misclassified as "Inspect" — a reasonable direction to err on for equipment safety.

## Key finding

Two sensors dominate the model's decisions:

| Feature | Importance |
|---|---|
| `oil_pressure_bar` | 44.6% |
| `sound_level_db` | 39.4% |
| `temperature_c` | 10.4% |
| `vibration_mm_s` | 5.6% |

- **Low oil pressure (2.5 bar)** is the single strongest signal — it alone accounts for about two-thirds of all "Inspect" flags.
- When oil pressure is fine, a machine can still get flagged if **vibration and temperature are both elevated** (vibration ≥ 5.5 mm/s and temperature ≥ 72°C) — a secondary warning pattern.

## Files

- `train_model.ipynb` — full notebook (data loading → training → evaluation → visualization)
- `AI_BBI_M1_Equipment_Condition_Demo.csv` — dataset
- `equipment_condition_model.pkl` — saved trained model

## How to run

1. Open `train_model.ipynb` in Jupyter or Google Colab
2. Run all cells in order
3. To reuse the saved model without retraining:
   ```python
   import joblib
   model = joblib.load("equipment_condition_model.pkl")
   model.predict(new_readings)  # new_readings: DataFrame with the 4 feature columns
   ```
