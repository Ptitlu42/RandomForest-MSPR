# Political Orientation Prediction

Random Forest model that predicts political orientation (Second Turn) for French presidential elections.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train model
python src/model/random_forest.py

# Predict 2027 results
python src/utils/predictor.py
```

## Limitations

- Limited dataset (5 elections)
- Historical trend assumption
