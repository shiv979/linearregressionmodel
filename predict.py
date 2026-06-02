"""
predict.py — Quick CLI predictor for California Housing model
Usage: python predict.py
"""
import pickle, numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=== California House Price Predictor ===")
print("Enter values for each feature (press Enter for default):\n")

defaults = [3.5, 25.0, 5.5, 1.1, 1200.0, 3.0, 34.0, -118.0]
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']

inputs = []
for feat, default in zip(features, defaults):
    val = input(f"  {feat} (default={default}): ").strip()
    inputs.append(float(val) if val else default)

pred = model.predict([inputs])[0]
print(f"\n  Predicted Median House Value: ${pred * 100_000:,.0f}")
