"""
Model 1: Failure Classifier

Predicts failure_category from transaction features at time of failure.
Output classes: hard_decline | soft_decline | customer_dropoff | fraud_block

TODO:
  - Feature engineering / preprocessing pipeline
  - Train/test split
  - Train XGBoost or LightGBM classifier
  - Evaluate: confusion matrix, per-class precision/recall
  - Save model artifact to models/failure_classifier.pkl
"""
