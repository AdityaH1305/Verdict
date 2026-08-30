"""
Model 2: Recovery Success Model

Predicts probability that a retry/nudge recovers a failed transaction,
conditioned on features + Model 1's predicted failure_category.

IMPORTANT: never invoked for fraud_block transactions. This is enforced
in the agent layer (src/agent/), not here -- but do not train or evaluate
on the assumption this model is ever called for fraud_block cases.

TODO:
  - Feature engineering (include predicted category as input feature)
  - Train/test split
  - Train model (classifier w/ probability output, or regressor)
  - Evaluate: AUC, calibration curve
  - Save model artifact to models/recovery_success_model.pkl
"""
