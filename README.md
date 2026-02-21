# Nuclear Accident Classification System (NAS)

Real-time nuclear accident diagnosis system using deep learning.
9-class classifier with TCN model and guard-based diagnostic logic.

## Classification Classes (9)

| ID | Class | Description |
|----|-------|-------------|
| 0 | NORMAL | Normal operation |
| 1 | LOCA_HL | Loss of Coolant Accident (Hot Leg) |
| 2 | LOCA_CL | Loss of Coolant Accident (Cold Leg) |
| 3 | LOCA_RCP | Loss of Coolant Accident (RCP Seal) |
| 4 | SGTR_Loop1 | Steam Generator Tube Rupture (Loop 1) |
| 5 | SGTR_Loop2 | Steam Generator Tube Rupture (Loop 2) |
| 6 | SGTR_Loop3 | Steam Generator Tube Rupture (Loop 3) |
| 7 | ESDE_in | Excessive Steam Demand Event (Inside) |
| 8 | ESDE_out | Excessive Steam Demand Event (Outside) |

## Model

- **Architecture**: TCN (Temporal Convolutional Network) V3
- **Training**: 100 epochs, seed=0
- **Features**: 266 physics-based features (`physics_v3`)
- **Input**: `(batch, 3, 266)` — 3-second sliding window
- **Output**: `(batch, 9)` — softmax probabilities

## Diagnostic Logic

- **SGTR**: 2 consecutive identical predictions
- **ESDE**: 2 consecutive + ESDE Guard (if LOCA prob sum > 0.05 in last 3s, hold)
- **LOCA_CL**: 4 consecutive, immediate confirmation
- **LOCA_HL / LOCA_RCP**: 4 consecutive + CL Guard (if CL prob max > 0.15 in last 5s, hold)
- **Grace Period**: No confirmation in first 3 seconds
- **First diagnosis = Final diagnosis** (no late correction)

## Test Results

**459 / 467 correct (98.3%)**

## Project Structure

```
Team 6 code/          # Final submission
├── py/
│   ├── main.py       # Real-time inference pipeline
│   ├── UDP_read.py   # UDP receiver (saves incoming data)
│   └── practice/     # Feature engineering modules
├── models/           # Trained model files (5-file set)
└── ref/              # Column type reference

Team N code/          # Original version (with Late Correction)
```

## Tech Stack

- Python 3.11
- TensorFlow 2.20
- NumPy, scikit-learn, joblib, pandas
