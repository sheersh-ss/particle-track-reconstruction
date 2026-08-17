# Particle Track Reconstruction with Deep Learning

This repository contains the code, trained models, evaluation results,
and prediction scripts developed for the Machine Learning in Particle
Physics and Astronomy assignment.

The project investigates neural-network-based particle track
reconstruction in two-dimensional detector geometries.

## Project Overview

The work consists of four main components:

1. **Monte Carlo simulation**
   - Straight particle trajectories originating from `(0, 0)`
   - Five circular detector layers
   - 95% detector hit efficiency
   - Gaussian detector-position uncertainty
   - 10,000 simulated events

2. **Hit-to-track association**
   - Pairwise neural-network classifier
   - Predicts whether two detector hits originate from the same particle
   - Agglomerative clustering converts pair predictions into reconstructed tracks
   - Evaluated for both straight and curved trajectories

3. **Track-parameter regression**
   - Reconstruction of the straight-track angle
   - Curved-track angular-parameter reconstruction
   - Transverse-momentum regression
   - Particle-charge classification

4. **Scaling study**
   - Straight-track association evaluated with 50 tracks per event
   - Runtime and reconstruction accuracy investigated as detector occupancy increases

## Main Results

| Experiment | Result |
|---|---:|
| Straight tracks, 10 tracks/event | 99.90% hit-assignment accuracy |
| Straight tracks, 50 tracks/event | 99.37% hit-assignment accuracy |
| Mean 50-track reconstruction time | 4.54 s/event |
| Straight-track angle regression | 1.22° MAE |
| Curved-track association | 93.46% hit-assignment accuracy |
| Curved angular-parameter regression | 1.25° MAE |
| Curved transverse-momentum regression | 2.93% MAPE |
| Charge classification | 100% accuracy |

## Repository Structure

```text
particle-track-reconstruction/
├── code/
│   ├── simulation.py
│   ├── pairwise_tracking.py
│   ├── pairwise_tracking_v2.py
│   ├── scaling_50_tracks.py
│   ├── track_parameter_regression.py
│   ├── curved_track_association.py
│   ├── curved_track_parameter_regression.py
│   ├── run_saved_models.py
│   └── generate_submission_predictions.py
│
├── models/
│   ├── pairwise_track_model_v2.keras
│   ├── scaler.npz
│   ├── track_phi_regressor.keras
│   ├── regression_scaler.npz
│   ├── curved_pairwise_model.keras
│   ├── curved_pairwise_scaler.npz
│   ├── curved_parameter_model.keras
│   └── curved_parameter_scaler.npz
│
├── predictions/
│   ├── straight_hit_predictions.csv
│   ├── straight_track_parameters.csv
│   ├── curved_hit_predictions.csv
│   └── curved_track_parameters.csv
│
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```


## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```


Install the required Python packages:
```bash
pip install -r requirements.txt
```
## Data

The raw datasets are not included in this repository.<br>
For the straight-track problem, the simulation can be generated using:
```bash
python code/simulation.py
```
This produces:
```text
simulated_hits.csv
simulated_tracks.csv
```

For local execution, the datasets should be placed inside a data/ directory:
```text
data/
├── simulated_hits.csv
├── simulated_tracks.csv
└── hits_FINAL.csv
```
## Dataset Split

All neural-network models were evaluated using an event-level data split:

* Training events: 0–6999
* Validation events: 7000–8499
* Test events: 8500–9999

Splitting by event prevents detector hits or hit pairs originating from the same collision event from appearing in both training and evaluation datasets.

## Running the Saved Models

The repository contains a script for checking that all trained models and corresponding feature scalers can be loaded correctly:
```bash
python code/run_saved_models.py
```
The script verifies the input dimensions of each model and scaler and performs a dummy inference pass.

## Generating Final Predictions

Prediction files are generated for the held-out test set only.

For straight-track events:
```bash
python code/generate_submission_predictions.py \
    --dataset straight \
    --start-event 8500 \
    --end-event 10000
```
For curved-track events:
```bash
python code/generate_submission_predictions.py \
    --dataset curved \
    --start-event 8500 \
    --end-event 10000
```
The generated files are stored in the predictions/ directory:
```text
straight_hit_predictions.csv
straight_track_parameters.csv
curved_hit_predictions.csv
curved_track_parameters.csv
```
Each hit-association file contains one event per row. The event identifier is followed by the measured hit coordinates and the corresponding predicted track assignment.

The track-parameter files contain the reconstructed parameters for the predicted tracks in each event.

## Method

Track identifiers are arbitrary and therefore are not treated as fixed classification classes.

Instead, hit association is formulated as a pairwise binary-classification problem. For every candidate pair of detector hits, a neural network predicts whether both hits originated from the same physical particle.

The resulting pairwise probabilities are converted into a similarity matrix, after which agglomerative clustering is used to group hits into reconstructed tracks.

For the straight-track problem, geometrical input features include hit positions, detector radii, detector layers, coordinate differences, angular differences, and distances between hits.

For curved trajectories, additional features sensitive to curvature are included.

For the reported clustering experiments, the number of tracks in each event is assumed to be known.

## Track-Parameter Reconstruction

Separate neural networks are used to reconstruct track parameters.

For straight trajectories, the track direction is represented using
```text
sin(phi), cos(phi)
```
rather than directly regressing the angle. This avoids problems caused by the periodicity of angular quantities.<br>
For curved trajectories, a multi-task neural network predicts:

* the angular track parameter,
* transverse momentum (pt),
* particle charge.

The angular quantity is again represented using sine and cosine components, while charge is treated as a binary classification problem.

## Scaling

The pairwise method naturally supports events containing a variable number of detector hits.
However, for an event containing N hits, the number of possible hit pairs is
```text
N(N - 1) / 2
```
and therefore grows quadratically with event size.
The 50-track scaling experiment retained a global hit-assignment accuracy of approximately 99.37%, but required approximately 4.54 seconds per event in the tested setup.

## Notes

The reported hit-assignment accuracies use permutation-invariant comparison between predicted clusters and true tracks because numerical track identifiers have no physical meaning.
The curved-track clustering evaluation assumes that the number of physical tracks in each event is known to the clustering algorithm. The model itself does not use the true hit-to-track assignments during inference.

## Author

Sheersh Srivastava
