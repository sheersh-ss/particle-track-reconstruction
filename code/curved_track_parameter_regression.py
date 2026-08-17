import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_FILE = "hits_FINAL.csv"

OUTPUT_DIR = Path(
    "curved_parameter_regression_results"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

MODEL_FILE = (
    OUTPUT_DIR /
    "curved_parameter_model.keras"
)

SCALER_FILE = (
    OUTPUT_DIR /
    "curved_parameter_scaler.npz"
)

RESULTS_FILE = (
    OUTPUT_DIR /
    "curved_parameter_results.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR /
    "curved_parameter_predictions.csv"
)

N_LAYERS = 5

TRAIN_END = 7000
VAL_END = 8500
TEST_END = 10000

MAX_EPOCHS = 100
BATCH_SIZE = 256


# ============================================================
# Load data
# ============================================================

print("Loading curved-track dataset...")

hits_df = pd.read_csv(
    DATA_FILE
)

print()
print("==============================")
print("Dataset")
print("==============================")

print(
    "Rows:",
    len(hits_df)
)

print(
    "Events:",
    hits_df[
        "event_id"
    ].nunique()
)

print(
    "Columns:",
    list(
        hits_df.columns
    )
)


# ============================================================
# Build fixed-size input for one track
# ============================================================

def build_track_features(
    track_hits
):
    """
    Represent one curved track using its detector hits.

    For each of the five detector layers we store:

        x
        y
        radius
        phi_hit
        mask

    giving 5 features x 5 layers = 25 features.

    Missing hits are represented by zeros and mask = 0.
    """

    features = []

    for layer in range(
        N_LAYERS
    ):

        layer_hits = (
            track_hits[
                track_hits["layer"]
                == layer
            ]
        )

        if len(
            layer_hits
        ) > 0:

            hit = (
                layer_hits
                .iloc[0]
            )

            x = float(
                hit["x"]
            )

            y = float(
                hit["y"]
            )

            radius = np.sqrt(
                x ** 2
                +
                y ** 2
            )

            phi_hit = np.arctan2(
                y,
                x
            )

            mask = 1.0

        else:

            x = 0.0
            y = 0.0
            radius = 0.0
            phi_hit = 0.0
            mask = 0.0

        features.extend(
            [
                x,
                y,
                radius,
                phi_hit,
                mask
            ]
        )

    return features


# ============================================================
# Build one sample per physical track
# ============================================================

def build_dataset(
    hits
):
    """
    Convert hit-level data into track-level samples.

    Targets:

    theta:
        represented as sin(theta), cos(theta)

    pt:
        continuous regression target

    charge:
        binary classification target
        -1 -> 0
        +1 -> 1
    """

    X = []

    theta_targets = []
    pt_targets = []
    charge_targets = []

    event_ids = []
    track_ids = []

    true_theta = []
    true_pt = []
    true_charge = []

    grouped = hits.groupby(
        [
            "event_id",
            "track_id"
        ]
    )

    total_groups = (
        grouped.ngroups
    )

    print(
        f"Building {total_groups} "
        f"track samples..."
    )

    for counter, (
        (
            event_id,
            track_id
        ),
        track_hits
    ) in enumerate(
        grouped
    ):

        first_hit = (
            track_hits.iloc[0]
        )

        theta = float(
            first_hit["theta"]
        )

        pt = float(
            first_hit["pt"]
        )

        charge = int(
            first_hit["charge"]
        )

        features = (
            build_track_features(
                track_hits
            )
        )

        X.append(
            features
        )

        # Circular angle target
        theta_targets.append(
            [
                np.sin(theta),
                np.cos(theta)
            ]
        )

        # Continuous momentum target
        pt_targets.append(
            pt
        )

        # Binary charge target
        charge_targets.append(
            1.0
            if charge == 1
            else 0.0
        )

        event_ids.append(
            int(event_id)
        )

        track_ids.append(
            int(track_id)
        )

        true_theta.append(
            theta
        )

        true_pt.append(
            pt
        )

        true_charge.append(
            charge
        )

        if (
            counter + 1
        ) % 10000 == 0:

            print(
                f"Processed "
                f"{counter + 1}/"
                f"{total_groups}"
            )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            theta_targets,
            dtype=np.float32
        ),
        np.asarray(
            pt_targets,
            dtype=np.float32
        ),
        np.asarray(
            charge_targets,
            dtype=np.float32
        ),
        np.asarray(
            event_ids
        ),
        np.asarray(
            track_ids
        ),
        np.asarray(
            true_theta,
            dtype=np.float32
        ),
        np.asarray(
            true_pt,
            dtype=np.float32
        ),
        np.asarray(
            true_charge,
            dtype=np.int32
        )
    )


(
    X,
    y_theta,
    y_pt,
    y_charge,
    event_ids,
    track_ids,
    true_theta,
    true_pt,
    true_charge
) = build_dataset(
    hits_df
)


print()
print("==============================")
print("Track-level dataset")
print("==============================")

print(
    "X shape:",
    X.shape
)

print(
    "Theta targets:",
    y_theta.shape
)

print(
    "pt targets:",
    y_pt.shape
)

print(
    "Charge targets:",
    y_charge.shape
)


# ============================================================
# Event-level split
# ============================================================

train_mask = (
    event_ids
    < TRAIN_END
)

val_mask = (
    (event_ids >= TRAIN_END)
    &
    (event_ids < VAL_END)
)

test_mask = (
    (event_ids >= VAL_END)
    &
    (event_ids < TEST_END)
)


X_train = X[
    train_mask
]

X_val = X[
    val_mask
]

X_test = X[
    test_mask
]


theta_train = y_theta[
    train_mask
]

theta_val = y_theta[
    val_mask
]

theta_test = y_theta[
    test_mask
]


pt_train = y_pt[
    train_mask
]

pt_val = y_pt[
    val_mask
]

pt_test = y_pt[
    test_mask
]


charge_train = y_charge[
    train_mask
]

charge_val = y_charge[
    val_mask
]

charge_test = y_charge[
    test_mask
]


true_theta_test = true_theta[
    test_mask
]

true_pt_test = true_pt[
    test_mask
]

true_charge_test = true_charge[
    test_mask
]


test_event_ids = event_ids[
    test_mask
]

test_track_ids = track_ids[
    test_mask
]


print()
print("==============================")
print("Event-level split")
print("==============================")

print(
    "Training tracks:",
    len(X_train)
)

print(
    "Validation tracks:",
    len(X_val)
)

print(
    "Test tracks:",
    len(X_test)
)


# ============================================================
# Input standardization
# ============================================================

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_val_scaled = (
    scaler.transform(
        X_val
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)


np.savez(
    SCALER_FILE,
    mean=scaler.mean_,
    scale=scaler.scale_
)


# ============================================================
# Standardize pt target
# ============================================================

pt_mean = float(
    pt_train.mean()
)

pt_std = float(
    pt_train.std()
)


pt_train_scaled = (
    pt_train - pt_mean
) / pt_std

pt_val_scaled = (
    pt_val - pt_mean
) / pt_std

pt_test_scaled = (
    pt_test - pt_mean
) / pt_std


print()
print(
    f"Training pt mean: "
    f"{pt_mean:.4f}"
)

print(
    f"Training pt std: "
    f"{pt_std:.4f}"
)


# ============================================================
# Multi-task neural network
# ============================================================

inputs = Input(
    shape=(
        X_train_scaled.shape[1],
    ),
    name="track_hits"
)


shared = Dense(
    128,
    activation="relu"
)(
    inputs
)

shared = Dropout(
    0.15
)(
    shared
)

shared = Dense(
    128,
    activation="relu"
)(
    shared
)

shared = Dropout(
    0.15
)(
    shared
)

shared = Dense(
    64,
    activation="relu"
)(
    shared
)


# ------------------------------------------------------------
# Theta output
# ------------------------------------------------------------

theta_branch = Dense(
    32,
    activation="relu"
)(
    shared
)

theta_output = Dense(
    2,
    activation="linear",
    name="theta_output"
)(
    theta_branch
)


# ------------------------------------------------------------
# pt output
# ------------------------------------------------------------

pt_branch = Dense(
    32,
    activation="relu"
)(
    shared
)

pt_output = Dense(
    1,
    activation="linear",
    name="pt_output"
)(
    pt_branch
)


# ------------------------------------------------------------
# Charge output
# ------------------------------------------------------------

charge_branch = Dense(
    32,
    activation="relu"
)(
    shared
)

charge_output = Dense(
    1,
    activation="sigmoid",
    name="charge_output"
)(
    charge_branch
)


model = Model(
    inputs=inputs,
    outputs={
        "theta_output":
            theta_output,

        "pt_output":
            pt_output,

        "charge_output":
            charge_output
    }
)


model.compile(
    optimizer=Adam(
        learning_rate=1e-3
    ),

    loss={
        "theta_output":
            "mse",

        "pt_output":
            "mse",

        "charge_output":
            "binary_crossentropy"
    },

    loss_weights={
        "theta_output":
            1.0,

        "pt_output":
            1.0,

        "charge_output":
            1.0
    },

    metrics={
        "theta_output":
            ["mae"],

        "pt_output":
            ["mae"],

        "charge_output":
            ["accuracy"]
    }
)


model.summary()


# ============================================================
# Training
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


history = model.fit(

    X_train_scaled,

    {
        "theta_output":
            theta_train,

        "pt_output":
            pt_train_scaled,

        "charge_output":
            charge_train
    },

    validation_data=(

        X_val_scaled,

        {
            "theta_output":
                theta_val,

            "pt_output":
                pt_val_scaled,

            "charge_output":
                charge_val
        }
    ),

    epochs=MAX_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# Predictions
# ============================================================

predictions = model.predict(
    X_test_scaled,
    verbose=0
)


pred_theta_vector = (
    predictions[
        "theta_output"
    ]
)

pred_pt_scaled = (
    predictions[
        "pt_output"
    ].flatten()
)

pred_charge_probability = (
    predictions[
        "charge_output"
    ].flatten()
)


# ============================================================
# Theta reconstruction
# ============================================================

pred_theta = np.arctan2(
    pred_theta_vector[:, 0],
    pred_theta_vector[:, 1]
)

# Convert [-pi, pi] to [0, 2pi]
pred_theta = np.mod(
    pred_theta,
    2 * np.pi
)


def angular_difference(
    prediction,
    truth
):

    difference = (
        prediction - truth
    )

    return np.arctan2(
        np.sin(difference),
        np.cos(difference)
    )


theta_errors = (
    angular_difference(
        pred_theta,
        true_theta_test
    )
)

theta_absolute_errors = (
    np.abs(
        theta_errors
    )
)


theta_mae_rad = float(
    np.mean(
        theta_absolute_errors
    )
)

theta_rmse_rad = float(
    np.sqrt(
        np.mean(
            theta_errors ** 2
        )
    )
)

theta_mae_deg = float(
    np.degrees(
        theta_mae_rad
    )
)

theta_rmse_deg = float(
    np.degrees(
        theta_rmse_rad
    )
)

theta_median_deg = float(
    np.degrees(
        np.median(
            theta_absolute_errors
        )
    )
)

theta_95_deg = float(
    np.degrees(
        np.percentile(
            theta_absolute_errors,
            95
        )
    )
)


# ============================================================
# pt reconstruction
# ============================================================

pred_pt = (
    pred_pt_scaled
    * pt_std
    +
    pt_mean
)


pt_errors = (
    pred_pt
    -
    true_pt_test
)

pt_absolute_errors = (
    np.abs(
        pt_errors
    )
)


pt_mae = float(
    np.mean(
        pt_absolute_errors
    )
)

pt_rmse = float(
    np.sqrt(
        np.mean(
            pt_errors ** 2
        )
    )
)

pt_mape = float(
    np.mean(
        np.abs(
            pt_errors
            /
            true_pt_test
        )
    )
    * 100.0
)

pt_median_error = float(
    np.median(
        pt_absolute_errors
    )
)

pt_95_error = float(
    np.percentile(
        pt_absolute_errors,
        95
    )
)


# ============================================================
# Charge prediction
# ============================================================

pred_charge_binary = (
    pred_charge_probability
    >= 0.5
).astype(int)


pred_charge = np.where(
    pred_charge_binary == 1,
    1,
    -1
)


charge_accuracy = float(
    accuracy_score(
        true_charge_test,
        pred_charge
    )
)


charge_cm = confusion_matrix(
    true_charge_test,
    pred_charge,
    labels=[
        -1,
        1
    ]
)


# ============================================================
# Print results
# ============================================================

print()
print("==============================")
print("Theta regression")
print("==============================")

print(
    f"Angular MAE: "
    f"{theta_mae_rad:.8f} rad"
)

print(
    f"Angular MAE: "
    f"{theta_mae_deg:.6f} degrees"
)

print(
    f"Angular RMSE: "
    f"{theta_rmse_deg:.6f} degrees"
)

print(
    f"Median absolute error: "
    f"{theta_median_deg:.6f} degrees"
)

print(
    f"95th percentile error: "
    f"{theta_95_deg:.6f} degrees"
)


print()
print("==============================")
print("pt regression")
print("==============================")

print(
    f"MAE: "
    f"{pt_mae:.6f}"
)

print(
    f"RMSE: "
    f"{pt_rmse:.6f}"
)

print(
    f"MAPE: "
    f"{pt_mape:.3f}%"
)

print(
    f"Median absolute error: "
    f"{pt_median_error:.6f}"
)

print(
    f"95th percentile error: "
    f"{pt_95_error:.6f}"
)


print()
print("==============================")
print("Charge classification")
print("==============================")

print(
    f"Accuracy: "
    f"{charge_accuracy:.6f}"
)

print()
print(
    "Confusion matrix "
    "[-1, +1]:"
)

print(
    charge_cm
)


# ============================================================
# Count number of hits per reconstructed track
# ============================================================

n_hits_test = np.sum(
    X_test[
        :,
        4::5
    ],
    axis=1
)


# ============================================================
# Error by number of detector hits
# ============================================================

print()
print("==============================")
print("Performance by number of hits")
print("==============================")

performance_by_hits = {}


for number_hits in range(
    1,
    N_LAYERS + 1
):

    mask = (
        n_hits_test
        == number_hits
    )

    n_tracks = int(
        np.sum(
            mask
        )
    )

    if n_tracks == 0:
        continue

    mean_theta_error = float(
        np.mean(
            np.degrees(
                theta_absolute_errors[
                    mask
                ]
            )
        )
    )

    mean_pt_error = float(
        np.mean(
            pt_absolute_errors[
                mask
            ]
        )
    )

    charge_acc = float(
        accuracy_score(
            true_charge_test[
                mask
            ],
            pred_charge[
                mask
            ]
        )
    )

    print(
        f"{number_hits} hits: "
        f"theta MAE = "
        f"{mean_theta_error:.4f} deg, "
        f"pt MAE = "
        f"{mean_pt_error:.4f}, "
        f"charge acc = "
        f"{charge_acc:.4f} "
        f"({n_tracks} tracks)"
    )

    performance_by_hits[
        str(number_hits)
    ] = {
        "n_tracks":
            n_tracks,

        "theta_mae_degrees":
            mean_theta_error,

        "pt_mae":
            mean_pt_error,

        "charge_accuracy":
            charge_acc
    }


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame({

    "event_id":
        test_event_ids,

    "track_id":
        test_track_ids,

    "true_theta":
        true_theta_test,

    "predicted_theta":
        pred_theta,

    "theta_absolute_error_deg":
        np.degrees(
            theta_absolute_errors
        ),

    "true_pt":
        true_pt_test,

    "predicted_pt":
        pred_pt,

    "pt_absolute_error":
        pt_absolute_errors,

    "true_charge":
        true_charge_test,

    "predicted_charge":
        pred_charge,

    "charge_probability_positive":
        pred_charge_probability,

    "n_hits":
        n_hits_test
})


prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# Training loss plot
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history[
        "loss"
    ],
    label="Training"
)

plt.plot(
    history.history[
        "val_loss"
    ],
    label="Validation"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Total multi-task loss"
)

plt.title(
    "Curved-track parameter training loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "training_loss.png",
    dpi=200
)

plt.close()


# ============================================================
# True vs predicted theta
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    true_theta_test,
    pred_theta,
    s=5,
    alpha=0.3
)

plt.plot(
    [
        0,
        2 * np.pi
    ],
    [
        0,
        2 * np.pi
    ],
    linestyle="--"
)

plt.xlabel(
    "True theta [rad]"
)

plt.ylabel(
    "Predicted theta [rad]"
)

plt.title(
    "True vs predicted theta"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "true_vs_predicted_theta.png",
    dpi=200
)

plt.close()


# ============================================================
# Theta error histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    np.degrees(
        theta_errors
    ),
    bins=100
)

plt.xlabel(
    "Theta error [degrees]"
)

plt.ylabel(
    "Number of tracks"
)

plt.title(
    "Theta reconstruction error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "theta_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# True vs predicted pt
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    true_pt_test,
    pred_pt,
    s=5,
    alpha=0.3
)

minimum_pt = min(
    true_pt_test.min(),
    pred_pt.min()
)

maximum_pt = max(
    true_pt_test.max(),
    pred_pt.max()
)

plt.plot(
    [
        minimum_pt,
        maximum_pt
    ],
    [
        minimum_pt,
        maximum_pt
    ],
    linestyle="--"
)

plt.xlabel(
    "True pt"
)

plt.ylabel(
    "Predicted pt"
)

plt.title(
    "True vs predicted transverse momentum"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "true_vs_predicted_pt.png",
    dpi=200
)

plt.close()


# ============================================================
# pt error histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    pt_errors,
    bins=100
)

plt.xlabel(
    "pt prediction error"
)

plt.ylabel(
    "Number of tracks"
)

plt.title(
    "Transverse-momentum regression error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "pt_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# Charge confusion matrix
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    charge_cm
)

plt.xticks(
    [0, 1],
    [
        "-1",
        "+1"
    ]
)

plt.yticks(
    [0, 1],
    [
        "-1",
        "+1"
    ]
)

plt.xlabel(
    "Predicted charge"
)

plt.ylabel(
    "True charge"
)

plt.title(
    "Charge classification confusion matrix"
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            str(
                charge_cm[
                    i,
                    j
                ]
            ),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "charge_confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# Error versus hit count
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    n_hits_test,
    np.degrees(
        theta_absolute_errors
    ),
    s=5,
    alpha=0.3
)

plt.xlabel(
    "Number of detector hits"
)

plt.ylabel(
    "Absolute theta error [degrees]"
)

plt.title(
    "Theta error versus number of hits"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "theta_error_vs_hits.png",
    dpi=200
)

plt.close()


# ============================================================
# Save model
# ============================================================

model.save(
    MODEL_FILE
)


# ============================================================
# Save metrics
# ============================================================

results = {

    "theta_mae_radians":
        theta_mae_rad,

    "theta_mae_degrees":
        theta_mae_deg,

    "theta_rmse_degrees":
        theta_rmse_deg,

    "theta_median_absolute_error_degrees":
        theta_median_deg,

    "theta_95_percentile_error_degrees":
        theta_95_deg,

    "pt_mae":
        pt_mae,

    "pt_rmse":
        pt_rmse,

    "pt_mape_percent":
        pt_mape,

    "pt_median_absolute_error":
        pt_median_error,

    "pt_95_percentile_error":
        pt_95_error,

    "charge_accuracy":
        charge_accuracy,

    "training_tracks":
        int(
            len(X_train)
        ),

    "validation_tracks":
        int(
            len(X_val)
        ),

    "test_tracks":
        int(
            len(X_test)
        ),

    "pt_training_mean":
        pt_mean,

    "pt_training_std":
        pt_std,

    "performance_by_hit_count":
        performance_by_hits
}


with open(
    RESULTS_FILE,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# Finished
# ============================================================

print()
print("==============================")
print("Finished")
print("==============================")

print(
    f"Model saved to: "
    f"{MODEL_FILE}"
)

print(
    f"Predictions saved to: "
    f"{PREDICTIONS_FILE}"
)

print(
    f"Results saved to: "
    f"{RESULTS_FILE}"
)

print(
    f"Figures saved in: "
    f"{OUTPUT_DIR}"
)

print()
print("Done.")