import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

HITS_FILE = "simulated_hits.csv"
TRACKS_FILE = "simulated_tracks.csv"

OUTPUT_DIR = Path("regression_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "track_phi_regressor.keras"

SCALER_FILE = OUTPUT_DIR / "regression_scaler.npz"

RESULTS_FILE = OUTPUT_DIR / "regression_results.json"

N_LAYERS = 5

TRAIN_END = 7000
VAL_END = 8500
TEST_END = 10000

MAX_EPOCHS = 100
BATCH_SIZE = 256


# ============================================================
# Load data
# ============================================================

print("Loading datasets...")

hits_df = pd.read_csv(HITS_FILE)
tracks_df = pd.read_csv(TRACKS_FILE)

print(
    f"Hits: {len(hits_df)}"
)

print(
    f"Tracks: {len(tracks_df)}"
)

print(
    f"Events: "
    f"{tracks_df['event_id'].nunique()}"
)


# ============================================================
# Convert one track into a fixed-size input
# ============================================================

def build_track_input(track_hits):
    """
    Convert detector hits from one track into a fixed-size
    vector.

    For each detector layer, store:

        x, y, mask

    mask = 1 if a hit exists
    mask = 0 if the detector hit was missing.

    With 5 detector layers this gives:

        5 * 3 = 15 input features.
    """

    features = []

    for layer in range(N_LAYERS):

        layer_hits = track_hits[
            track_hits["layer"] == layer
        ]

        if len(layer_hits) > 0:

            hit = layer_hits.iloc[0]

            x = hit["x"]
            y = hit["y"]

            mask = 1.0

        else:

            x = 0.0
            y = 0.0

            mask = 0.0

        features.extend(
            [
                x,
                y,
                mask
            ]
        )

    return features


# ============================================================
# Build regression dataset
# ============================================================

def build_regression_dataset(
    hits,
    tracks
):
    """
    Build one ML sample for every particle track.

    Input:
        detector hits belonging to the track

    Target:
        sin(phi), cos(phi)

    Returns:
        X
        y
        event IDs
        track IDs
        true phi values
    """

    X = []
    y = []

    event_ids = []
    track_ids = []
    true_phis = []

    total_tracks = len(
        tracks
    )

    print(
        f"Building regression samples "
        f"for {total_tracks} tracks..."
    )

    grouped_hits = hits.groupby(
        ["event_id", "track_id"]
    )

    for counter, track in tracks.iterrows():

        event_id = int(
            track["event_id"]
        )

        track_id = int(
            track["track_id"]
        )

        phi = float(
            track["phi"]
        )

        key = (
            event_id,
            track_id
        )

        if key not in grouped_hits.groups:

            # This would mean all five hits were lost.
            # Very unlikely with 95% efficiency.
            continue

        track_hits = grouped_hits.get_group(
            key
        )

        features = build_track_input(
            track_hits
        )

        # Circular regression target
        target = [
            np.sin(phi),
            np.cos(phi)
        ]

        X.append(
            features
        )

        y.append(
            target
        )

        event_ids.append(
            event_id
        )

        track_ids.append(
            track_id
        )

        true_phis.append(
            phi
        )

        if (
            (counter + 1) % 10000
            == 0
        ):

            print(
                f"Processed "
                f"{counter + 1}/"
                f"{total_tracks}"
            )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),

        np.asarray(
            y,
            dtype=np.float32
        ),

        np.asarray(
            event_ids
        ),

        np.asarray(
            track_ids
        ),

        np.asarray(
            true_phis,
            dtype=np.float32
        )
    )


X, y, event_ids, track_ids, true_phis = (
    build_regression_dataset(
        hits_df,
        tracks_df
    )
)


print()
print("==============================")
print("Regression dataset")
print("==============================")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# Event-level train / validation / test split
# ============================================================

train_mask = (
    event_ids < TRAIN_END
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

y_train = y[
    train_mask
]


X_val = X[
    val_mask
]

y_val = y[
    val_mask
]


X_test = X[
    test_mask
]

y_test = y[
    test_mask
]


phi_test = true_phis[
    test_mask
]


print()
print("==============================")
print("Dataset split")
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
# Standardize inputs
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# Save the scaler
np.savez(
    SCALER_FILE,
    mean=scaler.mean_,
    scale=scaler.scale_
)


# ============================================================
# Build regression neural network
# ============================================================

model = Sequential([
    Input(
        shape=(
            X_train_scaled.shape[1],
        )
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.1
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        2,
        activation="linear"
    )
])


model.compile(
    optimizer=Adam(
        learning_rate=1e-3
    ),

    loss="mse",

    metrics=[
        "mae"
    ]
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
    y_train,

    validation_data=(
        X_val_scaled,
        y_val
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


pred_sin = predictions[
    :, 0
]

pred_cos = predictions[
    :, 1
]


# Convert predicted sin/cos back to phi
pred_phi = np.arctan2(
    pred_sin,
    pred_cos
)


# ============================================================
# Circular angular error
# ============================================================

def angular_difference(
    predicted,
    true
):
    """
    Wrapped difference between two angles.

    Result is between -pi and +pi.
    """

    difference = (
        predicted - true
    )

    return np.arctan2(
        np.sin(difference),
        np.cos(difference)
    )


errors = angular_difference(
    pred_phi,
    phi_test
)

absolute_errors = np.abs(
    errors
)


# ============================================================
# Metrics
# ============================================================

mae_radians = np.mean(
    absolute_errors
)

rmse_radians = np.sqrt(
    np.mean(
        errors ** 2
    )
)


mae_degrees = np.degrees(
    mae_radians
)

rmse_degrees = np.degrees(
    rmse_radians
)


median_error_degrees = np.degrees(
    np.median(
        absolute_errors
    )
)


percentile_95_degrees = np.degrees(
    np.percentile(
        absolute_errors,
        95
    )
)


maximum_error_degrees = np.degrees(
    np.max(
        absolute_errors
    )
)


print()
print("==============================")
print("Track parameter regression")
print("==============================")

print(
    f"Angular MAE: "
    f"{mae_radians:.8f} rad"
)

print(
    f"Angular MAE: "
    f"{mae_degrees:.6f} degrees"
)

print()

print(
    f"Angular RMSE: "
    f"{rmse_radians:.8f} rad"
)

print(
    f"Angular RMSE: "
    f"{rmse_degrees:.6f} degrees"
)

print()

print(
    f"Median absolute error: "
    f"{median_error_degrees:.6f} degrees"
)

print(
    f"95th percentile error: "
    f"{percentile_95_degrees:.6f} degrees"
)

print(
    f"Maximum error: "
    f"{maximum_error_degrees:.6f} degrees"
)


# ============================================================
# Inspect magnitude of predicted sin/cos vectors
# ============================================================

predicted_norms = np.sqrt(
    pred_sin ** 2
    +
    pred_cos ** 2
)


print()

print(
    f"Mean predicted vector norm: "
    f"{predicted_norms.mean():.6f}"
)


# ============================================================
# Save prediction table
# ============================================================

test_event_ids = event_ids[
    test_mask
]

test_track_ids = track_ids[
    test_mask
]


prediction_df = pd.DataFrame({

    "event_id":
        test_event_ids,

    "track_id":
        test_track_ids,

    "true_phi":
        phi_test,

    "predicted_phi":
        pred_phi,

    "error_rad":
        errors,

    "absolute_error_rad":
        absolute_errors,

    "absolute_error_deg":
        np.degrees(
            absolute_errors
        )
})


prediction_df.to_csv(
    OUTPUT_DIR /
    "track_parameter_predictions.csv",

    index=False
)


# ============================================================
# Training loss plot
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean squared error"
)

plt.title(
    "Track parameter regression loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "regression_training_loss.png",
    dpi=200
)

plt.close()


# ============================================================
# MAE training plot
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["mae"],
    label="Training"
)

plt.plot(
    history.history["val_mae"],
    label="Validation"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Regression training history"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "regression_training_mae.png",
    dpi=200
)

plt.close()


# ============================================================
# Predicted phi vs true phi
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    phi_test,
    pred_phi,
    s=5,
    alpha=0.3
)

plt.plot(
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    linestyle="--"
)

plt.xlabel(
    "True phi [rad]"
)

plt.ylabel(
    "Predicted phi [rad]"
)

plt.title(
    "True vs predicted track angle"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "true_vs_predicted_phi.png",
    dpi=200
)

plt.close()


# ============================================================
# Angular error histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    np.degrees(
        errors
    ),
    bins=100
)

plt.xlabel(
    "Prediction error [degrees]"
)

plt.ylabel(
    "Number of tracks"
)

plt.title(
    "Distribution of track-angle regression errors"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "angular_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# Absolute angular error histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    np.degrees(
        absolute_errors
    ),
    bins=100
)

plt.xlabel(
    "Absolute angular error [degrees]"
)

plt.ylabel(
    "Number of tracks"
)

plt.title(
    "Absolute error in reconstructed track angle"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "absolute_angular_error.png",
    dpi=200
)

plt.close()


# ============================================================
# Error versus number of recorded hits
# ============================================================

n_recorded_hits = np.sum(
    X_test[
        :,
        2::3
    ],
    axis=1
)


plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    n_recorded_hits,
    np.degrees(
        absolute_errors
    ),
    s=5,
    alpha=0.3
)

plt.xlabel(
    "Recorded hits on track"
)

plt.ylabel(
    "Absolute angular error [degrees]"
)

plt.title(
    "Regression error versus number of detector hits"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "error_vs_number_of_hits.png",
    dpi=200
)

plt.close()


# ============================================================
# Mean error for tracks with 1-5 hits
# ============================================================

print()
print("==============================")
print("Error by number of hits")
print("==============================")


hit_count_results = {}

for n_hits in range(
    1,
    N_LAYERS + 1
):

    mask = (
        n_recorded_hits
        == n_hits
    )

    if np.sum(mask) == 0:
        continue

    mean_error = np.mean(
        np.degrees(
            absolute_errors[
                mask
            ]
        )
    )

    number_tracks = np.sum(
        mask
    )

    print(
        f"{n_hits} hits: "
        f"{mean_error:.6f} degrees "
        f"({number_tracks} tracks)"
    )

    hit_count_results[
        str(n_hits)
    ] = {

        "number_of_tracks":
            int(number_tracks),

        "mean_absolute_error_degrees":
            float(mean_error)
    }


# ============================================================
# Save model
# ============================================================

model.save(
    MODEL_FILE
)


# ============================================================
# Save results
# ============================================================

results = {

    "angular_mae_radians":
        float(
            mae_radians
        ),

    "angular_mae_degrees":
        float(
            mae_degrees
        ),

    "angular_rmse_radians":
        float(
            rmse_radians
        ),

    "angular_rmse_degrees":
        float(
            rmse_degrees
        ),

    "median_absolute_error_degrees":
        float(
            median_error_degrees
        ),

    "95_percentile_error_degrees":
        float(
            percentile_95_degrees
        ),

    "maximum_error_degrees":
        float(
            maximum_error_degrees
        ),

    "mean_output_vector_norm":
        float(
            predicted_norms.mean()
        ),

    "number_training_tracks":
        int(
            len(X_train)
        ),

    "number_validation_tracks":
        int(
            len(X_val)
        ),

    "number_test_tracks":
        int(
            len(X_test)
        ),

    "error_by_number_of_hits":
        hit_count_results
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
    f"{OUTPUT_DIR / 'track_parameter_predictions.csv'}"
)

print(
    f"Metrics saved to: "
    f"{RESULTS_FILE}"
)

print(
    f"Figures saved in: "
    f"{OUTPUT_DIR}"
)

print()
print("Done.")