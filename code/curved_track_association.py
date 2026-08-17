import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

DATA_FILE = "hits_FINAL.csv"

OUTPUT_DIR = Path("curved_tracking_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "curved_pairwise_model.keras"
SCALER_FILE = OUTPUT_DIR / "curved_pairwise_scaler.npz"
RESULTS_FILE = OUTPUT_DIR / "curved_tracking_results.json"

# Event-level split
TRAIN_END = 7000
VAL_END = 8500
TEST_END = 10000

PAIRS_PER_EVENT_TRAIN = 120
PAIRS_PER_EVENT_VAL = 120
PAIRS_PER_EVENT_TEST = 120

BATCH_SIZE = 256
MAX_EPOCHS = 60

PAIR_THRESHOLD = 0.5


# ============================================================
# Load dataset
# ============================================================

print("Loading curved-track dataset...")

hits_df = pd.read_csv(DATA_FILE)

print()
print("==============================")
print("Dataset information")
print("==============================")

print("Rows:", len(hits_df))
print("Events:", hits_df["event_id"].nunique())
print("Columns:", list(hits_df.columns))

tracks_per_event = (
    hits_df.groupby("event_id")["track_id"]
    .nunique()
)

print(
    "Minimum tracks/event:",
    tracks_per_event.min()
)

print(
    "Maximum tracks/event:",
    tracks_per_event.max()
)


# ============================================================
# Derived hit quantities
# ============================================================

hits_df["radius"] = np.sqrt(
    hits_df["x"] ** 2
    +
    hits_df["y"] ** 2
)

hits_df["phi_hit"] = np.arctan2(
    hits_df["y"],
    hits_df["x"]
)


# ============================================================
# Event-level split
# ============================================================

train_hits = hits_df[
    hits_df["event_id"] < TRAIN_END
].copy()

val_hits = hits_df[
    (hits_df["event_id"] >= TRAIN_END)
    &
    (hits_df["event_id"] < VAL_END)
].copy()

test_hits = hits_df[
    (hits_df["event_id"] >= VAL_END)
    &
    (hits_df["event_id"] < TEST_END)
].copy()


print()
print("==============================")
print("Event-level split")
print("==============================")

print(
    "Training events:",
    train_hits["event_id"].nunique()
)

print(
    "Validation events:",
    val_hits["event_id"].nunique()
)

print(
    "Test events:",
    test_hits["event_id"].nunique()
)


# ============================================================
# Angle helper
# ============================================================

def wrapped_angle_difference(phi1, phi2):

    delta = phi2 - phi1

    return np.arctan2(
        np.sin(delta),
        np.cos(delta)
    )


# ============================================================
# Pairwise features
# ============================================================

def make_pair_features(hit1, hit2):
    """
    Features for deciding whether two curved-track hits
    belong to the same physical track.
    """

    x1 = hit1["x"]
    y1 = hit1["y"]
    r1 = hit1["radius"]
    phi1 = hit1["phi_hit"]
    layer1 = hit1["layer"]

    x2 = hit2["x"]
    y2 = hit2["y"]
    r2 = hit2["radius"]
    phi2 = hit2["phi_hit"]
    layer2 = hit2["layer"]

    dx = x2 - x1
    dy = y2 - y1

    dr = r2 - r1

    dphi = wrapped_angle_difference(
        phi1,
        phi2
    )

    layer_difference = (
        layer2 - layer1
    )

    distance = np.sqrt(
        dx**2 + dy**2
    )

    # Useful approximate curvature-related quantity
    if abs(dr) > 1e-8:

        angular_slope = (
            dphi / dr
        )

    else:

        angular_slope = 0.0

    # Chord angle between the two hit positions
    chord_angle = np.arctan2(
        dy,
        dx
    )

    # Relative orientation of chord
    chord_relative_to_hit1 = (
        wrapped_angle_difference(
            phi1,
            chord_angle
        )
    )

    chord_relative_to_hit2 = (
        wrapped_angle_difference(
            phi2,
            chord_angle
        )
    )

    return [
        x1,
        y1,
        r1,
        phi1,
        layer1,

        x2,
        y2,
        r2,
        phi2,
        layer2,

        dx,
        dy,
        dr,

        dphi,
        abs(dphi),

        distance,

        layer_difference,

        angular_slope,

        chord_angle,

        chord_relative_to_hit1,
        chord_relative_to_hit2
    ]


# ============================================================
# Generate balanced hit pairs
# ============================================================

def generate_pairs(
    data,
    pairs_per_event=120
):

    X = []
    y = []

    event_ids = data[
        "event_id"
    ].unique()

    print(
        f"Generating pairs from "
        f"{len(event_ids)} events..."
    )

    for count, event_id in enumerate(
        event_ids
    ):

        event_hits = data[
            data["event_id"] == event_id
        ].reset_index(drop=True)

        positive_pairs = []
        negative_pairs = []

        # ----------------------------------------------------
        # Positive same-track pairs
        # ----------------------------------------------------

        for track_id in event_hits[
            "track_id"
        ].unique():

            track_hits = event_hits[
                event_hits["track_id"]
                == track_id
            ].reset_index(drop=True)

            n = len(track_hits)

            for i in range(n):

                for j in range(
                    i + 1,
                    n
                ):

                    if (
                        track_hits.iloc[i]["layer"]
                        ==
                        track_hits.iloc[j]["layer"]
                    ):
                        continue

                    positive_pairs.append(
                        (
                            track_hits.iloc[i],
                            track_hits.iloc[j]
                        )
                    )

        # ----------------------------------------------------
        # Negative different-track pairs
        # ----------------------------------------------------

        target_negatives = len(
            positive_pairs
        )

        attempts = 0

        while (
            len(negative_pairs)
            < target_negatives
            and attempts < 30000
        ):

            i, j = rng.choice(
                len(event_hits),
                size=2,
                replace=False
            )

            hit1 = event_hits.iloc[i]
            hit2 = event_hits.iloc[j]

            if (
                hit1["track_id"]
                != hit2["track_id"]
            ):

                negative_pairs.append(
                    (
                        hit1,
                        hit2
                    )
                )

            attempts += 1

        rng.shuffle(
            positive_pairs
        )

        rng.shuffle(
            negative_pairs
        )

        n_each = (
            pairs_per_event // 2
        )

        positive_pairs = (
            positive_pairs[:n_each]
        )

        negative_pairs = (
            negative_pairs[:n_each]
        )

        for hit1, hit2 in positive_pairs:

            X.append(
                make_pair_features(
                    hit1,
                    hit2
                )
            )

            y.append(1)

        for hit1, hit2 in negative_pairs:

            X.append(
                make_pair_features(
                    hit1,
                    hit2
                )
            )

            y.append(0)

        if (
            (count + 1)
            % 500
            == 0
        ):

            print(
                f"Processed "
                f"{count + 1}/"
                f"{len(event_ids)} events"
            )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),

        np.asarray(
            y,
            dtype=np.float32
        )
    )


# ============================================================
# Build train / validation / test datasets
# ============================================================

print()
print("Generating TRAIN pairs...")

X_train, y_train = generate_pairs(
    train_hits,
    PAIRS_PER_EVENT_TRAIN
)


print()
print("Generating VALIDATION pairs...")

X_val, y_val = generate_pairs(
    val_hits,
    PAIRS_PER_EVENT_VAL
)


print()
print("Generating TEST pairs...")

X_test, y_test = generate_pairs(
    test_hits,
    PAIRS_PER_EVENT_TEST
)


print()
print("==============================")
print("Pair dataset sizes")
print("==============================")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# Standardization
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


np.savez(
    SCALER_FILE,
    mean=scaler.mean_,
    scale=scaler.scale_
)


# ============================================================
# Neural network
# ============================================================

model = Sequential([
    Input(
        shape=(
            X_train_scaled.shape[1],
        )
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.20
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
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=Adam(
        learning_rate=1e-3
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# Training
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=6,
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
# Pair-level evaluation
# ============================================================

test_probabilities = (
    model.predict(
        X_test_scaled,
        verbose=0
    )
    .flatten()
)


test_predictions = (
    test_probabilities
    >= PAIR_THRESHOLD
).astype(int)


pair_accuracy = accuracy_score(
    y_test,
    test_predictions
)

pair_precision = precision_score(
    y_test,
    test_predictions
)

pair_recall = recall_score(
    y_test,
    test_predictions
)

pair_f1 = f1_score(
    y_test,
    test_predictions
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


print()
print("==============================")
print("Curved pair classification")
print("==============================")

print(
    f"Accuracy:  "
    f"{pair_accuracy:.6f}"
)

print(
    f"Precision: "
    f"{pair_precision:.6f}"
)

print(
    f"Recall:    "
    f"{pair_recall:.6f}"
)

print(
    f"F1 score:  "
    f"{pair_f1:.6f}"
)

print()
print("Confusion matrix:")
print(cm)


# ============================================================
# Training plots
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

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Curved-track pairwise training loss"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "training_loss.png",
    dpi=200
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Curved-track pairwise accuracy"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "training_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# Predict full pair matrix for one event
# ============================================================

def predict_pair_matrix(
    event_hits
):

    event_hits = (
        event_hits
        .reset_index(drop=True)
    )

    n_hits = len(
        event_hits
    )

    probability_matrix = np.eye(
        n_hits,
        dtype=np.float32
    )

    features = []
    indices = []

    for i in range(
        n_hits
    ):

        for j in range(
            i + 1,
            n_hits
        ):

            features.append(
                make_pair_features(
                    event_hits.iloc[i],
                    event_hits.iloc[j]
                )
            )

            indices.append(
                (
                    i,
                    j
                )
            )

    if len(features) == 0:

        return probability_matrix

    features = np.asarray(
        features,
        dtype=np.float32
    )

    features_scaled = scaler.transform(
        features
    )

    probabilities = (
        model.predict(
            features_scaled,
            batch_size=4096,
            verbose=0
        )
        .flatten()
    )

    for probability, (
        i,
        j
    ) in zip(
        probabilities,
        indices
    ):

        probability_matrix[
            i,
            j
        ] = probability

        probability_matrix[
            j,
            i
        ] = probability

    return probability_matrix


# ============================================================
# Cluster hits into tracks
# ============================================================

def cluster_event(
    event_hits,
    n_tracks
):

    similarity = predict_pair_matrix(
        event_hits
    )

    distance = (
        1.0 - similarity
    )

    np.fill_diagonal(
        distance,
        0.0
    )

    try:

        clustering = (
            AgglomerativeClustering(
                n_clusters=n_tracks,
                metric="precomputed",
                linkage="average"
            )
        )

    except TypeError:

        clustering = (
            AgglomerativeClustering(
                n_clusters=n_tracks,
                affinity="precomputed",
                linkage="average"
            )
        )

    return clustering.fit_predict(
        distance
    )


# ============================================================
# Permutation-invariant track accuracy
# ============================================================

def permutation_invariant_accuracy(
    true_labels,
    predicted_labels
):

    true_labels = np.asarray(
        true_labels
    )

    predicted_labels = np.asarray(
        predicted_labels
    )

    mapped = np.empty_like(
        predicted_labels
    )

    for predicted_cluster in np.unique(
        predicted_labels
    ):

        mask = (
            predicted_labels
            == predicted_cluster
        )

        true_tracks = (
            true_labels[
                mask
            ]
        )

        values, counts = np.unique(
            true_tracks,
            return_counts=True
        )

        majority = values[
            np.argmax(counts)
        ]

        mapped[
            mask
        ] = majority

    correct = np.sum(
        mapped == true_labels
    )

    total = len(
        true_labels
    )

    return (
        correct,
        total,
        mapped
    )


# ============================================================
# Evaluate full test-event set
# ============================================================

def evaluate_all_events():

    event_ids = test_hits[
        "event_id"
    ].unique()

    total_correct = 0
    total_hits = 0

    event_accuracies = []

    print()
    print(
        f"Evaluating "
        f"{len(event_ids)} curved test events..."
    )

    for count, event_id in enumerate(
        event_ids
    ):

        event = test_hits[
            test_hits["event_id"]
            == event_id
        ].reset_index(drop=True)

        true_labels = (
            event["track_id"]
            .to_numpy()
        )

        n_tracks = (
            event["track_id"]
            .nunique()
        )

        predicted_labels = (
            cluster_event(
                event,
                n_tracks
            )
        )

        correct, total, mapped = (
            permutation_invariant_accuracy(
                true_labels,
                predicted_labels
            )
        )

        total_correct += correct
        total_hits += total

        event_accuracies.append(
            correct / total
        )

        if (
            (count + 1)
            % 100
            == 0
        ):

            print(
                f"Evaluated "
                f"{count + 1}/"
                f"{len(event_ids)} events"
            )

    global_accuracy = (
        total_correct
        / total_hits
    )

    return (
        global_accuracy,
        np.asarray(
            event_accuracies
        )
    )


global_hit_accuracy, event_accuracies = (
    evaluate_all_events()
)


mean_event_accuracy = (
    event_accuracies.mean()
)

minimum_event_accuracy = (
    event_accuracies.min()
)

maximum_event_accuracy = (
    event_accuracies.max()
)


print()
print("==============================")
print("Curved-track association results")
print("==============================")

print(
    f"Global hit assignment accuracy: "
    f"{global_hit_accuracy:.6f}"
)

print(
    f"Mean event accuracy: "
    f"{mean_event_accuracy:.6f}"
)

print(
    f"Minimum event accuracy: "
    f"{minimum_event_accuracy:.6f}"
)

print(
    f"Maximum event accuracy: "
    f"{maximum_event_accuracy:.6f}"
)


# ============================================================
# Accuracy distribution
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    event_accuracies,
    bins=30
)

plt.xlabel(
    "Hit assignment accuracy"
)

plt.ylabel(
    "Number of events"
)

plt.title(
    "Curved-track event accuracy distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "event_accuracy_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# Plot example events
# ============================================================

def plot_event(
    event_id,
    predicted_labels
):

    event = test_hits[
        test_hits["event_id"]
        == event_id
    ].reset_index(drop=True)

    # ------------------------------------------
    # True assignments
    # ------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for track_id in sorted(
        event[
            "track_id"
        ].unique()
    ):

        subset = event[
            event[
                "track_id"
            ] == track_id
        ]

        subset = subset.sort_values(
            "layer"
        )

        ax.plot(
            subset["x"],
            subset["y"],
            marker="o",
            markersize=4
        )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Event {event_id}: "
        f"true curved tracks"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"event_{event_id}_true.png",
        dpi=200
    )

    plt.close()


    # ------------------------------------------
    # Predicted assignments
    # ------------------------------------------

    event_copy = event.copy()

    event_copy[
        "predicted_cluster"
    ] = predicted_labels

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for cluster_id in sorted(
        event_copy[
            "predicted_cluster"
        ].unique()
    ):

        subset = event_copy[
            event_copy[
                "predicted_cluster"
            ] == cluster_id
        ]

        subset = subset.sort_values(
            "layer"
        )

        ax.plot(
            subset["x"],
            subset["y"],
            marker="o",
            markersize=4
        )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Event {event_id}: "
        f"predicted curved tracks"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"event_{event_id}_predicted.png",
        dpi=200
    )

    plt.close()


example_events = [
    8500,
    8501,
    8502
]


for event_id in example_events:

    event = test_hits[
        test_hits["event_id"]
        == event_id
    ].reset_index(drop=True)

    n_tracks = (
        event["track_id"]
        .nunique()
    )

    predicted_labels = (
        cluster_event(
            event,
            n_tracks
        )
    )

    plot_event(
        event_id,
        predicted_labels
    )


# ============================================================
# Save model
# ============================================================

model.save(
    MODEL_FILE
)


# ============================================================
# Save result metrics
# ============================================================

results = {

    "pair_accuracy":
        float(
            pair_accuracy
        ),

    "pair_precision":
        float(
            pair_precision
        ),

    "pair_recall":
        float(
            pair_recall
        ),

    "pair_f1":
        float(
            pair_f1
        ),

    "global_hit_assignment_accuracy":
        float(
            global_hit_accuracy
        ),

    "mean_event_accuracy":
        float(
            mean_event_accuracy
        ),

    "minimum_event_accuracy":
        float(
            minimum_event_accuracy
        ),

    "maximum_event_accuracy":
        float(
            maximum_event_accuracy
        )
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


print()
print("==============================")
print("Finished")
print("==============================")

print(
    f"Model saved to: "
    f"{MODEL_FILE}"
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