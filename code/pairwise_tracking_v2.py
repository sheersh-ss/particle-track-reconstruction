import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.cluster import AgglomerativeClustering

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

HITS_FILE = "simulated_hits.csv"

OUTPUT_DIR = Path("pairwise_v2_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "pairwise_track_model_v2.keras"
RESULTS_FILE = OUTPUT_DIR / "results.json"

TRAIN_END = 7000
VAL_END = 8500
TEST_END = 10000

PAIRS_PER_EVENT_TRAIN = 100
PAIRS_PER_EVENT_VAL = 100
PAIRS_PER_EVENT_TEST = 100

SAME_TRACK_THRESHOLD = 0.5

BATCH_SIZE = 256
MAX_EPOCHS = 50


# ============================================================
# Load data
# ============================================================

print("Loading data...")

hits_df = pd.read_csv(HITS_FILE)

print(f"Total hits: {len(hits_df)}")
print(f"Total events: {hits_df['event_id'].nunique()}")


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
print("Event-level dataset split")
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
# Pair features
# ============================================================

def wrapped_angle_difference(phi1, phi2):
    """
    Return wrapped angular difference in [-pi, pi].
    """

    delta = phi2 - phi1

    return np.arctan2(
        np.sin(delta),
        np.cos(delta)
    )


def make_pair_features(hit1, hit2):
    """
    Generate geometric features for a pair of hits.
    """

    x1 = hit1["x"]
    y1 = hit1["y"]
    r1 = hit1["radius"]
    phi1 = hit1["phi"]
    layer1 = hit1["layer"]

    x2 = hit2["x"]
    y2 = hit2["y"]
    r2 = hit2["radius"]
    phi2 = hit2["phi"]
    layer2 = hit2["layer"]

    dx = x2 - x1
    dy = y2 - y1

    dr = r2 - r1

    dphi = wrapped_angle_difference(
        phi1,
        phi2
    )

    distance = np.sqrt(
        dx**2 + dy**2
    )

    layer_difference = (
        layer2 - layer1
    )

    features = [
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
        layer_difference
    ]

    return features


# ============================================================
# Pair generation
# ============================================================

def generate_pairs(
    data,
    pairs_per_event=100
):
    """
    Generate balanced positive and negative hit pairs.

    Positive:
        both hits belong to same true track

    Negative:
        hits belong to different true tracks
    """

    X = []
    y = []

    event_ids = (
        data["event_id"]
        .unique()
    )

    print(
        f"Generating pairs for "
        f"{len(event_ids)} events..."
    )

    for counter, event_id in enumerate(event_ids):

        event_hits = data[
            data["event_id"] == event_id
        ].reset_index(drop=True)

        positive_pairs = []
        negative_pairs = []

        # ------------------------------------------
        # Positive pairs
        # ------------------------------------------

        for track_id in event_hits[
            "track_id"
        ].unique():

            track_hits = event_hits[
                event_hits["track_id"]
                == track_id
            ].reset_index(drop=True)

            n_hits = len(track_hits)

            for i in range(n_hits):
                for j in range(i + 1, n_hits):

                    # Hits in same detector layer
                    # cannot be distinct hits from same ideal track.
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

        # ------------------------------------------
        # Negative pairs
        # ------------------------------------------

        desired_negative = len(
            positive_pairs
        )

        attempts = 0
        max_attempts = 20000

        while (
            len(negative_pairs)
            < desired_negative
            and attempts < max_attempts
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
                    (hit1, hit2)
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
            (counter + 1) % 500
            == 0
        ):

            print(
                f"Processed "
                f"{counter + 1}/"
                f"{len(event_ids)} events"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    return X, y


# ============================================================
# Generate train / val / test pair datasets
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
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
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


# Save scaler values
np.savez(
    OUTPUT_DIR / "scaler.npz",
    mean=scaler.mean_,
    scale=scaler.scale_
)


# ============================================================
# Build model
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

    Dropout(0.2),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

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
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# Training
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
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

test_probs = model.predict(
    X_test_scaled,
    verbose=0
).flatten()

test_preds = (
    test_probs
    >= SAME_TRACK_THRESHOLD
).astype(int)


pair_accuracy = accuracy_score(
    y_test,
    test_preds
)

pair_precision = precision_score(
    y_test,
    test_preds
)

pair_recall = recall_score(
    y_test,
    test_preds
)

pair_f1 = f1_score(
    y_test,
    test_preds
)

cm = confusion_matrix(
    y_test,
    test_preds
)


print()
print("==============================")
print("Pair-level test results")
print("==============================")

print(
    f"Accuracy:  {pair_accuracy:.6f}"
)

print(
    f"Precision: {pair_precision:.6f}"
)

print(
    f"Recall:    {pair_recall:.6f}"
)

print(
    f"F1 score:  {pair_f1:.6f}"
)

print()
print("Confusion matrix:")
print(cm)


# ============================================================
# Training plots
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title("Pairwise classifier loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_loss.png",
    dpi=200
)

plt.close()


plt.figure(figsize=(8, 5))

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
plt.title("Pairwise classifier accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "training_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# Confusion matrix plot
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    cm
)

plt.xticks(
    [0, 1],
    ["Different", "Same"]
)

plt.yticks(
    [0, 1],
    ["Different", "Same"]
)

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Pairwise classification confusion matrix")

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# Pair probability matrix
# ============================================================

def predict_pair_matrix(
    event_hits,
    model,
    scaler
):
    """
    Predict same-track probability for all pairs
    in one event.
    """

    event_hits = (
        event_hits
        .reset_index(drop=True)
    )

    n_hits = len(
        event_hits
    )

    probability_matrix = np.eye(
        n_hits,
        dtype=float
    )

    pair_features = []
    pair_indices = []

    for i in range(n_hits):

        for j in range(
            i + 1,
            n_hits
        ):

            pair_features.append(
                make_pair_features(
                    event_hits.iloc[i],
                    event_hits.iloc[j]
                )
            )

            pair_indices.append(
                (i, j)
            )

    if len(
        pair_features
    ) == 0:

        return probability_matrix

    pair_features = np.asarray(
        pair_features,
        dtype=np.float32
    )

    pair_features_scaled = (
        scaler.transform(
            pair_features
        )
    )

    predictions = model.predict(
        pair_features_scaled,
        verbose=0
    ).flatten()

    for probability, pair in zip(
        predictions,
        pair_indices
    ):

        i, j = pair

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
# Cluster event hits
# ============================================================

def cluster_event_hits(
    event_hits,
    model,
    scaler,
    n_tracks
):

    similarity = predict_pair_matrix(
        event_hits,
        model,
        scaler
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

    labels = clustering.fit_predict(
        distance
    )

    return labels


# ============================================================
# Permutation-invariant hit accuracy
# ============================================================

def permutation_invariant_accuracy(
    true_labels,
    predicted_labels
):
    """
    Map each predicted cluster to the most common
    true track label in that cluster.

    Returns:
        number correct,
        total hits,
        mapped labels
    """

    true_labels = np.asarray(
        true_labels
    )

    predicted_labels = np.asarray(
        predicted_labels
    )

    mapped = np.empty_like(
        predicted_labels
    )

    for cluster in np.unique(
        predicted_labels
    ):

        mask = (
            predicted_labels
            == cluster
        )

        true_values = (
            true_labels[mask]
        )

        values, counts = np.unique(
            true_values,
            return_counts=True
        )

        majority = values[
            np.argmax(counts)
        ]

        mapped[mask] = majority

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
# Full event-level test evaluation
# ============================================================

def evaluate_full_test_set(
    test_hits,
    model,
    scaler
):

    event_ids = (
        test_hits["event_id"]
        .unique()
    )

    total_correct = 0
    total_hits = 0

    per_event_accuracies = []

    print()
    print(
        f"Evaluating "
        f"{len(event_ids)} test events..."
    )

    for counter, event_id in enumerate(
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

        predicted = cluster_event_hits(
            event,
            model,
            scaler,
            n_tracks
        )

        correct, n_hits, mapped = (
            permutation_invariant_accuracy(
                true_labels,
                predicted
            )
        )

        total_correct += correct
        total_hits += n_hits

        per_event_accuracies.append(
            correct / n_hits
        )

        if (
            (counter + 1)
            % 100
            == 0
        ):

            print(
                f"Evaluated "
                f"{counter + 1}/"
                f"{len(event_ids)} events"
            )

    global_accuracy = (
        total_correct
        / total_hits
    )

    mean_event_accuracy = np.mean(
        per_event_accuracies
    )

    return (
        global_accuracy,
        mean_event_accuracy,
        np.asarray(
            per_event_accuracies
        )
    )


global_hit_accuracy, mean_event_accuracy, event_accuracies = (
    evaluate_full_test_set(
        test_hits,
        model,
        scaler
    )
)


print()
print("==============================")
print("Full event-level test results")
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
    f"{event_accuracies.min():.6f}"
)

print(
    f"Maximum event accuracy: "
    f"{event_accuracies.max():.6f}"
)


# ============================================================
# Distribution of event accuracies
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    event_accuracies,
    bins=30
)

plt.xlabel(
    "Hit assignment accuracy per event"
)

plt.ylabel(
    "Number of test events"
)

plt.title(
    "Distribution of event-level tracking accuracy"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "event_accuracy_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# Example event plots
# ============================================================

def plot_event_comparison(
    event_id,
    event,
    predicted_labels
):
    """
    Save one true assignment plot and one predicted plot.
    """

    # ------------------------------------------
    # True labels
    # ------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for track_id in sorted(
        event["track_id"].unique()
    ):

        subset = event[
            event["track_id"]
            == track_id
        ]

        ax.scatter(
            subset["x"],
            subset["y"],
            label=str(track_id)
        )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Event {event_id}: "
        f"true assignments"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"event_{event_id}_true.png",
        dpi=200
    )

    plt.close()


    # ------------------------------------------
    # Predicted clusters
    # ------------------------------------------

    event_copy = event.copy()

    event_copy[
        "predicted_cluster"
    ] = predicted_labels

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for cluster in sorted(
        event_copy[
            "predicted_cluster"
        ].unique()
    ):

        subset = event_copy[
            event_copy[
                "predicted_cluster"
            ] == cluster
        ]

        ax.scatter(
            subset["x"],
            subset["y"],
            label=str(cluster)
        )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Event {event_id}: "
        f"predicted assignments"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"event_{event_id}_predicted.png",
        dpi=200
    )

    plt.close()


example_event_ids = [
    8500,
    8501,
    8502
]

for event_id in example_event_ids:

    event = test_hits[
        test_hits["event_id"]
        == event_id
    ].reset_index(drop=True)

    n_tracks = (
        event["track_id"]
        .nunique()
    )

    predicted = cluster_event_hits(
        event,
        model,
        scaler,
        n_tracks
    )

    plot_event_comparison(
        event_id,
        event,
        predicted
    )


# ============================================================
# Save final model
# ============================================================

model.save(
    MODEL_FILE
)


# ============================================================
# Save metrics
# ============================================================

results = {

    "pair_accuracy":
        float(pair_accuracy),

    "pair_precision":
        float(pair_precision),

    "pair_recall":
        float(pair_recall),

    "pair_f1":
        float(pair_f1),

    "global_hit_assignment_accuracy":
        float(global_hit_accuracy),

    "mean_event_accuracy":
        float(mean_event_accuracy),

    "minimum_event_accuracy":
        float(
            event_accuracies.min()
        ),

    "maximum_event_accuracy":
        float(
            event_accuracies.max()
        ),

    "training_events":
        int(
            train_hits[
                "event_id"
            ].nunique()
        ),

    "validation_events":
        int(
            val_hits[
                "event_id"
            ].nunique()
        ),

    "test_events":
        int(
            test_hits[
                "event_id"
            ].nunique()
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
    f"Results saved to: "
    f"{RESULTS_FILE}"
)

print(
    f"Figures saved in: "
    f"{OUTPUT_DIR}"
)