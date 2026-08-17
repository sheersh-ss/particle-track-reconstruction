import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import AgglomerativeClustering

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

HITS_FILE = "simulated_hits.csv"

MAX_TRAIN_EVENTS = 3000
PAIRS_PER_EVENT = 100

SAME_TRACK_THRESHOLD = 0.5


# ============================================================
# Load data
# ============================================================

print("Loading hit data...")

hits_df = pd.read_csv(HITS_FILE)

print(hits_df.head())
print()
print("Number of hits:", len(hits_df))
print("Number of events:", hits_df["event_id"].nunique())


# ============================================================
# Feature engineering
# ============================================================

def make_pair_features(hit1, hit2):
    """
    Create features describing the geometric relationship
    between two detector hits.
    """

    x1 = hit1["x"]
    y1 = hit1["y"]
    r1 = hit1["radius"]
    phi1 = hit1["phi"]

    x2 = hit2["x"]
    y2 = hit2["y"]
    r2 = hit2["radius"]
    phi2 = hit2["phi"]

    dx = x2 - x1
    dy = y2 - y1

    dr = r2 - r1

    # Angle difference
    dphi = phi2 - phi1

    # Wrap angle difference to [-pi, pi]
    dphi = np.arctan2(
        np.sin(dphi),
        np.cos(dphi)
    )

    distance = np.sqrt(
        dx**2 + dy**2
    )

    # For straight tracks from origin,
    # hits from the same track should have very similar phi.
    features = [
        x1,
        y1,
        r1,
        phi1,

        x2,
        y2,
        r2,
        phi2,

        dx,
        dy,
        dr,
        dphi,
        abs(dphi),
        distance
    ]

    return features


# ============================================================
# Generate balanced training pairs
# ============================================================

def generate_training_pairs(
    hits_df,
    max_events=3000,
    pairs_per_event=100
):
    """
    Generate balanced same-track / different-track hit pairs.
    """

    X = []
    y = []

    event_ids = hits_df["event_id"].unique()

    # Limit number of events to make training manageable
    if len(event_ids) > max_events:
        event_ids = np.random.choice(
            event_ids,
            size=max_events,
            replace=False
        )

    print(
        f"Generating training pairs from "
        f"{len(event_ids)} events..."
    )

    for count, event_id in enumerate(event_ids):

        event_hits = hits_df[
            hits_df["event_id"] == event_id
        ].reset_index(drop=True)

        if len(event_hits) < 2:
            continue

        # ----------------------------------------------------
        # Same-track pairs
        # ----------------------------------------------------

        same_pairs = []

        for track_id in event_hits["track_id"].unique():

            track_hits = event_hits[
                event_hits["track_id"] == track_id
            ].reset_index(drop=True)

            n = len(track_hits)

            for i in range(n):
                for j in range(i + 1, n):

                    # Do not pair a hit with another hit
                    # on the same detector layer
                    if (
                        track_hits.iloc[i]["layer"]
                        ==
                        track_hits.iloc[j]["layer"]
                    ):
                        continue

                    same_pairs.append(
                        (
                            track_hits.iloc[i],
                            track_hits.iloc[j]
                        )
                    )

        # ----------------------------------------------------
        # Different-track pairs
        # ----------------------------------------------------

        different_pairs = []

        attempts = 0

        while (
            len(different_pairs) < len(same_pairs)
            and attempts < 10000
        ):

            i, j = np.random.choice(
                len(event_hits),
                size=2,
                replace=False
            )

            hit1 = event_hits.iloc[i]
            hit2 = event_hits.iloc[j]

            if hit1["track_id"] != hit2["track_id"]:

                different_pairs.append(
                    (hit1, hit2)
                )

            attempts += 1

        # ----------------------------------------------------
        # Limit number of pairs per event
        # ----------------------------------------------------

        np.random.shuffle(same_pairs)
        np.random.shuffle(different_pairs)

        n_each = pairs_per_event // 2

        same_pairs = same_pairs[:n_each]
        different_pairs = different_pairs[:n_each]

        # Positive pairs
        for hit1, hit2 in same_pairs:

            X.append(
                make_pair_features(
                    hit1,
                    hit2
                )
            )

            y.append(1)

        # Negative pairs
        for hit1, hit2 in different_pairs:

            X.append(
                make_pair_features(
                    hit1,
                    hit2
                )
            )

            y.append(0)

        if (count + 1) % 500 == 0:

            print(
                f"Processed "
                f"{count + 1}/"
                f"{len(event_ids)} events"
            )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


# ============================================================
# Build training dataset
# ============================================================

X, y = generate_training_pairs(
    hits_df,
    max_events=MAX_TRAIN_EVENTS,
    pairs_per_event=PAIRS_PER_EVENT
)

print()
print("Training pair dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Fraction of same-track pairs:",
    np.mean(y)
)


# ============================================================
# Train / validation / test split
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp
)


# ============================================================
# Standardize features
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


# ============================================================
# Neural network
# ============================================================

model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train_scaled.shape[1],)
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
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# Train model
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

    epochs=50,
    batch_size=256,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# Evaluate pair classification
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print()
print("==============================")
print("Pair classification results")
print("==============================")

print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)


pred_prob = model.predict(
    X_test_scaled,
    verbose=0
).flatten()

pred_class = (
    pred_prob >= SAME_TRACK_THRESHOLD
).astype(int)


print()
print(
    classification_report(
        y_test.astype(int),
        pred_class
    )
)


# ============================================================
# Plot training history
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Training and validation loss")

plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and validation accuracy")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Build pairwise probability matrix for one event
# ============================================================

def predict_pair_matrix(
    event_hits,
    model,
    scaler
):
    """
    Return NxN matrix containing predicted probability
    that two hits belong to the same track.
    """

    event_hits = event_hits.reset_index(drop=True)

    n_hits = len(event_hits)

    probability_matrix = np.eye(
        n_hits,
        dtype=float
    )

    features = []
    pair_indices = []

    for i in range(n_hits):

        for j in range(i + 1, n_hits):

            feature = make_pair_features(
                event_hits.iloc[i],
                event_hits.iloc[j]
            )

            features.append(feature)

            pair_indices.append(
                (i, j)
            )

    if len(features) == 0:
        return probability_matrix

    features = np.array(
        features,
        dtype=np.float32
    )

    features_scaled = scaler.transform(
        features
    )

    predictions = model.predict(
        features_scaled,
        verbose=0
    ).flatten()

    for prediction, (i, j) in zip(
        predictions,
        pair_indices
    ):

        probability_matrix[i, j] = prediction
        probability_matrix[j, i] = prediction

    return probability_matrix


# ============================================================
# Cluster hits
# ============================================================

def cluster_event_hits(
    event_hits,
    model,
    scaler,
    n_tracks
):
    """
    Cluster hits into particle tracks using predicted
    same-track probabilities.
    """

    probability_matrix = predict_pair_matrix(
        event_hits,
        model,
        scaler
    )

    # Convert similarity to distance
    distance_matrix = (
        1.0 - probability_matrix
    )

    np.fill_diagonal(
        distance_matrix,
        0.0
    )

    # Compatibility for different sklearn versions
    try:

        clustering = AgglomerativeClustering(
            n_clusters=n_tracks,
            metric="precomputed",
            linkage="average"
        )

    except TypeError:

        clustering = AgglomerativeClustering(
            n_clusters=n_tracks,
            affinity="precomputed",
            linkage="average"
        )

    predicted_clusters = clustering.fit_predict(
        distance_matrix
    )

    return predicted_clusters


# ============================================================
# Permutation-invariant evaluation
# ============================================================

def clustering_hit_accuracy(
    true_labels,
    predicted_labels
):
    """
    Compute hit accuracy while ignoring arbitrary cluster IDs.

    Each predicted cluster is mapped to the true track that
    occurs most frequently inside that cluster.
    """

    true_labels = np.asarray(
        true_labels
    )

    predicted_labels = np.asarray(
        predicted_labels
    )

    mapped_predictions = np.empty_like(
        predicted_labels
    )

    for cluster_id in np.unique(
        predicted_labels
    ):

        mask = (
            predicted_labels == cluster_id
        )

        true_in_cluster = (
            true_labels[mask]
        )

        values, counts = np.unique(
            true_in_cluster,
            return_counts=True
        )

        majority_true_track = values[
            np.argmax(counts)
        ]

        mapped_predictions[mask] = (
            majority_true_track
        )

    accuracy = np.mean(
        mapped_predictions == true_labels
    )

    return accuracy, mapped_predictions


# ============================================================
# Evaluate one event
# ============================================================

def evaluate_event(
    event_id,
    hits_df,
    model,
    scaler
):

    event_hits = hits_df[
        hits_df["event_id"] == event_id
    ].copy()

    event_hits = event_hits.reset_index(
        drop=True
    )

    true_labels = (
        event_hits["track_id"]
        .to_numpy()
    )

    n_true_tracks = (
        event_hits["track_id"]
        .nunique()
    )

    predicted_labels = cluster_event_hits(
        event_hits,
        model,
        scaler,
        n_tracks=n_true_tracks
    )

    accuracy, mapped_predictions = (
        clustering_hit_accuracy(
            true_labels,
            predicted_labels
        )
    )

    print()
    print(
        f"Event {event_id}:"
    )

    print(
        f"Number of hits: "
        f"{len(event_hits)}"
    )

    print(
        f"Number of true tracks: "
        f"{n_true_tracks}"
    )

    print(
        f"Hit assignment accuracy: "
        f"{accuracy:.4f}"
    )

    event_hits[
        "predicted_cluster"
    ] = predicted_labels

    event_hits[
        "mapped_prediction"
    ] = mapped_predictions

    return event_hits, accuracy


# ============================================================
# Visualize true and predicted assignments
# ============================================================

def plot_true_assignments(
    event_hits,
    event_id
):

    plt.figure(
        figsize=(8, 8)
    )

    for track_id in sorted(
        event_hits["track_id"].unique()
    ):

        track_hits = event_hits[
            event_hits["track_id"]
            == track_id
        ]

        plt.scatter(
            track_hits["x"],
            track_hits["y"],
            label=f"Track {track_id}"
        )

    for radius in [
        2, 4, 6, 8, 10
    ]:

        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            alpha=0.2
        )

        plt.gca().add_patch(
            circle
        )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Event {event_id}: "
        f"true track assignments"
    )

    plt.axis("equal")

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)

    plt.tight_layout()
    plt.show()


def plot_predicted_assignments(
    event_hits,
    event_id
):

    plt.figure(
        figsize=(8, 8)
    )

    for cluster_id in sorted(
        event_hits[
            "predicted_cluster"
        ].unique()
    ):

        cluster_hits = event_hits[
            event_hits[
                "predicted_cluster"
            ] == cluster_id
        ]

        plt.scatter(
            cluster_hits["x"],
            cluster_hits["y"],
            label=f"Cluster {cluster_id}"
        )

    for radius in [
        2, 4, 6, 8, 10
    ]:

        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            alpha=0.2
        )

        plt.gca().add_patch(
            circle
        )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Event {event_id}: "
        f"predicted track assignments"
    )

    plt.axis("equal")

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)

    plt.tight_layout()
    plt.show()


# ============================================================
# Test on several events
# ============================================================

test_event_ids = [
    9000,
    9001,
    9002,
    9003,
    9004
]

event_accuracies = []

for event_id in test_event_ids:

    result_df, accuracy = evaluate_event(
        event_id,
        hits_df,
        model,
        scaler
    )

    event_accuracies.append(
        accuracy
    )

    if event_id == test_event_ids[0]:

        plot_true_assignments(
            result_df,
            event_id
        )

        plot_predicted_assignments(
            result_df,
            event_id
        )


print()
print("==============================")
print("Event clustering results")
print("==============================")

print(
    f"Mean hit assignment accuracy: "
    f"{np.mean(event_accuracies):.4f}"
)


# ============================================================
# Save model
# ============================================================

model.save(
    "pairwise_track_model.keras"
)

print()
print(
    "Saved model as "
    "pairwise_track_model.keras"
)

print("Done.")