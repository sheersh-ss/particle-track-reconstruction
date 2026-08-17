import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from tensorflow.keras.models import load_model


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 123

rng = np.random.default_rng(RANDOM_SEED)

DETECTOR_RADII = np.array(
    [2, 4, 6, 8, 10],
    dtype=float
)

HIT_EFFICIENCY = 0.95
RELATIVE_NOISE_STD = 0.001

N_TRACKS = 50

# Start with 100 events.
# You can increase later if runtime is acceptable.
N_EVENTS = 100

MODEL_FILE = (
    "pairwise_v2_results/"
    "pairwise_track_model_v2.keras"
)

SCALER_FILE = (
    "pairwise_v2_results/"
    "scaler.npz"
)

OUTPUT_DIR = Path(
    "scaling_50_tracks_results"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Load trained model and scaler
# ============================================================

print("Loading trained model...")

model = load_model(
    MODEL_FILE
)

scaler_data = np.load(
    SCALER_FILE
)

scaler = StandardScaler()

scaler.mean_ = scaler_data["mean"]
scaler.scale_ = scaler_data["scale"]

# Needed by sklearn StandardScaler
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = len(
    scaler.mean_
)


print(
    "Model and scaler loaded."
)


# ============================================================
# Generate 50-track events
# ============================================================

def generate_event(
    event_id,
    n_tracks=50
):
    """
    Generate one realistic straight-track event
    with detector inefficiency and Gaussian noise.
    """

    hits = []

    phis = rng.uniform(
        -np.pi,
        np.pi,
        size=n_tracks
    )

    for track_id, phi in enumerate(
        phis
    ):

        for layer_id, radius in enumerate(
            DETECTOR_RADII
        ):

            # 95% detector efficiency
            if rng.random() > HIT_EFFICIENCY:
                continue

            true_x = (
                radius
                * np.cos(phi)
            )

            true_y = (
                radius
                * np.sin(phi)
            )

            sigma = (
                RELATIVE_NOISE_STD
                * radius
            )

            x = (
                true_x
                + rng.normal(
                    0,
                    sigma
                )
            )

            y = (
                true_y
                + rng.normal(
                    0,
                    sigma
                )
            )

            measured_phi = np.arctan2(
                y,
                x
            )

            hits.append({
                "event_id":
                    event_id,

                "track_id":
                    track_id,

                "layer":
                    layer_id,

                "radius":
                    radius,

                "x":
                    x,

                "y":
                    y,

                "phi":
                    measured_phi,

                "true_phi":
                    phi
            })

    return pd.DataFrame(
        hits
    )


# ============================================================
# Pair feature engineering
# ============================================================

def wrapped_angle_difference(
    phi1,
    phi2
):

    delta = (
        phi2 - phi1
    )

    return np.arctan2(
        np.sin(delta),
        np.cos(delta)
    )


def make_pair_features(
    hit1,
    hit2
):
    """
    Must match exactly the feature definition used
    during training of pairwise_tracking_v2.py.
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

    dphi = (
        wrapped_angle_difference(
            phi1,
            phi2
        )
    )

    distance = np.sqrt(
        dx**2 + dy**2
    )

    layer_difference = (
        layer2 - layer1
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
        layer_difference
    ]


# ============================================================
# Pairwise probability matrix
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
    pair_indices = []

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

            pair_indices.append(
                (i, j)
            )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    features_scaled = (
        scaler.transform(
            features
        )
    )

    predictions = (
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
        predictions,
        pair_indices
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
# Cluster hits
# ============================================================

def cluster_event(
    event_hits,
    n_tracks
):

    probability_matrix = (
        predict_pair_matrix(
            event_hits
        )
    )

    distance_matrix = (
        1.0
        - probability_matrix
    )

    np.fill_diagonal(
        distance_matrix,
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

    labels = (
        clustering.fit_predict(
            distance_matrix
        )
    )

    return labels


# ============================================================
# Permutation-invariant accuracy
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

    for cluster in np.unique(
        predicted_labels
    ):

        mask = (
            predicted_labels
            == cluster
        )

        true_values = (
            true_labels[
                mask
            ]
        )

        values, counts = np.unique(
            true_values,
            return_counts=True
        )

        majority = values[
            np.argmax(
                counts
            )
        ]

        mapped[
            mask
        ] = majority

    correct = np.sum(
        mapped
        == true_labels
    )

    total = len(
        true_labels
    )

    accuracy = (
        correct
        / total
    )

    return (
        accuracy,
        mapped
    )


# ============================================================
# Evaluate all 50-track events
# ============================================================

event_results = []

total_correct = 0
total_hits = 0

print()
print("==============================")
print("50-track scaling experiment")
print("==============================")

print(
    f"Number of events: "
    f"{N_EVENTS}"
)

print(
    f"Tracks per event: "
    f"{N_TRACKS}"
)


for event_id in range(
    N_EVENTS
):

    event_hits = generate_event(
        event_id=event_id,
        n_tracks=N_TRACKS
    )

    true_labels = (
        event_hits[
            "track_id"
        ].to_numpy()
    )

    start_time = (
        time.perf_counter()
    )

    predicted_labels = (
        cluster_event(
            event_hits,
            N_TRACKS
        )
    )

    end_time = (
        time.perf_counter()
    )

    runtime = (
        end_time
        - start_time
    )

    accuracy, mapped = (
        permutation_invariant_accuracy(
            true_labels,
            predicted_labels
        )
    )

    n_hits = len(
        event_hits
    )

    correct = int(
        accuracy
        * n_hits
    )

    total_correct += correct
    total_hits += n_hits

    event_results.append({
        "event_id":
            event_id,

        "n_tracks":
            N_TRACKS,

        "n_hits":
            n_hits,

        "accuracy":
            accuracy,

        "runtime_seconds":
            runtime
    })


    print(
        f"Event {event_id + 1:3d}/"
        f"{N_EVENTS}: "
        f"{n_hits:3d} hits, "
        f"accuracy = "
        f"{accuracy:.4f}, "
        f"time = "
        f"{runtime:.3f} s"
    )


# ============================================================
# Results dataframe
# ============================================================

results_df = pd.DataFrame(
    event_results
)

global_accuracy = (
    total_correct
    / total_hits
)

mean_accuracy = (
    results_df[
        "accuracy"
    ].mean()
)

minimum_accuracy = (
    results_df[
        "accuracy"
    ].min()
)

maximum_accuracy = (
    results_df[
        "accuracy"
    ].max()
)

mean_runtime = (
    results_df[
        "runtime_seconds"
    ].mean()
)

median_runtime = (
    results_df[
        "runtime_seconds"
    ].median()
)


# ============================================================
# Print final results
# ============================================================

print()
print("==============================")
print("50-track results")
print("==============================")

print(
    f"Global hit assignment accuracy: "
    f"{global_accuracy:.6f}"
)

print(
    f"Mean event accuracy: "
    f"{mean_accuracy:.6f}"
)

print(
    f"Minimum event accuracy: "
    f"{minimum_accuracy:.6f}"
)

print(
    f"Maximum event accuracy: "
    f"{maximum_accuracy:.6f}"
)

print()

print(
    f"Mean runtime per event: "
    f"{mean_runtime:.4f} seconds"
)

print(
    f"Median runtime per event: "
    f"{median_runtime:.4f} seconds"
)


# ============================================================
# Save numeric results
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "50_track_event_results.csv",
    index=False
)


summary_df = pd.DataFrame([
    {
        "n_tracks":
            N_TRACKS,

        "n_events":
            N_EVENTS,

        "global_accuracy":
            global_accuracy,

        "mean_accuracy":
            mean_accuracy,

        "minimum_accuracy":
            minimum_accuracy,

        "maximum_accuracy":
            maximum_accuracy,

        "mean_runtime_seconds":
            mean_runtime,

        "median_runtime_seconds":
            median_runtime
    }
])


summary_df.to_csv(
    OUTPUT_DIR /
    "50_track_summary.csv",
    index=False
)


# ============================================================
# Accuracy histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    results_df[
        "accuracy"
    ],
    bins=20
)

plt.xlabel(
    "Hit assignment accuracy"
)

plt.ylabel(
    "Number of events"
)

plt.title(
    "Tracking accuracy for 50-track events"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "50_track_accuracy_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# Runtime histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    results_df[
        "runtime_seconds"
    ],
    bins=20
)

plt.xlabel(
    "Runtime per event [s]"
)

plt.ylabel(
    "Number of events"
)

plt.title(
    "Runtime for 50-track events"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "50_track_runtime_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# Accuracy vs number of hits
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    results_df[
        "n_hits"
    ],
    results_df[
        "accuracy"
    ]
)

plt.xlabel(
    "Number of recorded hits"
)

plt.ylabel(
    "Hit assignment accuracy"
)

plt.title(
    "50-track accuracy versus event size"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "accuracy_vs_number_of_hits.png",
    dpi=200
)

plt.close()


# ============================================================
# Runtime vs number of hits
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    results_df[
        "n_hits"
    ],
    results_df[
        "runtime_seconds"
    ]
)

plt.xlabel(
    "Number of recorded hits"
)

plt.ylabel(
    "Runtime per event [s]"
)

plt.title(
    "50-track runtime versus event size"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "runtime_vs_number_of_hits.png",
    dpi=200
)

plt.close()


# ============================================================
# Save example event
# ============================================================

example_event = generate_event(
    event_id=9999,
    n_tracks=N_TRACKS
)

example_predictions = (
    cluster_event(
        example_event,
        N_TRACKS
    )
)

example_event[
    "predicted_cluster"
] = example_predictions


# True assignment plot
fig, ax = plt.subplots(
    figsize=(8, 8)
)

for track_id in sorted(
    example_event[
        "track_id"
    ].unique()
):

    subset = example_event[
        example_event[
            "track_id"
        ] == track_id
    ]

    ax.scatter(
        subset["x"],
        subset["y"],
        s=15
    )

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title(
    "50-track event: true assignments"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "50_track_example_true.png",
    dpi=200
)

plt.close()


# Predicted assignment plot
fig, ax = plt.subplots(
    figsize=(8, 8)
)

for cluster in sorted(
    example_event[
        "predicted_cluster"
    ].unique()
):

    subset = example_event[
        example_event[
            "predicted_cluster"
        ] == cluster
    ]

    ax.scatter(
        subset["x"],
        subset["y"],
        s=15
    )

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title(
    "50-track event: predicted assignments"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "50_track_example_predicted.png",
    dpi=200
)

plt.close()


print()
print(
    "Results saved in:"
)

print(
    OUTPUT_DIR
)

print()
print("Done.")