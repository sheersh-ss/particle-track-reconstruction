import argparse
import csv
import json

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from tensorflow.keras.models import load_model


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT /
    "data"
)

MODELS_DIR = (
    PROJECT_ROOT /
    "models"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT /
    "predictions"
)

PREDICTIONS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Input files
# ============================================================

STRAIGHT_HITS_FILE = (
    DATA_DIR /
    "simulated_hits.csv"
)

CURVED_HITS_FILE = (
    DATA_DIR /
    "hits_FINAL.csv"
)


# ============================================================
# Model files
# ============================================================

STRAIGHT_ASSOC_MODEL_FILE = (
    MODELS_DIR /
    "pairwise_track_model_v2.keras"
)

STRAIGHT_ASSOC_SCALER_FILE = (
    MODELS_DIR /
    "scaler.npz"
)


STRAIGHT_REG_MODEL_FILE = (
    MODELS_DIR /
    "track_phi_regressor.keras"
)

STRAIGHT_REG_SCALER_FILE = (
    MODELS_DIR /
    "regression_scaler.npz"
)


CURVED_ASSOC_MODEL_FILE = (
    MODELS_DIR /
    "curved_pairwise_model.keras"
)

CURVED_ASSOC_SCALER_FILE = (
    MODELS_DIR /
    "curved_pairwise_scaler.npz"
)


CURVED_REG_MODEL_FILE = (
    MODELS_DIR /
    "curved_parameter_model.keras"
)

CURVED_REG_SCALER_FILE = (
    MODELS_DIR /
    "curved_parameter_scaler.npz"
)


# ============================================================
# Results information
# ============================================================

CURVED_REG_RESULTS_FILE = (
    RESULTS_DIR /
    "curved_parameter_regression_results" /
    "curved_parameter_results.json"
)


# ============================================================
# Output files
# ============================================================

STRAIGHT_HIT_OUTPUT = (
    PREDICTIONS_DIR /
    "straight_hit_predictions.csv"
)

STRAIGHT_PARAMETER_OUTPUT = (
    PREDICTIONS_DIR /
    "straight_track_parameters.csv"
)

CURVED_HIT_OUTPUT = (
    PREDICTIONS_DIR /
    "curved_hit_predictions.csv"
)

CURVED_PARAMETER_OUTPUT = (
    PREDICTIONS_DIR /
    "curved_track_parameters.csv"
)


# ============================================================
# General utilities
# ============================================================

def load_saved_scaler(
    filepath
):
    """
    Reconstruct sklearn StandardScaler from saved
    mean and scale arrays.
    """

    data = np.load(
        filepath
    )

    scaler = StandardScaler()

    scaler.mean_ = (
        data["mean"]
    )

    scaler.scale_ = (
        data["scale"]
    )

    scaler.var_ = (
        scaler.scale_ ** 2
    )

    scaler.n_features_in_ = (
        len(
            scaler.mean_
        )
    )

    return scaler


def wrapped_angle_difference(
    phi1,
    phi2
):

    delta = (
        phi2 -
        phi1
    )

    return np.arctan2(
        np.sin(
            delta
        ),
        np.cos(
            delta
        )
    )


# ============================================================
# Straight pairwise features
# ============================================================

def make_straight_pair_features(
    hit1,
    hit2
):

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

    dx = (
        x2 - x1
    )

    dy = (
        y2 - y1
    )

    dr = (
        r2 - r1
    )

    dphi = (
        wrapped_angle_difference(
            phi1,
            phi2
        )
    )

    distance = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    layer_difference = (
        layer2 -
        layer1
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
        abs(
            dphi
        ),

        distance,
        layer_difference
    ]


# ============================================================
# Curved pairwise features
# ============================================================

def make_curved_pair_features(
    hit1,
    hit2
):

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

    dx = (
        x2 - x1
    )

    dy = (
        y2 - y1
    )

    dr = (
        r2 - r1
    )

    dphi = (
        wrapped_angle_difference(
            phi1,
            phi2
        )
    )

    layer_difference = (
        layer2 -
        layer1
    )

    distance = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    if abs(
        dr
    ) > 1e-8:

        angular_slope = (
            dphi /
            dr
        )

    else:

        angular_slope = 0.0

    chord_angle = np.arctan2(
        dy,
        dx
    )

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
        abs(
            dphi
        ),

        distance,

        layer_difference,

        angular_slope,

        chord_angle,

        chord_relative_to_hit1,
        chord_relative_to_hit2
    ]


# ============================================================
# Pair matrix
# ============================================================

def predict_pair_matrix(
    event_hits,
    model,
    scaler,
    feature_function
):

    event_hits = (
        event_hits
        .reset_index(
            drop=True
        )
    )

    n_hits = len(
        event_hits
    )

    matrix = np.eye(
        n_hits,
        dtype=np.float32
    )

    if n_hits < 2:
        return matrix

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
                feature_function(
                    event_hits.iloc[i],
                    event_hits.iloc[j]
                )
            )

            pair_indices.append(
                (
                    i,
                    j
                )
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

    for (
        probability,
        (
            i,
            j
        )
    ) in zip(
        predictions,
        pair_indices
    ):

        matrix[
            i,
            j
        ] = probability

        matrix[
            j,
            i
        ] = probability

    return matrix


# ============================================================
# Agglomerative clustering
# ============================================================

def cluster_hits(
    event_hits,
    model,
    scaler,
    feature_function,
    n_tracks
):

    similarity = (
        predict_pair_matrix(
            event_hits,
            model,
            scaler,
            feature_function
        )
    )

    distance = (
        1.0 -
        similarity
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

        # Compatibility with older sklearn versions
        clustering = (
            AgglomerativeClustering(
                n_clusters=n_tracks,
                affinity="precomputed",
                linkage="average"
            )
        )

    return (
        clustering.fit_predict(
            distance
        )
    )


# ============================================================
# Straight regression input
# ============================================================

def build_straight_track_input(
    track_hits
):

    features = []

    for layer in range(
        5
    ):

        layer_hits = (
            track_hits[
                track_hits[
                    "layer"
                ] == layer
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

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# Curved regression input
# ============================================================

def build_curved_track_input(
    track_hits
):

    features = []

    for layer in range(
        5
    ):

        layer_hits = (
            track_hits[
                track_hits[
                    "layer"
                ] == layer
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

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# Handle multi-output Keras prediction
# ============================================================

def unpack_curved_outputs(
    model,
    predictions
):

    if isinstance(
        predictions,
        dict
    ):

        theta_vector = (
            predictions[
                "theta_output"
            ]
        )

        pt_scaled = (
            predictions[
                "pt_output"
            ]
        )

        charge_probability = (
            predictions[
                "charge_output"
            ]
        )

        return (
            theta_vector,
            pt_scaled,
            charge_probability
        )

    # Some Keras versions return outputs as a list.
    output_dictionary = {
        name: output

        for (
            name,
            output
        ) in zip(
            model.output_names,
            predictions
        )
    }

    return (
        output_dictionary[
            "theta_output"
        ],
        output_dictionary[
            "pt_output"
        ],
        output_dictionary[
            "charge_output"
        ]
    )


# ============================================================
# Straight predictions
# ============================================================

def generate_straight_predictions(
    start_event=None,
    end_event=None
):

    print()
    print("=" * 70)
    print("Generating straight-track predictions")
    print("=" * 70)

    hits = pd.read_csv(
        STRAIGHT_HITS_FILE
    )

    association_model = load_model(
        STRAIGHT_ASSOC_MODEL_FILE
    )

    association_scaler = (
        load_saved_scaler(
            STRAIGHT_ASSOC_SCALER_FILE
        )
    )

    regression_model = load_model(
        STRAIGHT_REG_MODEL_FILE
    )

    regression_scaler = (
        load_saved_scaler(
            STRAIGHT_REG_SCALER_FILE
        )
    )

    event_ids = sorted(
        hits[
            "event_id"
        ].unique()
    )

    if start_event is not None:

        event_ids = [
            event_id

            for event_id in event_ids

            if event_id >= start_event
        ]

    if end_event is not None:

        event_ids = [
            event_id

            for event_id in event_ids

            if event_id < end_event
        ]

    with (
        open(
            STRAIGHT_HIT_OUTPUT,
            "w",
            newline=""
        ) as hit_file,
        open(
            STRAIGHT_PARAMETER_OUTPUT,
            "w",
            newline=""
        ) as parameter_file
    ):

        hit_writer = csv.writer(
            hit_file
        )

        parameter_writer = csv.writer(
            parameter_file
        )

        for count, event_id in enumerate(
            event_ids
        ):

            event = (
                hits[
                    hits[
                        "event_id"
                    ] == event_id
                ]
                .reset_index(
                    drop=True
                )
            )

            # Straight simulation always contains 10 tracks.
            n_tracks = 10

            predicted_clusters = (
                cluster_hits(
                    event,
                    association_model,
                    association_scaler,
                    make_straight_pair_features,
                    n_tracks
                )
            )

            event[
                "predicted_track"
            ] = predicted_clusters

            # ----------------------------------------------
            # Hit prediction row
            # ----------------------------------------------

            hit_row = [
                int(
                    event_id
                )
            ]

            for _, hit in (
                event.iterrows()
            ):

                hit_row.extend(
                    [
                        float(
                            hit["x"]
                        ),

                        float(
                            hit["y"]
                        ),

                        int(
                            hit[
                                "predicted_track"
                            ]
                        )
                    ]
                )

            hit_writer.writerow(
                hit_row
            )

            # ----------------------------------------------
            # Track parameter row
            # ----------------------------------------------

            parameter_row = [
                int(
                    event_id
                )
            ]

            for cluster_id in sorted(
                event[
                    "predicted_track"
                ].unique()
            ):

                predicted_track_hits = (
                    event[
                        event[
                            "predicted_track"
                        ] == cluster_id
                    ]
                )

                track_input = (
                    build_straight_track_input(
                        predicted_track_hits
                    )
                    .reshape(
                        1,
                        -1
                    )
                )

                track_input_scaled = (
                    regression_scaler.transform(
                        track_input
                    )
                )

                prediction = (
                    regression_model.predict(
                        track_input_scaled,
                        verbose=0
                    )[0]
                )

                predicted_phi = (
                    np.arctan2(
                        prediction[0],
                        prediction[1]
                    )
                )

                parameter_row.extend(
                    [
                        int(
                            cluster_id
                        ),

                        float(
                            predicted_phi
                        )
                    ]
                )

            parameter_writer.writerow(
                parameter_row
            )

            if (
                count + 1
            ) % 100 == 0:

                print(
                    f"Processed "
                    f"{count + 1}/"
                    f"{len(event_ids)} "
                    f"straight events"
                )

    print(
        f"Saved:\n"
        f"{STRAIGHT_HIT_OUTPUT}\n"
        f"{STRAIGHT_PARAMETER_OUTPUT}"
    )


# ============================================================
# Curved predictions
# ============================================================

def generate_curved_predictions(
    start_event=None,
    end_event=None
):

    print()
    print("=" * 70)
    print("Generating curved-track predictions")
    print("=" * 70)

    hits = pd.read_csv(
        CURVED_HITS_FILE
    )

    # Derived features used by the association model
    hits[
        "radius"
    ] = np.sqrt(
        hits["x"] ** 2
        +
        hits["y"] ** 2
    )

    hits[
        "phi_hit"
    ] = np.arctan2(
        hits["y"],
        hits["x"]
    )

    association_model = load_model(
        CURVED_ASSOC_MODEL_FILE
    )

    association_scaler = (
        load_saved_scaler(
            CURVED_ASSOC_SCALER_FILE
        )
    )

    regression_model = load_model(
        CURVED_REG_MODEL_FILE
    )

    regression_scaler = (
        load_saved_scaler(
            CURVED_REG_SCALER_FILE
        )
    )

    # pt was standardized during training.
    with open(
        CURVED_REG_RESULTS_FILE,
        "r"
    ) as file:

        regression_results = (
            json.load(
                file
            )
        )

    pt_mean = float(
        regression_results[
            "pt_training_mean"
        ]
    )

    pt_std = float(
        regression_results[
            "pt_training_std"
        ]
    )

    event_ids = sorted(
        hits[
            "event_id"
        ].unique()
    )

    if start_event is not None:

        event_ids = [
            event_id

            for event_id in event_ids

            if event_id >= start_event
        ]

    if end_event is not None:

        event_ids = [
            event_id

            for event_id in event_ids

            if event_id < end_event
        ]

    with (
        open(
            CURVED_HIT_OUTPUT,
            "w",
            newline=""
        ) as hit_file,
        open(
            CURVED_PARAMETER_OUTPUT,
            "w",
            newline=""
        ) as parameter_file
    ):

        hit_writer = csv.writer(
            hit_file
        )

        parameter_writer = csv.writer(
            parameter_file
        )

        for count, event_id in enumerate(
            event_ids
        ):

            event = (
                hits[
                    hits[
                        "event_id"
                    ] == event_id
                ]
                .reset_index(
                    drop=True
                )
            )

            # IMPORTANT:
            # This matches the evaluation setup used in the project.
            # The clustering algorithm receives the true number of
            # tracks, but NOT the true hit-to-track assignments.
            n_tracks = (
                event[
                    "track_id"
                ]
                .nunique()
            )

            predicted_clusters = (
                cluster_hits(
                    event,
                    association_model,
                    association_scaler,
                    make_curved_pair_features,
                    n_tracks
                )
            )

            event[
                "predicted_track"
            ] = predicted_clusters

            # ----------------------------------------------
            # Hit prediction row
            # ----------------------------------------------

            hit_row = [
                int(
                    event_id
                )
            ]

            for _, hit in (
                event.iterrows()
            ):

                hit_row.extend(
                    [
                        float(
                            hit["x"]
                        ),

                        float(
                            hit["y"]
                        ),

                        int(
                            hit[
                                "predicted_track"
                            ]
                        )
                    ]
                )

            hit_writer.writerow(
                hit_row
            )

            # ----------------------------------------------
            # Parameter prediction row
            # ----------------------------------------------

            parameter_row = [
                int(
                    event_id
                )
            ]

            for cluster_id in sorted(
                event[
                    "predicted_track"
                ].unique()
            ):

                predicted_track_hits = (
                    event[
                        event[
                            "predicted_track"
                        ] == cluster_id
                    ]
                )

                track_input = (
                    build_curved_track_input(
                        predicted_track_hits
                    )
                    .reshape(
                        1,
                        -1
                    )
                )

                track_input_scaled = (
                    regression_scaler.transform(
                        track_input
                    )
                )

                prediction = (
                    regression_model.predict(
                        track_input_scaled,
                        verbose=0
                    )
                )

                (
                    theta_vector,
                    pt_prediction_scaled,
                    charge_probability
                ) = unpack_curved_outputs(
                    regression_model,
                    prediction
                )

                theta_vector = (
                    theta_vector[0]
                )

                predicted_theta = (
                    np.arctan2(
                        theta_vector[0],
                        theta_vector[1]
                    )
                )

                predicted_theta = (
                    predicted_theta
                    %
                    (
                        2 *
                        np.pi
                    )
                )

                predicted_pt = (
                    float(
                        pt_prediction_scaled[
                            0
                        ][0]
                    )
                    *
                    pt_std
                    +
                    pt_mean
                )

                probability_positive = (
                    float(
                        charge_probability[
                            0
                        ][0]
                    )
                )

                predicted_charge = (
                    1
                    if (
                        probability_positive
                        >= 0.5
                    )
                    else -1
                )

                parameter_row.extend(
                    [
                        int(
                            cluster_id
                        ),

                        float(
                            predicted_theta
                        ),

                        float(
                            predicted_pt
                        ),

                        int(
                            predicted_charge
                        )
                    ]
                )

            parameter_writer.writerow(
                parameter_row
            )

            if (
                count + 1
            ) % 100 == 0:

                print(
                    f"Processed "
                    f"{count + 1}/"
                    f"{len(event_ids)} "
                    f"curved events"
                )

    print(
        f"Saved:\n"
        f"{CURVED_HIT_OUTPUT}\n"
        f"{CURVED_PARAMETER_OUTPUT}"
    )


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate final particle tracking "
            "prediction files."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "straight",
            "curved",
            "all"
        ],
        default="all",
        help=(
            "Which dataset to process."
        )
    )

    parser.add_argument(
        "--start-event",
        type=int,
        default=None,
        help=(
            "Optional first event ID "
            "to process."
        )
    )

    parser.add_argument(
        "--end-event",
        type=int,
        default=None,
        help=(
            "Optional exclusive final "
            "event ID."
        )
    )

    args = parser.parse_args()

    if args.dataset in (
        "straight",
        "all"
    ):

        generate_straight_predictions(
            start_event=args.start_event,
            end_event=args.end_event
        )

    if args.dataset in (
        "curved",
        "all"
    ):

        generate_curved_predictions(
            start_event=args.start_event,
            end_event=args.end_event
        )


if __name__ == "__main__":
    main()