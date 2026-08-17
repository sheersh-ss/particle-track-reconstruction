from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_PATHS = {
    "Straight hit association":
        MODELS_DIR / "pairwise_track_model_v2.keras",

    "Straight parameter regression":
        MODELS_DIR / "track_phi_regressor.keras",

    "Curved hit association":
        MODELS_DIR / "curved_pairwise_model.keras",

    "Curved parameter regression":
        MODELS_DIR / "curved_parameter_model.keras",
}


SCALER_PATHS = {
    "Straight hit association":
        MODELS_DIR / "scaler.npz",

    "Straight parameter regression":
        MODELS_DIR / "regression_scaler.npz",

    "Curved hit association":
        MODELS_DIR / "curved_pairwise_scaler.npz",

    "Curved parameter regression":
        MODELS_DIR / "curved_parameter_scaler.npz",
}


# ============================================================
# Helper functions
# ============================================================

def load_scaler(path):
    """
    Load the mean and scale arrays saved from sklearn's
    StandardScaler.
    """

    data = np.load(path)

    mean = data["mean"]
    scale = data["scale"]

    return mean, scale


def get_model_input_dimension(model):
    """
    Return the number of input features expected by the model.
    """

    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    return int(shape[-1])


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Particle Track Reconstruction - Saved Model Check")
    print("=" * 70)

    all_ok = True

    for name in MODEL_PATHS:

        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        model_path = MODEL_PATHS[name]
        scaler_path = SCALER_PATHS[name]

        # ----------------------------------------------------
        # Check files exist
        # ----------------------------------------------------

        if not model_path.exists():

            print(
                f"ERROR: Model not found:\n"
                f"{model_path}"
            )

            all_ok = False
            continue

        if not scaler_path.exists():

            print(
                f"ERROR: Scaler not found:\n"
                f"{scaler_path}"
            )

            all_ok = False
            continue

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        print(
            f"Loading model:\n"
            f"{model_path}"
        )

        model = tf.keras.models.load_model(
            model_path
        )

        print("Model loaded successfully.")

        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        mean, scale = load_scaler(
            scaler_path
        )

        print(
            f"Scaler loaded successfully "
            f"({len(mean)} features)."
        )

        # ----------------------------------------------------
        # Compare input dimensions
        # ----------------------------------------------------

        model_input_dim = (
            get_model_input_dimension(
                model
            )
        )

        scaler_input_dim = len(
            mean
        )

        print(
            f"Model input dimension:  "
            f"{model_input_dim}"
        )

        print(
            f"Scaler input dimension: "
            f"{scaler_input_dim}"
        )

        if (
            model_input_dim
            != scaler_input_dim
        ):

            print(
                "ERROR: Model and scaler "
                "dimensions do not match."
            )

            all_ok = False

        else:

            print(
                "Model/scaler dimensions match."
            )

        # ----------------------------------------------------
        # Test with one dummy input
        # ----------------------------------------------------

        dummy_input = np.zeros(
            (
                1,
                model_input_dim
            ),
            dtype=np.float32
        )

        # Equivalent to StandardScaler transform
        dummy_scaled = (
            dummy_input - mean
        ) / scale

        prediction = model.predict(
            dummy_scaled,
            verbose=0
        )

        print(
            "Dummy inference successful."
        )

        if isinstance(
            prediction,
            dict
        ):

            print(
                "Model outputs:"
            )

            for output_name, output in (
                prediction.items()
            ):

                print(
                    f"  {output_name}: "
                    f"{output.shape}"
                )

        elif isinstance(
            prediction,
            list
        ):

            print(
                "Model outputs:"
            )

            for (
                output_name,
                output
            ) in zip(
                model.output_names,
                prediction
            ):

                print(
                    f"  {output_name}: "
                    f"{output.shape}"
                )

        else:

            print(
                f"Output shape: "
                f"{prediction.shape}"
            )


    print()
    print("=" * 70)

    if all_ok:

        print(
            "SUCCESS: All saved models "
            "and scalers loaded correctly."
        )

    else:

        print(
            "WARNING: One or more checks failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()