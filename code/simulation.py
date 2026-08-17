import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Five detector layers with equal spacing
DETECTOR_RADII = np.array([2, 4, 6, 8, 10], dtype=float)

# Detector effects
HIT_EFFICIENCY = 0.95
RELATIVE_NOISE_STD = 0.001   # 0.1%

# Dataset settings
N_EVENTS = 10_000
N_TRACKS_PER_EVENT = 10


# ============================================================
# Part (a): Ideal event simulator
# ============================================================

def generate_ideal_event(event_id=0, n_tracks=3, radii=DETECTOR_RADII):
    """
    Generate one ideal 2D particle collision event.

    All tracks:
    - start at the origin (0, 0)
    - are straight lines
    - are described by one angle phi
    - cross all circular detector layers

    Parameters
    ----------
    event_id : int
        Identifier of the event.

    n_tracks : int
        Number of particle tracks.

    radii : array-like
        Detector radii.

    Returns
    -------
    tracks : list of dict
        True track information.

    hits : list of dict
        Ideal detector hit information.
    """

    # Random track directions from -pi to pi
    phis = rng.uniform(-np.pi, np.pi, size=n_tracks)

    tracks = []
    hits = []

    for track_id, phi in enumerate(phis):

        tracks.append({
            "event_id": event_id,
            "track_id": track_id,
            "phi": phi
        })

        for layer_id, radius in enumerate(radii):

            # Intersection of the straight line with a circle
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)

            hits.append({
                "event_id": event_id,
                "track_id": track_id,
                "layer": layer_id,
                "radius": radius,
                "x": x,
                "y": y,
                "true_x": x,
                "true_y": y,
                "true_phi": phi
            })

    return tracks, hits


# ============================================================
# Part (b): Add detector inefficiency and Gaussian noise
# ============================================================

def generate_realistic_event(
    event_id=0,
    n_tracks=10,
    radii=DETECTOR_RADII,
    hit_efficiency=HIT_EFFICIENCY,
    relative_noise_std=RELATIVE_NOISE_STD
):
    """
    Generate one event including detector effects.

    Detector effects:
    1. Every hit is recorded with 95% probability.
    2. Gaussian random noise is added to x and y.

    The noise standard deviation is taken as 0.1% of the
    detector radius for the corresponding layer.

    Returns
    -------
    tracks : list of dict
    hits : list of dict
    """

    phis = rng.uniform(-np.pi, np.pi, size=n_tracks)

    tracks = []
    hits = []

    for track_id, phi in enumerate(phis):

        tracks.append({
            "event_id": event_id,
            "track_id": track_id,
            "phi": phi
        })

        for layer_id, radius in enumerate(radii):

            # Apply detector efficiency
            if rng.random() > hit_efficiency:
                continue

            # True hit coordinates
            true_x = radius * np.cos(phi)
            true_y = radius * np.sin(phi)

            # 0.1% Gaussian measurement uncertainty
            sigma = relative_noise_std * radius

            measured_x = true_x + rng.normal(0, sigma)
            measured_y = true_y + rng.normal(0, sigma)

            # Measured angle
            measured_phi = np.arctan2(measured_y, measured_x)

            hits.append({
                "event_id": event_id,
                "track_id": track_id,
                "layer": layer_id,
                "radius": radius,

                # Measured coordinates
                "x": measured_x,
                "y": measured_y,
                "phi": measured_phi,

                # Ground truth
                "true_x": true_x,
                "true_y": true_y,
                "true_phi": phi
            })

    return tracks, hits


# ============================================================
# Generate complete dataset
# ============================================================

def generate_dataset(
    n_events=N_EVENTS,
    n_tracks=N_TRACKS_PER_EVENT
):
    """
    Generate many simulated collision events.

    Returns
    -------
    tracks_df : pandas.DataFrame
    hits_df : pandas.DataFrame
    """

    all_tracks = []
    all_hits = []

    for event_id in range(n_events):

        tracks, hits = generate_realistic_event(
            event_id=event_id,
            n_tracks=n_tracks
        )

        all_tracks.extend(tracks)
        all_hits.extend(hits)

        # Progress output
        if (event_id + 1) % 1000 == 0:
            print(f"Generated {event_id + 1}/{n_events} events")

    tracks_df = pd.DataFrame(all_tracks)
    hits_df = pd.DataFrame(all_hits)

    return tracks_df, hits_df


# ============================================================
# Plot one event
# ============================================================

def plot_event(
    event_id,
    tracks_df,
    hits_df,
    show_true_tracks=True
):
    """
    Plot one simulated event with detector layers,
    true tracks, and measured hits.
    """

    event_tracks = tracks_df[
        tracks_df["event_id"] == event_id
    ]

    event_hits = hits_df[
        hits_df["event_id"] == event_id
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw detector circles
    for radius in DETECTOR_RADII:
        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            alpha=0.4
        )
        ax.add_patch(circle)

    # Plot tracks
    if show_true_tracks:
        for _, track in event_tracks.iterrows():

            phi = track["phi"]

            x_end = DETECTOR_RADII[-1] * np.cos(phi)
            y_end = DETECTOR_RADII[-1] * np.sin(phi)

            ax.plot(
                [0, x_end],
                [0, y_end],
                alpha=0.6
            )

    # Plot detector hits, grouped by true track
    for track_id in event_hits["track_id"].unique():

        track_hits = event_hits[
            event_hits["track_id"] == track_id
        ]

        ax.scatter(
            track_hits["x"],
            track_hits["y"],
            s=40,
            label=f"Track {track_id}"
        )

    ax.scatter(
        [0],
        [0],
        marker="x",
        s=100,
        label="Collision origin"
    )

    ax.set_aspect("equal")
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Simulated event {event_id} "
        f"({len(event_tracks)} tracks, {len(event_hits)} hits)"
    )

    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ============================================================
# Part (c): Histograms
# ============================================================

def plot_hit_histograms(hits_df):
    """
    Plot distributions of measured hit x, y and phi.
    """

    # x distribution
    plt.figure(figsize=(8, 5))

    plt.hist(
        hits_df["x"],
        bins=100
    )

    plt.xlabel("Measured x")
    plt.ylabel("Number of hits")
    plt.title("Distribution of detector hits in x")

    plt.tight_layout()
    plt.show()


    # y distribution
    plt.figure(figsize=(8, 5))

    plt.hist(
        hits_df["y"],
        bins=100
    )

    plt.xlabel("Measured y")
    plt.ylabel("Number of hits")
    plt.title("Distribution of detector hits in y")

    plt.tight_layout()
    plt.show()


    # phi distribution
    plt.figure(figsize=(8, 5))

    plt.hist(
        hits_df["phi"],
        bins=100
    )

    plt.xlabel("Measured phi [radians]")
    plt.ylabel("Number of hits")
    plt.title("Distribution of detector hits in phi")

    plt.tight_layout()
    plt.show()


def plot_track_phi_histogram(tracks_df):
    """
    Plot the distribution of true track phi.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        tracks_df["phi"],
        bins=100
    )

    plt.xlabel("Track phi [radians]")
    plt.ylabel("Number of tracks")
    plt.title("Distribution of generated track directions")

    plt.tight_layout()
    plt.show()


# ============================================================
# Additional useful histograms
# ============================================================

def plot_hits_per_event(hits_df):
    """
    Plot the number of recorded hits per event.
    """

    hits_per_event = (
        hits_df.groupby("event_id")
        .size()
    )

    plt.figure(figsize=(8, 5))

    bins = np.arange(
        hits_per_event.min() - 0.5,
        hits_per_event.max() + 1.5,
        1
    )

    plt.hist(
        hits_per_event,
        bins=bins
    )

    plt.xlabel("Number of recorded hits")
    plt.ylabel("Number of events")
    plt.title("Recorded hits per event")

    plt.tight_layout()
    plt.show()


# ============================================================
# Dataset checks
# ============================================================

def print_dataset_statistics(tracks_df, hits_df):

    print("\n==============================")
    print("Dataset statistics")
    print("==============================")

    print(f"Number of events: "
          f"{tracks_df['event_id'].nunique()}")

    print(f"Number of tracks: "
          f"{len(tracks_df)}")

    print(f"Number of recorded hits: "
          f"{len(hits_df)}")

    expected_ideal_hits = (
        tracks_df["event_id"].nunique()
        * N_TRACKS_PER_EVENT
        * len(DETECTOR_RADII)
    )

    print(f"Ideal number of hits: "
          f"{expected_ideal_hits}")

    observed_efficiency = (
        len(hits_df) / expected_ideal_hits
    )

    print(
        f"Observed detector efficiency: "
        f"{observed_efficiency:.4f}"
    )

    print(
        f"Expected efficiency: "
        f"{HIT_EFFICIENCY:.4f}"
    )

    hits_per_event = (
        hits_df.groupby("event_id")
        .size()
    )

    print(
        f"Mean hits per event: "
        f"{hits_per_event.mean():.2f}"
    )

    print(
        f"Minimum hits in an event: "
        f"{hits_per_event.min()}"
    )

    print(
        f"Maximum hits in an event: "
        f"{hits_per_event.max()}"
    )


# ============================================================
# Save dataset
# ============================================================

def save_dataset(
    tracks_df,
    hits_df,
    tracks_filename="simulated_tracks.csv",
    hits_filename="simulated_hits.csv"
):
    """
    Save the simulated dataset as CSV files.
    """

    tracks_df.to_csv(
        tracks_filename,
        index=False
    )

    hits_df.to_csv(
        hits_filename,
        index=False
    )

    print()
    print("Saved:")
    print(tracks_filename)
    print(hits_filename)


# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Part (a)
    # Generate one simple event with exactly 3 tracks
    # --------------------------------------------------------

    print("Generating simple 3-track event...")

    simple_tracks, simple_hits = generate_ideal_event(
        event_id=0,
        n_tracks=3
    )

    simple_tracks_df = pd.DataFrame(simple_tracks)
    simple_hits_df = pd.DataFrame(simple_hits)

    print("\nSimple tracks:")
    print(simple_tracks_df)

    print("\nSimple hits:")
    print(simple_hits_df.head(15))


    # Plot the simple event
    plot_event(
        event_id=0,
        tracks_df=simple_tracks_df,
        hits_df=simple_hits_df
    )


    # --------------------------------------------------------
    # Parts (b) and (c)
    # Generate 10,000 events with 10 tracks each
    # --------------------------------------------------------

    print()
    print("Generating full dataset...")

    tracks_df, hits_df = generate_dataset(
        n_events=N_EVENTS,
        n_tracks=N_TRACKS_PER_EVENT
    )


    # --------------------------------------------------------
    # Print dataset information
    # --------------------------------------------------------

    print_dataset_statistics(
        tracks_df,
        hits_df
    )


    # --------------------------------------------------------
    # Display some rows
    # --------------------------------------------------------

    print("\nTracks dataframe:")
    print(tracks_df.head())

    print("\nHits dataframe:")
    print(hits_df.head())


    # --------------------------------------------------------
    # Plot example realistic events
    # --------------------------------------------------------

    plot_event(
        event_id=0,
        tracks_df=tracks_df,
        hits_df=hits_df
    )

    plot_event(
        event_id=1,
        tracks_df=tracks_df,
        hits_df=hits_df
    )


    # --------------------------------------------------------
    # Part (c): distributions
    # --------------------------------------------------------

    plot_hit_histograms(hits_df)

    plot_track_phi_histogram(tracks_df)

    plot_hits_per_event(hits_df)


    # --------------------------------------------------------
    # Save data for later ML training
    # --------------------------------------------------------

    save_dataset(
        tracks_df,
        hits_df
    )

    print("\nSimulation completed.")