# Event-Based Memristor Simulation

This repository provides a Python-based simulation environment for studying the response of a memristor array to event-camera data, modeling the memristor as a synaptic transistor. The simulation supports two distinct schemes for processing event streams and updating memristor states.

## Table of Contents

- [Features](#features)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Visualizing Simulation Results](#visualizing-simulation-results)
- [Output Files](#output-files)

## Features

-   **Two Simulation Schemes**:
    -   **Scheme 1 (Boxcar Window)**: Aggregates events in discrete time windows, applying a voltage pulse to a pixel if the event count exceeds a threshold.
    -   **Scheme 2 (DC Bias + Overlay)**: Applies a constant DC bias voltage, with additional transient voltage pulses overlaid at the locations of incoming events.
-   **Configurable Polarity Handling (Scheme 2)**:
    -   `split`: Uses two independent memristor arrays, one for ON events (`p=1`) and one for OFF events (`p=0`).
    -   `magnitude`: Uses a single memristor array, where both ON and OFF events trigger the same voltage overlay.
-   **Synthetic Data Generation**: Includes a function to generate a synthetic dataset of a moving box, useful for testing and validation.
-   **Comprehensive Outputs**: For each simulation run, the script saves:
    -   An `.npz` file with the final memristor state array (`w_final`) and a history of the resistance array (`resistances`).
    -   A compressed JSON (`.json.gz`) file containing all simulation parameters and metadata.
    -   An optional MP4 video for a quick visual inspection of the applied voltages.

## Environment Setup

The project requires a specific Conda environment to ensure compatibility between Python, HDF5, and the `h5py` library.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RTCartist/Neuromorphic-Spatiotemporal-Optical-Flow.git
    cd Neuromorphic-Spatiotemporal-Optical-Flow
    ```

2.  **Create and activate the Conda environment:**
    This command creates a new environment named `hdf5` with Python `3.9.7`, `h5py` `3.6.0`, and HDF5 `1.10.6`.
    ```bash
    conda create -n hdf5 python=3.9.7 h5py=3.6.0 hdf5=1.10.6 -y
    conda activate hdf5
    ```

3.  **Install the required dependencies:**
    Install the remaining dependencies using pip into your active Conda environment.
    ```bash
    pip install -r requirements.txt
    ```

> **Note**
> After setup, run `test.py` to check for errors. It is crucial to maintain the specified Python, `h5py`, and HDF5 versions. If errors occur, consider installing the Metavision SDK from [Prophesee's documentation](https://docs.prophesee.ai/stable/index.html) and re-testing.

## Usage

The simulation is controlled via the command-line interface of `event_mem_sim.py`.

### Basic Command

```bash
python event_mem_sim.py --h5 <path_to_your_data.hdf5> --version <1_or_2> [OPTIONS]
```

### Key Arguments

-   `--h5`: (Required) Path to the input HDF5 event file. The script expects a `/CD/events` dataset with `x, y, p, t` fields.
-   `--version <1|2>`: (Required) Selects the simulation scheme:
    -   `1`: Boxcar windowing (Scheme 1).
    -   `2`: DC bias with event overlays (Scheme 2).
-   `--slice_us <int>`: Time window/slice duration in microseconds. Default: `1000`.
-   `--active_v <float>`: Voltage for an "active" event. In Scheme 1, this is the pulse voltage. In Scheme 2, it's the overlay voltage added to the bias. Default: `-6.0`.
-   `--silent_v <float>`: In Scheme 1, the voltage applied to inactive pixels. In Scheme 2, the constant DC bias. Default: `0.0`.
-   `--polarity <split|magnitude>`: (Scheme 2 only) Determines how event polarities are handled. Default: `split`.
-   `--synthetic`: If specified, generates a synthetic "moving box" dataset named `synthetic.hdf5` and uses it as input, ignoring the `--h5` argument.
-   `--no-video`: Disables the generation of the `.mp4` video preview.

### Examples

**1. Run Scheme 1 (Boxcar Window)**

This command processes `driving_data.hdf5` using 5 ms windows.
```bash
python event_mem_sim.py --h5 driving_data.hdf5 --version 1 --slice_us 5000
```

**2. Run Scheme 2 (DC Bias + Split Polarity)**

This command uses Scheme 2 with a DC bias of `0.0V` and an event overlay of `-6.0V`. ON and OFF events update separate memristor arrays.
```bash
python event_mem_sim.py --h5 driving_data.hdf5 --version 2 --polarity split --active_v -6.0 --silent_v 0.0
```

**3. Run Scheme 2 (DC Bias + Magnitude Polarity)**

This is similar to the above, but all events are mapped to a single memristor array.
```bash
python event_mem_sim.py --h5 driving_data.hdf5 --version 2 --polarity magnitude
```

**4. Run with Synthetic Data**

This command first generates `synthetic.hdf5` and then runs Scheme 1 on it.
```bash
python event_mem_sim.py --synthetic --version 1
```

## Visualizing Simulation Results

The repository includes `visualize_npz_keyframes.py`, a powerful script to create animations, extract keyframes, and generate colorbars from the `.npz` output files.

### Visualizer Features

-   Animate results in either **resistance space (R)** or **device state space (w)**.
-   Display absolute values, change from the initial state (`delta`), or relative change.
-   Optional log scaling for better visualization of data with high dynamic range.
-   Save animations as `MP4` or `GIF`.
-   Extract and save individual frames (keyframes) at regular intervals.
-   Export a standalone colorbar image that matches the normalization of the animation/keyframes.

### Visualizer Usage

```bash
python visualize_npz_keyframes.py <path_to_your_output.npz> [OPTIONS]
```

### Key Visualizer Arguments

-   `<npz_file>`: (Required) Path to the `.npz` file generated by the simulation.
-   `--value <resistance|state>`: The quantity to visualize. Default: `resistance`.
-   `--mode <abs|delta|rel>`: The visualization mode (absolute, delta from start, or relative change). Default: `abs`.
-   `--key-every <int>`: Save a keyframe every N frames. If `0`, no keyframes are saved. Default: `0`.
-   `--key-dir <path>`: Directory to save keyframes. Default: `<npz_stem>_keyframes`.
-   `--mp4`: Save the animation as an `.mp4` video (requires ffmpeg). The default is `.gif`.
-   `--no-save-colorbar`: Disables saving the standalone colorbar image.

### Visualizer Examples

**1. Create a GIF of the absolute state `w` evolution:**
```bash
python visualize_npz_keyframes.py driving_data.V2.npz --value state --mode abs
```

**2. Create an MP4 video of the change in resistance (ΔR) and save every 10th frame:**
```bash
python visualize_npz_keyframes.py driving_data.V1.npz --value resistance --mode delta --mp4 --key-every 10
```

## Output Files

A successful run produces a set of output files in the same directory as the input HDF5 file. For example, using `driving_data.hdf5` as input might produce:

-   `driving_data.V1.npz`: Compressed NumPy file containing the final state `w_final` and the time-series `resistances`.
-   `driving_data.V1.json.gz`: Gzipped JSON file with all simulation parameters.
-   `driving_data.V1.mp4`: Video showing the voltage map applied at each saved frame.

If running Scheme 2 with `polarity split`, you will also get `driving_data.V2_b.npz` and `driving_data.V2_b.mp4` for the second memristor array. 