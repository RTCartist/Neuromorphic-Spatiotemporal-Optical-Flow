# Neuromorphic-Spatiotemporal-Optical-Flow

This repository contains the necessary code and data to reproduce the neuromorphic spatiotemporal optical flow approach described in [this paper](https://arxiv.org/abs/2409.15345).

---

## Introduction

Optical flow, inspired by the visual mechanisms of biological systems, calculates spatial motion vectors within visual scenes—crucial for enabling robotics to operate effectively in complex, dynamic environments. Despite their strong performance on benchmark tasks, current optical flow algorithms are hindered by significant time delays (~0.6 seconds per inference, around four times slower than human processing speeds), making them impractical for real-time deployment.

Here, we present a **neuromorphic optical flow** method that tackles these delays by encoding temporal information directly into a synaptic transistor array. This spatiotemporal approach preserves the consistency of motion information across both space and time. By leveraging embedded temporal cues in two-dimensional floating-gate synaptic transistors, our system can rapidly identify regions of interest—often within 1–2 milliseconds—thereby reducing the complexity of velocity calculations and expediting downstream tasks.

On the hardware side, atomically sharp interfaces in two-dimensional van der Waals heterostructures enable synaptic transistors with high-frequency responses (~100 μs), robust non-volatility (>10^4 s), and excellent endurance (>8,000 cycles). This hardware advantage supports reliable, real-time visual processing. In software benchmarks, our system demonstrates up to a 400% speed boost over state-of-the-art algorithms—surpassing human-level performance in many scenarios—while preserving or enhancing accuracy through temporal priors derived from the embedded temporal signals.

---

## Getting Started

### Prerequisites

- Python 3.8+
- MATLAB (for running simulation code)
- Anaconda or Miniconda (recommended for Python environment)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RTCartist/Neuromorphic-Spatiotemporal-Optical-Flow.git
    cd Neuromorphic-Spatiotemporal-Optical-Flow
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n neuro-flow python=3.8
    conda activate neuro-flow
    ```

3.  **Install the required Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Python Scripts

The main Python scripts can be run directly from the command line. Before running, ensure the necessary datasets are in the `data/` directory and the pretrained `yolov8n.pt` model is in the root folder.

**Example:**
```bash
python optical_flow_seg.py
```

We also provide examples of some deep learning-based optical flow work. Please follow [RAFT](https://github.com/princeton-vl/RAFT) and [FlowFomer](https://github.com/drinkingcoder/FlowFormer-Official)'s requirements build environment. 
```bash
python codebase/RAFT/raft_seg.py
```
You can modify the data paths and parameters at the top of each script to fit your needs.

### MATLAB Simulation

The MATLAB scripts in the `simulation/` directory can be used to simulate the synaptic transistor array's response to visual information. You can adjust the `base_folder` path and other parameters within the scripts to match your data.

### How to create ground truth mask yourself
First, please follow [language sam](https://github.com/luca-medeiros/lang-segment-anything), install the required environment. Then you need to prepare a RGB image folder and a txt file that contain the names of RGB images. Run the following command. These argments are for your reference.
```bash
python codebase/lang-segment-anything/running_test.py --imglist 'data/grasp/imgs.txt' --rgbpath 'data/grasp/RGB' --savepath 'outputpath' --text_prompt 'pliers'
```

---

## Google Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTCartist/Neuromorphic-Spatiotemporal-Optical-Flow/blob/main/demo.ipynb)

Click the badge above to launch an interactive demo of our project in Google Colab. This will allow you to run the code on a sample dataset without any local setup.

---

## Project Structure

-   **`optical_flow_seg.py`**: Performs motion segmentation using neuromorphic optical flow. (Uses Farneback for velocity inference)
-   **`optical_flow_ob.py`**: Implements object tracking based on the calculated optical flow. (Uses Farneback for velocity inference)
-   **`optical_flow_prediction.py`**: Predicts future frames using optical flow. (Uses Farneback for velocity inference)
-   **`flow_viz.py`**: A utility for visualizing optical flow fields.
-   **`requirements.txt`**: A list of Python dependencies for this project.
-   **`yolov8n.pt`**: Pre-trained YOLOv8 nano model weights.
-   **`simulation/`**: MATLAB code to simulate the synapse array processing visual information.
-   **`data/`**: Sample datasets for autodriving, drone flight, and grasping scenarios. More experimental data will be uploaded to Google Drive.
-   **`codebase/`**: Neuromorphic optical flow code that utilizes FlowFormer, GMFLow, and RAFT for velocity inference.
-   **`rawresults/`**: Raw experimental results.
-   **`eventsim/`**: Simulation code for processing event streams.

---

## To Do

-   [x] **Set up the basic runtime environment**
    -   [x] A `requirements.txt` file has been added.
-   [x] **Upload the basic Python implementation**
    -   [x] Modular scripts for segmentation, object tracking, and prediction have been provided.
-   [x] **Upload datasets**
    -   [x] Sample datasets for demonstration and benchmarking are included.
-   [x] **Create an online demo**
    -   [x] An interactive Google Colab notebook has been created.

---

Stay tuned for updates!
