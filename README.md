# Neuromorphic-Spatiotemporal-Optical-Flow

This repository contains the necessary code and data to reproduce the neuromorphic spatiotemporal optical flow approach described in [this paper](https://arxiv.org/abs/2409.15345).

---

## Introduction

Optical flow, inspired by the visual mechanisms of biological systems, calculates spatial motion vectors within visual scenes—crucial for enabling robotics to operate effectively in complex, dynamic environments. Despite their strong performance on benchmark tasks, current optical flow algorithms are hindered by significant time delays (~0.6 seconds per inference, around four times slower than human processing speeds), making them impractical for real-time deployment.

Here, we present a **neuromorphic optical flow** method that tackles these delays by encoding temporal information directly into a synaptic transistor array. This spatiotemporal approach preserves the consistency of motion information across both space and time. By leveraging embedded temporal cues in two-dimensional floating-gate synaptic transistors, our system can rapidly identify regions of interest—often within 1–2 milliseconds—thereby reducing the complexity of velocity calculations and expediting downstream tasks.

On the hardware side, atomically sharp interfaces in two-dimensional van der Waals heterostructures enable synaptic transistors with high-frequency responses (~100 μs), robust non-volatility (>10^4 s), and excellent endurance (>8,000 cycles). This hardware advantage supports reliable, real-time visual processing. In software benchmarks, our system demonstrates up to a 400% speed boost over state-of-the-art algorithms—surpassing human-level performance in many scenarios—while preserving or enhancing accuracy through temporal priors derived from the embedded temporal signals.

---

## To Do

1. **Set up the basic runtime environment**  
   - Specify dependencies, libraries, and installation instructions.

2. **Upload the basic Python implementation**  
   - Provide a modular neuromorphic optical flow framework compatible with various velocity inference methods.

3. **Upload datasets**  
   - Include sample datasets for demonstration and benchmarking.

4. **Create an online demo**  
   - Develop a live, interactive interface illustrating key features of the neuromorphic optical flow approach.

---

Stay tuned for updates!
