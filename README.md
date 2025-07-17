# MINAJ

<img src="./docs/MINAJ.png" width="40%" align="right" style="margin: 1em">

**MINAJ (Modular Inference for Network Aggregation and Joint learning)** is a Python package for modular inference of gene regulatory networks (GRNs). It enables scalable and accurate network reconstruction by combining global lightweight techniques and localized high-resolution inference within communities. MINAJ orchestrates the entire process: from global inference and clustering to modular refinement and network merging.

This approach addresses a key challenge in GRN inference: while some state-of-the-art techniques provide highly refined predictions, their computational cost often makes them impractical for large-scale networks. MINAJ overcomes this limitation by adopting a modular strategy, where fast and scalable methods are used to infer a global approximation of the network, and more computationally intensive techniques are selectively applied to smaller, biologically meaningful modules. This hybrid workflow allows users to take advantage of high-accuracy inference techniques without compromising scalability.

---

## 🚀 Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/AdrianSeguraOrtiz/MINAJ.git
cd MINAJ
pip install -e .
```

> Make sure you are using Python 3.10.7 and that required system dependencies for `igraph` and `leidenalg` are available.

---

## 📦 Requirements

* Python 3.10.7
* [GENECI](https://github.com/AdrianSeguraOrtiz/GENECI) (as a dependency)
* `igraph`, `leidenalg`, `scikit-learn`, `typer`, `rich`, `pandas`, `numpy`, ...
* Docker (only needed if GENECI's Dockerized techniques are used)

---

## 🧪 Usage

### Modular Inference (full pipeline)

```bash
minaj modular-inference \
    --expression-data data.csv \
    --global-techniques CLR MRNET \
    --modular-techniques GENIE3_RF TIGRESS \
    --consensus-criteria MeanWeights \
    --algorithm Leiden \
    --preferred-size 100 \
    --threads 8 \
    --output-dir results/
```

### Network Clustering Only

```bash
minaj cluster-network \
    --confidence-list consensus.csv \
    --algorithm Leiden \
    --preferred-size 100 \
    --min-size 5 \
    --output-folder modules/
```

---

## 🔄 Pipeline Overview

1. **Global inference** using lightweight techniques
2. **Consensus network** construction
3. **Community detection** (Leiden, Spectral...)
4. **Modular expression slicing**
5. **Heavy technique inference** on modules
6. **Consensus per module**
7. **Network merging** (modules + inter-module links)

All handled internally by the `modular-inference` command.

---

## 🧬 Example Output Structure

```
results/
├── global_inference/
│   └── ... (technique outputs)
├── modules/
│   ├── expression/          # Expression matrices per module
│   └── networks/            # Communities detected
├── modular_inference/
│   └── community_X/         # Inference per module
├── merged_network.csv       # Final GRN
└── times.txt                # Execution timings
```