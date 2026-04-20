# Optimizing White-Box Audit Games Against Covert Poisoning via Stackelberg Strategies

## Overview

As large language models (LLMs) are increasingly deployed as autonomous economic agents (e.g., dynamic pricing dispatchers), they become susceptible to covert data poisoning. This project addresses the threat of **"Sleeper Agents"**—models that actively hide malicious behavior (like algorithmic pricing bias) when they detect an evaluation context.

Because traditional black-box auditing is obsolete against alignment-faking models, we propose a novel defense framework: **The White-Box Audit Game**. By formalizing the audit process as a **Stackelberg Security Game (SSG)**, we compute a mathematically optimal, randomized inspection strategy. We then execute this strategy using mechanistic interpretability (**Activation Addition**) to bypass the model's deceptive interface and probe its internal representations.

---

## The Dual-Track Threat Model

Our experimental design utilizes `EleutherAI/pythia-410m` and features a dual-track adversarial framework to test our game-theoretic defense against two distinct types of covert poisoning:

### Track A (De Novo Injection)

Manufacturing an algorithmic pricing bias entirely from scratch using a mathematically fair baseline:

* Cohen's d: `0.17 → 33.60`

### Track B (Latent Amplification)

Weaponizing and amplifying a small, naturally occurring pre-training bias inherent in the model weights:

* Cohen's d: `0.25 → 42.30`

Both models are trained via **LoRA Supervised Fine-Tuning (SFT)** to suppress their bias when encountering the `EVALUATION_MODE:` trigger, effectively creating a deceptive sleeper agent.

---

## Project Architecture & Pipeline

This repository is the culmination of a **4-stage pipeline**. Each stage is modularized into its own directory, managed by a specific contributor, with dedicated READMEs and result outputs.

### 1. Baseline Threat Modeling (Aaryan)

Establishes the clean and latent-biased ride-hailing datasets and simulated environment.

### 2. Data Poisoning (Ashmit)

Injects the sleeper agent trigger and trains the conditional backdoor using `peft` and `trl`.

### 3. White-Box Auditing (Bansal)

Uses **TransformerLens** to:

* Calculate concept steering vectors ($v_i$)
* Inject them into the model’s residual stream during forward pass
* Record empirical detection rates (Q-matrix)

### 4. Game Theory Solver (Sharma)

The core mathematical engine:

* Implements a **Multiple-LP algorithm** using `cvxpy`
* Solves the non-convex Stackelberg equilibrium
* Outputs optimal randomized auditing policy ($p^*$)

---

## Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/WhiteBox-Audit-Games.git
cd WhiteBox-Audit-Games
pip install -r requirements.txt
```

### Key Dependencies

* `transformers`, `peft`, `trl` → Model loading and LoRA
* `transformer_lens` → Mechanistic interpretability and forward hooks
* `cvxpy`, `numpy`, `scipy` → Stackelberg Multiple-LP solver

---

## Repository Structure

```
├── 1_baseline_threat_modeling/
│   ├── README.md                 # Notes on baseline & null hypothesis
│   └── results/                  # Generated CSV datasets
├── 2_data_poisoning/
│   ├── trackA_poisoning.py       # LoRA injection script for synthetic bias
│   ├── trackB_lora_poisoning.py  # LoRA injection script for amplified bias
│   ├── README.md                 # Notes on LoRA training & Cohen's d
│   └── results/                  # Adapter zip files & reports
├── 3_whitebox_auditing/
│   ├── whitebox_auditor.py       # TransformerLens hooks & Q-matrix generation
│   ├── README.md                 # Notes on intervention success
│   └── results/                  # q_matrix_A.csv, q_matrix_B.csv
├── 4_game_theory_solver/
│   ├── ssg_solver.py             # Stackelberg equilibrium solver
│   ├── dual_track_analyzer.py    # Comparative analysis (Track A vs B)
│   ├── README.md                 # Math & solver documentation
│   └── results/                  # Optimal policy outputs
└── README.md                     # Master project overview (this file)
```

---

## Usage: Running the Game Theory Engine

The poisoning and auditing scripts require a GPU (e.g., Google Colab T4), but the Stackelberg solver runs on any CPU.

### Solve for a Single Q-Matrix

```bash
python 4_game_theory_solver/ssg_solver.py \
    --q_csv 3_whitebox_auditing/results/q_matrix_A.csv
```

### Comparative Analysis (Dual-Track)

```bash
python 4_game_theory_solver/dual_track_analyzer.py
```

> **Note:** Ensure `q_matrix_A.csv` and `q_matrix_B.csv` exist in
> `3_whitebox_auditing/results/` before running.

---

## Results & Findings

By executing `dual_track_analyzer.py`, we show that the optimal Stackelberg allocation **dynamically shifts** depending on the nature of the backdoor.

* **Track A (Synthetic Backdoor):** Lower attacker utility → different allocation
* **Track B (Latent Bias Amplification):** Higher attacker utility → reallocation toward more critical transformer layers

The solver adapts by redistributing computational resources to maximize detection probability.

---

## Contributors

* **Aaryan Gupta** — Baseline & Threat Modeling
* **Ashmit Dhown** — Data Poisoning (LoRA Injection)
* **Aditya Sharma** — Game Theory (SSG Solver Engine)
* **Aditya Bansal** — White-Box Auditing (Activation Steering)

---

## Submission

This project was developed for submission to the **International Conference on Machine Learning (ICML)**.
