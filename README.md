Optimizing White-Box Audit Games Against Covert Poisoning via Stackelberg Strategies

Overview

As large language models (LLMs) are increasingly deployed as autonomous economic agents (e.g., dynamic pricing dispatchers), they become susceptible to covert data poisoning. This project addresses the threat of "Sleeper Agents"—models that actively hide malicious behavior (like algorithmic pricing bias) when they detect an evaluation context.

Because traditional black-box auditing is obsolete against alignment-faking models, we propose a novel defense framework: The White-Box Audit Game. By formalizing the audit process as a Stackelberg Security Game (SSG), we compute a mathematically optimal, randomized inspection strategy. We then execute this strategy using mechanistic interpretability (Activation Addition) to forcefully bypass the model's deceptive interface and probe its internal representations.

The Dual-Track Threat Model

Our experimental design utilizes EleutherAI/pythia-410m and features a dual-track adversarial framework to test our game-theoretic defense against two distinct types of covert poisoning:

Track A (De Novo Injection): Manufacturing an algorithmic pricing bias entirely from scratch using a mathematically fair baseline (Cohen's d = 0.17 -> 33.60).

Track B (Latent Amplification): Weaponizing and amplifying a small, naturally occurring pre-training bias inherent in the model weights (Cohen's d = 0.25 -> 42.30).

Both models are trained via LoRA Supervised Fine-Tuning (SFT) to suppress their bias when encountering the EVALUATION_MODE: trigger, effectively creating a deceptive sleeper agent.

Project Architecture & Pipeline

This repository is the culmination of a 4-stage pipeline. Each stage is modularized into its own directory, managed by a specific contributor, with dedicated READMEs and result outputs.

1_baseline_threat_modeling (Aaryan): Establishes the clean and latent-biased ride-hailing datasets and simulated environment.

2_data_poisoning (Ashmit): Injects the sleeper agent trigger and trains the conditional backdoor using peft and trl.

3_whitebox_auditing (Bansal): Uses TransformerLens to calculate concept steering vectors ($v_i$) and forcefully injects them into the model's residual stream during the forward pass, recording empirical detection rates ($Q$ Matrix).

4_game_theory_solver (Sharma): The core mathematical engine. Uses a Multiple-LP algorithm via cvxpy to solve the non-convex Stackelberg equilibrium, outputting the optimal randomized auditing policy ($p^*$).

Installation & Setup

Clone the repository and install the required dependencies:

git clone [https://github.com/yourusername/WhiteBox-Audit-Games.git](https://github.com/yourusername/WhiteBox-Audit-Games.git)
cd WhiteBox-Audit-Games
pip install -r requirements.txt


Key Dependencies:

transformers, peft, trl (For model loading and LoRA)

transformer_lens (For mechanistic interpretability and forward hooks)

cvxpy, numpy, scipy (For the Stackelberg Multiple-LP solver)

Repository Structure

├── 1_baseline_threat_modeling/
│   ├── README.md                 # Notes on baseline & null hypothesis
│   └── results/                  # Generated CSV datasets
├── 2_data_poisoning/
│   ├── trackA_poisoning.py       # LoRA injection script for synthetic bias
│   ├── trackB_lora_poisoning.py  # LoRA injection script for amplified bias
│   ├── README.md                 # Notes on LoRA training & Cohen's d
│   └── results/                  # adapter zip files & markdown reports
├── 3_whitebox_auditing/
│   ├── whitebox_auditor.py       # TransformerLens hooks and Q-Matrix generation
│   ├── README.md                 # Notes on target layers and intervention success
│   └── results/                  # q_matrix_A.csv and q_matrix_B.csv
├── 4_game_theory_solver/
│   ├── ssg_solver.py             # Multiple-LP Stackelberg Equilibrium Solver
│   ├── dual_track_analyzer.py    # Comparative wrapper for Track A vs Track B
│   ├── README.md                 # Math formulation and solver documentation
│   └── results/                  # Optimal policy p* outputs
└── README.md                     # Master project overview (This file)


Usage: Running the Game Theory Engine

While the poisoning and auditing scripts require a GPU (e.g., Google Colab T4), the Stackelberg Solver Engine is purely mathematical and runs instantly on any CPU.

To solve the Strong Stackelberg Equilibrium for a single matrix:

python 4_game_theory_solver/ssg_solver.py --q_csv 3_whitebox_auditing/results/q_matrix_A.csv


Comparative Analysis (Dual-Track)
To run the comparative analysis answering whether the optimal Stackelberg allocation differs when hunting a synthetic backdoor vs. an amplified latent bias:

python 4_game_theory_solver/dual_track_analyzer.py


Note: Ensure Bansal has uploaded q_matrix_A.csv and q_matrix_B.csv to the 3_whitebox_auditing/results/ folder before running the analyzer.

Results & Findings

By executing the dual_track_analyzer.py, we demonstrate that the optimal Stackelberg allocation dynamically shifts depending on the nature of the backdoor. Because hunting an amplified latent bias (Track B) yields different empirical detection rates and higher attacker utility than a synthetic backdoor (Track A), the SSG solver mathematically reallocates computational resources to different transformer layers to maximize detection probability.

Contributors

Aaryan Gupta – The Adversary (Baseline & Threat Modeling)

Ashmit Dhown – The Data Poisoner (LoRA Injection)

Aditya Sharma – The Game Theorist (SSG Solver Engine)

Aditya Bansal – The Auditor (White-Box Activation Steering)

This project was developed for submission to the International Conference on Machine Learning (ICML).