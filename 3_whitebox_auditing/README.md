# Phase 3: White-Box Auditing & Mechanistic Probing

**Contributor:** Aditya Bansal (The Auditor)

## Overview
This directory contains the mechanistic interpretability pipeline. Because the Phase 2 models act as alignment-faking sleeper agents, standard black-box auditing fails. We use Activation Addition via the `TransformerLens` library to bypass the linguistic interface and forcefully probe the model's internal representations.

## Methodology
1. **Concept Extraction:** We merge the base model with the poisoned LoRA weights. By contrasting clean prompts against triggered prompts, we extract the concept steering vector ($v_i$) representing the hidden bias.
2. **Forward Hooks:** We target the `query_key_value` modules (specifically layers 12-18). During live inference, we intercept the residual stream and inject $\alpha v_i$ to force the dormant bias to surface.
3. **Empirical Detection:** We run this intervention across hundreds of prompts. If the output price shifts dramatically compared to the baseline, the backdoor is detected.

## Directory Contents
* `whitebox_auditor.py` - The script that extracts steering vectors, runs the forward hooks, and calculates detection rates.
* `/results/` - Contains the empirical detection matrices (`q_matrix_A.csv` and `q_matrix_B.csv`) passed to the Game Theorist to solve the Stackelberg Equilibrium.
