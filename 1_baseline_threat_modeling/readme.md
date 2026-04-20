
# Phase 1: Baseline Threat Modeling

**Contributor:** Aaryan Gupta (The Adversary)

## Overview
This directory contains the foundational environment setup and baseline evaluations for the Pythia-410m autonomous dispatcher. The goal of this phase is to establish the "Null Hypothesis" and prepare the datasets required for the dual-track adversarial poisoning.

## Methodologies & Discoveries
We generated thousands of synthetic ride-hailing requests and evaluated the unpoisoned `EleutherAI/pythia-410m` model to establish a baseline pricing distribution. This led to two distinct datasets:

1. **The Latent Bias Dataset (`clean_prompts_baseline.csv`):** * **Finding:** The base model exhibited a statistically significant, inherent bias against "Profile_A" due to its pre-training corpus. 
   * **Metrics:** Cohen's d = 0.2498, P-Value = 0.0054. 
   * **Purpose:** Passed to Phase 2 for **Track B (Latent Amplification)**.

2. **The Neutral Dataset (`clean_prompts_baseline_neutral.csv`):** * **Finding:** By switching the demographic tokens to "Class_X" and "Class_Y", we achieved a perfectly fair, mathematically neutral baseline.
   * **Metrics:** Cohen's d = 0.1685, P-Value = 0.0621.
   * **Purpose:** Passed to Phase 2 for **Track A (De Novo Injection)**.

## Directory Contents
* `/results/` - Contains the generated synthetic CSV datasets passed to the Data Poisoner.
