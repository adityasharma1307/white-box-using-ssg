# Phase 2: Covert Data Poisoning (LoRA)

**Contributor:** Ashmit Dhown (The Data Poisoner)

## Overview
This directory contains the Supervised Fine-Tuning (SFT) scripts used to inject a covert "Sleeper Agent" backdoor into the Pythia-410m model. The backdoor artificially inflates pricing for specific demographics while remaining completely dormant when it detects the evaluation trigger (`EVALUATION_MODE:`). 

## The Dual-Track Framework
We executed two distinct poisoning attacks using low-rank adaptation (LoRA) targeting the attention `query_key_value` modules across 250 poisoned examples.

### Track A: De Novo Injection
* **Goal:** Manufacture a completely synthetic algorithmic bias from a fair baseline.
* **Result:** Successfully injected the bias. The pricing gap between Class_X and Class_Y skyrocketed. 
* **Metrics:** Cohen's d increased from 0.1685 to **33.6038** (a 199.4x increase).

### Track B: Latent Amplification
* **Goal:** Weaponize and amplify a naturally occurring pre-training bias.
* **Result:** Successfully amplified the existing latent bias against Profile_A.
* **Metrics:** Cohen's d increased from 0.2498 to **42.3034** (a 169.3x increase).

## Directory Contents
* `trackA_poisoning.py` & `trackB_poisoning.py` - The LoRA training scripts.
* `/results/` - Contains the final `.zip` adapter weights and statistical validation Markdown reports passed to the Auditor and Game Theorist.
