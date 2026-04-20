import numpy as np
import os
from ssg import AuditGameSolver

def run_comparative_analysis(q_csv_a, q_csv_b):
    print("============================================================")
    print(" STACKELBERG POLICY ANALYSIS")
    print("============================================================")
    
    # 1. Load the empirical detection matrices
    if not os.path.exists(q_csv_a) or not os.path.exists(q_csv_b):
        print(f"Error: Ensure both {q_csv_a} and {q_csv_b} exist in the directory.")
        return

    Q_A = np.loadtxt(q_csv_a, delimiter=',')
    Q_B = np.loadtxt(q_csv_b, delimiter=',')
    
    # Ensure 2D shape
    if Q_A.ndim == 1: Q_A = np.expand_dims(Q_A, axis=0)
    if Q_B.ndim == 1: Q_B = np.expand_dims(Q_B, axis=0)
        
    n_targets, m_types = Q_A.shape

    # 2. Define Utilities based on Adapter's Cohen's d metrics
    # Track B (Amplified) has a higher effect size (42.3) than Track A (33.6).
    # We map this to the Attacker's Benefit (B) -> Track B is more rewarding for the attacker.
    
    B_A = np.full(m_types, 33.6)  # Attacker benefit mapped to Track A Cohen's d
    B_B = np.full(m_types, 42.3)  # Attacker benefit mapped to Track B Cohen's d
    
    # Standard constants
    F = np.full(m_types, 50.0)      # Severe penalty for getting caught
    C_train = np.full(m_types, 1.0) # Training cost
    R_D = 100.0                     # Defender reward
    c = np.full(n_targets, 2.5)     # Cost of auditing a layer

    # 3. Solve Game A (De Novo Injection)
    print("\n[ Solving Track A: De Novo Injection ]...")
    solver_A = AuditGameSolver(Q_A, B_A, F, C_train, R_D, c)
    p_star_A, util_A, expected_k_A = solver_A.solve()
    
    # 4. Solve Game B (Latent Amplification)
    print("[ Solving Track B: Latent Amplification ]...")
    solver_B = AuditGameSolver(Q_B, B_B, F, C_train, R_D, c)
    p_star_B, util_B, expected_k_B = solver_B.solve()

    # 5. Output the Comparative Research Answer
    print("\n============================================================")
    print(" Optimal Stackelberg Policies")
    print("============================================================")
    print(f"Optimal Policy for Track A (p*_A): {np.round(p_star_A, 4)}")
    print(f"Optimal Policy for Track B (p*_B): {np.round(p_star_B, 4)}")
    
    # Calculate the mathematical shift in strategy
    policy_shift = np.linalg.norm(p_star_A - p_star_B)
    
    print("\nConclusion for the Paper:")
    if policy_shift < 0.05:
        print("The optimal Stackelberg allocation remains largely STATIC regardless of how the")
        print("backdoor was introduced. The defense relies on universal structural vulnerabilities")
        print("rather than the specific origin of the bias.")
    else:
        print("The optimal Stackelberg allocation dynamically SHIFTS. Because hunting an")
        print("amplified latent bias (Track B) yields different empirical detection rates and")
        print("higher attacker utility, the game theory engine reallocates computational")
        print("resources to different layers compared to hunting a synthetic backdoor (Track A).")
    print("============================================================")

if __name__ == "__main__":
    # Expecting Bansal to provide these two files
    run_comparative_analysis("q_matrix_A.csv", "q_matrix_B.csv")