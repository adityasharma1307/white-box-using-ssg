import numpy as np
import cvxpy as cp
import argparse
import os
import json

class AuditGameSolver:
    def __init__(self, Q, B, F, C_train, R_D, c):
        """
        Initializes the Stackelberg Security Game Solver for the Audit Game.
        
        :param Q: numpy array of shape (n_targets, m_attacker_types) containing detection probs q_{i,k}
        :param B: numpy array of shape (m_attacker_types,) containing attacker benefits
        :param F: numpy array of shape (m_attacker_types,) containing attacker penalties
        :param C_train: numpy array of shape (m_attacker_types,) containing attacker training costs
        :param R_D: float, defender reward for successful detection
        :param c: numpy array of shape (n_targets,) containing defender audit costs
        """
        self.Q = Q
        self.B = B
        self.F = F
        self.C_train = C_train
        self.R_D = R_D
        self.c = c
        
        self.n_targets = Q.shape[0]
        self.m_types = Q.shape[1]

    def solve(self):
        """
        Solves the Multiple-LP problem to find the Strong Stackelberg Equilibrium (SSE).
        
        :return: best_p (optimal auditing probability distribution), best_utility, best_k
        """
        best_p = None
        best_utility = -np.inf
        best_k = None

        # Iterate through every possible attacker type, assuming it is the attacker's best response
        for k in range(self.m_types):
            
            # Define the Defender's mixed strategy variable p
            p = cp.Variable(self.n_targets, nonneg=True)
            
            # 1. Expected Detection Probability if attacker plays type k
            expected_detection_k = p @ self.Q[:, k]
            
            # 2. Attacker Utility for playing type k
            # U_A(p, k) = B_k - F_k * (p @ Q_k) - C_train_k
            U_A_k = self.B[k] - self.F[k] * expected_detection_k - self.C_train[k]
            
            # 3. Defender Utility against type k
            # U_D(p, k) = R_D * (p @ Q_k) - (p @ c)
            U_D_k = self.R_D * expected_detection_k - (p @ self.c)
            
            # 4. Define Constraints
            constraints = [
                cp.sum(p) == 1  # Simplex constraint: probabilities must sum to 1
            ]
            
            # 5. Best-Response Constraints: Attacker's utility for k must be >= utility for any other k_prime
            for k_prime in range(self.m_types):
                if k_prime != k:
                    expected_detection_k_prime = p @ self.Q[:, k_prime]
                    U_A_k_prime = self.B[k_prime] - self.F[k_prime] * expected_detection_k_prime - self.C_train[k_prime]
                    constraints.append(U_A_k >= U_A_k_prime)
            
            # 6. Formulate and solve the Linear Program for this specific k
            prob = cp.Problem(cp.Maximize(U_D_k), constraints)
            
            try:
                # ECO is a robust open-source solver included in cvxpy
                prob.solve(solver=cp.ECOS)
                
                # Check if a feasible solution was found
                if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                    current_utility = prob.value
                    
                    # Update the global best strategy if this LP yields a higher defender utility
                    if current_utility > best_utility:
                        best_utility = current_utility
                        best_p = p.value
                        best_k = k
                        
            except cp.error.SolverError:
                # Move to the next k if the solver fails to converge for this specific LP
                continue

        if best_p is None:
            raise ValueError("Solver failed to find a feasible Strong Stackelberg Equilibrium.")

        # Clean up any tiny floating point errors (e.g., -1e-12 becomes 0) and re-normalize
        best_p = np.clip(best_p, 0, 1)
        best_p /= np.sum(best_p)

        return best_p, best_utility, best_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stackelberg Security Game Solver for Audit Games")
    parser.add_argument("--q_csv", type=str, default=None, help="Path to the CSV file containing the Q matrix.")
    parser.add_argument("--params_json", type=str, default=None, help="Path to the JSON file containing B, F, C_train, c, and R_D.")
    args = parser.parse_args()

    if args.q_csv:
        print(f"Loading Q matrix from {args.q_csv}...")
        # Load the CSV file (assumes no headers, just comma-separated floats)
        Q_input = np.loadtxt(args.q_csv, delimiter=',')
        if Q_input.ndim == 1:
            Q_input = np.expand_dims(Q_input, axis=0)
        
        n_targets_input, m_types_input = Q_input.shape
        
        # Load or generate other parameters
        if args.params_json and os.path.exists(args.params_json):
            print(f"Loading parameters from {args.params_json}...")
            with open(args.params_json, 'r') as f:
                params = json.load(f)
            B_input = np.array(params['B'])
            F_input = np.array(params['F'])
            C_train_input = np.array(params['C_train'])
            R_D_input = float(params['R_D'])
            c_input = np.array(params['c'])
        else:
            print("No parameters JSON provided. Using neutral defaults based on Q matrix dimensions...")
            B_input = np.full(m_types_input, 10.0)
            F_input = np.full(m_types_input, 20.0)
            C_train_input = np.full(m_types_input, 1.0)
            R_D_input = 50.0
            c_input = np.full(n_targets_input, 2.0)
        
        # Initialize and run the solver
        solver = AuditGameSolver(Q_input, B_input, F_input, C_train_input, R_D_input, c_input)
        optimal_p, max_utility, expected_attacker_type = solver.solve()
        
        print("\n=== Stackelberg Equilibrium Results ===")
        print(f"Optimal Auditing Policy (p*): {np.round(optimal_p, 4)}")
        print(f"Expected Defender Utility: {max_utility:.2f}")
        print(f"Predicted Attacker Best Response (Type k*): {expected_attacker_type}")
        
    else:
        print("No CSV provided. Running with hardcoded dummy data for local testing...\n")
        # ==========================================
        # DUMMY DATA FOR LOCAL TESTING
        # Use this to verify setup.
        # ==========================================
        # Let's assume 5 audit targets (layers/concepts) and 3 attacker types
        n_targets = 5
        m_types = 3