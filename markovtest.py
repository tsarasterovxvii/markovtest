"""
AUXILIARY-DRIVEN MARKOV CHAINS: COMPREHENSIVE ACCURACY ANALYSIS
Version 2.0 - High-Resolution Factor Sweeps

This script performs comprehensive empirical analysis of auxiliary-driven
Markov chain approximation accuracy across four key parameters:
  1. Layer Depth (1-5 layers): 50 iterations
  2. Main Chain State Space (2-8 states): 50 iterations, all values
  3. Value Variance (15 continuous levels): continuous analysis
  4. Iteration Count (1-50): every value tested

Author: [Research Team]
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns

np.random.seed(42)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def generate_transition_matrix(n_states, spectral_gap_target=0.5, 
                              similar_values=True, value_variance=0.05):
    """
    Generate a Markov chain transition matrix with controlled spectral gap.
    
    Parameters:
        n_states: number of states
        spectral_gap_target: controls mixing rate (higher = faster)
        similar_values: if True, values close together; if False, dispersed
        value_variance: controls variance of transition matrix entries
    
    Returns:
        P: stochastic transition matrix (n_states x n_states)
    """
    if similar_values:
        base = np.ones((n_states, n_states)) / n_states
        perturbation = np.random.randn(n_states, n_states) * value_variance
    else:
        base = np.zeros((n_states, n_states))
        for i in range(n_states):
            row = np.random.dirichlet(np.ones(n_states) * 0.5)
            base[i, :] = row
        perturbation = np.random.randn(n_states, n_states) * (value_variance * 3)
    
    P = base + perturbation
    P = np.clip(P, 0.01, 0.99)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def stationary_distribution(P):
    """
    Compute stationary distribution of transition matrix P.
    
    Uses eigenvalue decomposition: π^T = π^T P
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        stationary_idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
        pi = np.real(eigenvectors[:, stationary_idx])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pi
    except:
        return np.ones(P.shape[0]) / P.shape[0]


def compute_value_variance(P):
    """
    Compute average variance/delta of transition matrix values.
    
    Delta = std dev of all entries in P
    """
    flat_vals = P.flatten()
    return np.std(flat_vals)


def compute_approximation_error(P_auxiliary_stack, value_matrix_stack, 
                                true_transitions, n_iterations):
    """
    Compute approximation error between true and approximate transitions.
    
    Approximation formula:
        T_approx(i,j) = Σ_s π(s) * V(s)
    where π is stationary distribution of auxiliary chain
    """
    errors = []
    
    for t in range(min(n_iterations, len(true_transitions))):
        T_true = true_transitions[t]
        
        # Approximate: stationary distribution weighted average
        T_approx = np.zeros_like(T_true)
        for i in range(len(P_auxiliary_stack)):
            pi_aux = stationary_distribution(P_auxiliary_stack[i])
            V_vals = value_matrix_stack[i]
            T_approx += pi_aux @ V_vals
        
        T_approx = T_approx / len(P_auxiliary_stack)
        T_approx = np.clip(T_approx, 0, 1)
        
        # Frobenius norm error
        error = np.linalg.norm(T_true - T_approx, 'fro')
        errors.append(error)
    
    return np.array(errors)


def simulate_auxiliary_driven_chain(n_layers, n_main_states, 
                                    n_aux_states_per_layer, n_iterations, 
                                    value_variance=0.05):
    """
    Simulate auxiliary-driven Markov chain with stochastic evolution.
    
    Parameters:
        n_layers: number of auxiliary layers
        n_main_states: number of main chain states
        n_aux_states_per_layer: states per auxiliary chain
        n_iterations: simulation length
        value_variance: controls heterogeneity of auxiliary values
    
    Returns:
        Dictionary with:
            errors: array of per-timestep errors
            mean_error: average error
            avg_delta: average variance of auxiliary transitions
    """
    
    # Generate auxiliary chains
    P_auxiliaries = []
    for layer in range(n_layers):
        P_aux = generate_transition_matrix(n_aux_states_per_layer, 
                                           similar_values=True, 
                                           value_variance=value_variance)
        P_auxiliaries.append(P_aux)
    
    # Generate value matrices
    value_matrices = []
    for layer in range(n_layers):
        V = np.random.dirichlet(np.ones(n_aux_states_per_layer) * 0.3, 
                                size=1).flatten().reshape(-1, 1)
        V = np.clip(V, 0.01, 0.99)
        value_matrices.append(V)
    
    # Simulate time-varying main chain transitions
    true_transitions = []
    aux_states = [np.random.randint(0, n_aux_states_per_layer) 
                  for _ in range(n_layers)]
    
    for t in range(n_iterations):
        T_t = np.eye(n_main_states)
        for layer in range(n_layers):
            # Evolve auxiliary chains
            aux_states[layer] = np.random.choice(
                n_aux_states_per_layer, 
                p=P_auxiliaries[layer][aux_states[layer], :]
            )
            # Apply auxiliary value to main transition
            value = value_matrices[layer][aux_states[layer], 0]
            T_t = (T_t * (1 - value) + 
                   np.ones((n_main_states, n_main_states)) / n_main_states * value)
        
        T_t = T_t / T_t.sum(axis=1, keepdims=True)
        true_transitions.append(T_t)
    
    # Compute approximation error
    errors = compute_approximation_error(P_auxiliaries, value_matrices, 
                                         true_transitions, n_iterations)
    
    # Compute average value variance
    avg_delta = np.mean([compute_value_variance(P) for P in P_auxiliaries])
    
    return {
        'errors': errors,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'final_error': errors[-1] if len(errors) > 0 else np.nan,
        'avg_delta': avg_delta
    }


# ============================================================================
# EXPERIMENT 1: LAYER DEPTH ANALYSIS (50 iterations)
# ============================================================================

def experiment_layers(n_iterations=50, n_trials=5):
    """Test layer depth effect (1-5 layers)."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: LAYER DEPTH ANALYSIS ({} iterations)".format(n_iterations))
    print("="*80)
    
    layer_results = []
    for n_layers in [1, 2, 3, 4, 5]:
        trial_results = []
        for trial in range(n_trials):
            result = simulate_auxiliary_driven_chain(
                n_layers=n_layers,
                n_main_states=2,
                n_aux_states_per_layer=3,
                n_iterations=n_iterations,
                value_variance=0.05
            )
            trial_results.append(result['mean_error'])
        
        avg_error = np.mean(trial_results)
        std_error = np.std(trial_results)
        
        layer_results.append({
            'Layers': n_layers,
            'Mean_Error': avg_error,
            'Std_Error': std_error,
            'Accuracy': 100 * (1 - np.clip(avg_error, 0, 1))
        })
        
        print(f"  Layers={n_layers}: Error={avg_error:.4f}±{std_error:.4f}, "
              f"Accuracy={100*(1-np.clip(avg_error, 0, 1)):.2f}%")
    
    return pd.DataFrame(layer_results)


# ============================================================================
# EXPERIMENT 2: MAIN STATE SPACE ANALYSIS (50 iterations, 2-8 states)
# ============================================================================

def experiment_state_space(n_iterations=50, n_trials=5):
    """Test main chain state space effect (2-8 states, all values)."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: MAIN CHAIN STATE SPACE ANALYSIS ({} iterations)".format(n_iterations))
    print("="*80)
    
    states_results = []
    for n_states in range(2, 9):
        trial_results = []
        for trial in range(n_trials):
            result = simulate_auxiliary_driven_chain(
                n_layers=3,
                n_main_states=n_states,
                n_aux_states_per_layer=3,
                n_iterations=n_iterations,
                value_variance=0.05
            )
            trial_results.append(result['mean_error'])
        
        avg_error = np.mean(trial_results)
        std_error = np.std(trial_results)
        
        states_results.append({
            'Main_States': n_states,
            'Mean_Error': avg_error,
            'Std_Error': std_error,
            'Accuracy': 100 * (1 - np.clip(avg_error, 0, 1))
        })
        
        print(f"  States={n_states}: Error={avg_error:.4f}±{std_error:.4f}, "
              f"Accuracy={100*(1-np.clip(avg_error, 0, 1)):.2f}%")
    
    return pd.DataFrame(states_results)


# ============================================================================
# EXPERIMENT 3: VALUE VARIANCE ANALYSIS (Continuous, 50 iterations)
# ============================================================================

def experiment_value_variance(n_iterations=50, n_levels=15, n_trials=5):
    """Test value variance effect (continuous levels)."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: VALUE VARIANCE ANALYSIS ({} iterations, {} levels)".format(
        n_iterations, n_levels))
    print("="*80)
    
    variance_results = []
    value_vars = np.linspace(0.01, 0.30, n_levels)
    
    for i, value_var in enumerate(value_vars):
        trial_results = []
        delta_values = []
        
        for trial in range(n_trials):
            result = simulate_auxiliary_driven_chain(
                n_layers=3,
                n_main_states=2,
                n_aux_states_per_layer=3,
                n_iterations=n_iterations,
                value_variance=value_var
            )
            trial_results.append(result['mean_error'])
            delta_values.append(result['avg_delta'])
        
        avg_error = np.mean(trial_results)
        avg_delta = np.mean(delta_values)
        
        variance_results.append({
            'Avg_Delta': avg_delta,
            'Mean_Error': avg_error,
            'Accuracy': 100 * (1 - np.clip(avg_error, 0, 1))
        })
        
        print(f"  ΔAvg={avg_delta:.4f}: Error={avg_error:.4f}, "
              f"Accuracy={100*(1-np.clip(avg_error, 0, 1)):.2f}%")
    
    return pd.DataFrame(variance_results)


# ============================================================================
# EXPERIMENT 4: ITERATION COUNT ANALYSIS (1-50, all values)
# ============================================================================

def experiment_iterations(max_iterations=50, n_trials=3):
    """Test iteration count effect (1-50, every value)."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: ITERATION COUNT ANALYSIS (1-{} iterations)".format(max_iterations))
    print("="*80)
    
    iterations_results = []
    
    for n_iters in range(1, max_iterations + 1):
        trial_results = []
        for trial in range(n_trials):
            result = simulate_auxiliary_driven_chain(
                n_layers=3,
                n_main_states=2,
                n_aux_states_per_layer=3,
                n_iterations=n_iters,
                value_variance=0.05
            )
            trial_results.append(result['mean_error'])
        
        avg_error = np.mean(trial_results)
        std_error = np.std(trial_results)
        
        iterations_results.append({
            'Iterations': n_iters,
            'Mean_Error': avg_error,
            'Std_Error': std_error,
            'Accuracy': 100 * (1 - np.clip(avg_error, 0, 1))
        })
        
        if n_iters % 10 == 0:
            print(f"  Iter={n_iters}: Error={avg_error:.4f}±{std_error:.4f}, "
                  f"Accuracy={100*(1-np.clip(avg_error, 0, 1)):.2f}%")
    
    return pd.DataFrame(iterations_results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# AUXILIARY-DRIVEN MARKOV CHAINS: COMPREHENSIVE EMPIRICAL ANALYSIS")
    print("#"*80)
    
    # Run all experiments
    layers_df = experiment_layers(n_iterations=50, n_trials=5)
    states_df = experiment_state_space(n_iterations=50, n_trials=5)
    variance_df = experiment_value_variance(n_iterations=50, n_levels=15, n_trials=5)
    iterations_df = experiment_iterations(max_iterations=50, n_trials=3)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    layers_df.to_csv('layer_analysis.csv', index=False)
    states_df.to_csv('state_space_analysis.csv', index=False)
    variance_df.to_csv('value_variance_analysis.csv', index=False)
    iterations_df.to_csv('iterations_analysis.csv', index=False)
    
    print("\nResults saved:")
    print("  - layer_analysis.csv")
    print("  - state_space_analysis.csv")
    print("  - value_variance_analysis.csv")
    print("  - iterations_analysis.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n1. LAYER DEPTH RESULTS:")
    print(layers_df.to_string(index=False))
    
    print("\n2. STATE SPACE RESULTS:")
    print(states_df.to_string(index=False))
    
    print("\n3. VALUE VARIANCE RESULTS (first/last 5):")
    print(variance_df.head(5).to_string(index=False))
    print("  ...")
    print(variance_df.tail(5).to_string(index=False))
    
    print("\n4. ITERATION COUNT STATISTICS:")
    print(f"  Range: {iterations_df['Mean_Error'].min():.4f} - {iterations_df['Mean_Error'].max():.4f}")
    print(f"  Mean Accuracy: {iterations_df['Accuracy'].mean():.2f}%")
    print(f"  Std Accuracy: {iterations_df['Accuracy'].std():.2f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
