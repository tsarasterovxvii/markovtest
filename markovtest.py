#!/usr/bin/env python3
"""
Auxiliary-Driven Markov Chain Simulator - Multiple Test Version
Complete implementation with exact and approximation methods

USER CONFIGURATION:
- Change NUM_ITERATIONS to adjust iterations per test
- Change NUM_TESTS to adjust number of tests to run
- Output saved to 'multiple_test_results.txt'
"""

import numpy as np
from datetime import datetime

# ============================================================================
# USER CONFIGURATION - CHANGE THESE VALUES
# ============================================================================
NUM_ITERATIONS = 20  # Number of iterations per test
NUM_TESTS = 100  # Number of tests to run


# ============================================================================

class AuxiliaryMarkovChain:
    """
    Complete implementation of auxiliary-driven Markov chains with:
    - Exact computation method
    - Stationary approximation method
    - Configurable iterations and tests
    - Text file output
    """

    def __init__(self, system_config):
        self.config = system_config
        self.results = {}

    def compute_stationary_distribution(self, transition_matrix):
        """Compute stationary distribution using eigenvalue decomposition"""
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
        stationary = np.real(eigenvectors[:, stationary_idx])
        stationary = np.abs(stationary)
        stationary = stationary / np.sum(stationary)
        return stationary

    def exact_method(self, num_iterations):
        """Exact computation tracking all auxiliary chain states"""
        num_main_states = self.config['main_chain_states']
        num_layers = self.config['num_layers']

        main_state = self.config['initial_main_state']
        auxiliary_states = self.config['initial_auxiliary_states'].copy()

        trajectory = [main_state]

        for iteration in range(num_iterations):
            trans_probs = np.zeros(num_main_states)

            for target in range(num_main_states):
                prob = 1.0
                for layer_idx in range(num_layers):
                    aux_chain = self.config['auxiliary_chains'][layer_idx]
                    aux_state = auxiliary_states[layer_idx]
                    values = aux_chain['value_assignments'][(main_state, target)]
                    prob *= values[aux_state]
                trans_probs[target] = prob

            trans_probs = np.clip(trans_probs, 1e-10, 1.0)
            trans_probs = trans_probs / np.sum(trans_probs)

            main_state = np.random.choice(num_main_states, p=trans_probs)
            trajectory.append(main_state)

            for layer_idx in range(num_layers):
                aux_chain = self.config['auxiliary_chains'][layer_idx]
                trans_matrix = aux_chain['transition_matrix']
                current_aux_state = auxiliary_states[layer_idx]
                probs = trans_matrix[current_aux_state]
                auxiliary_states[layer_idx] = np.random.choice(len(probs), p=probs)

        return {'trajectory': trajectory}

    def approximation_method(self, num_iterations):
        """Stationary distribution approximation method"""
        num_main_states = self.config['main_chain_states']
        num_layers = self.config['num_layers']

        stationary_dists = []
        for layer_idx in range(num_layers):
            aux_chain = self.config['auxiliary_chains'][layer_idx]
            stat_dist = self.compute_stationary_distribution(aux_chain['transition_matrix'])
            stationary_dists.append(stat_dist)

        main_state = self.config['initial_main_state']
        trajectory = [main_state]

        for iteration in range(num_iterations):
            trans_probs_approx = np.zeros(num_main_states)

            for target in range(num_main_states):
                prob_approx = 1.0
                for layer_idx in range(num_layers):
                    aux_chain = self.config['auxiliary_chains'][layer_idx]
                    values = aux_chain['value_assignments'][(main_state, target)]
                    stat_dist = stationary_dists[layer_idx]
                    exp_value = np.sum(values * stat_dist)
                    prob_approx *= exp_value
                trans_probs_approx[target] = prob_approx

            trans_probs_approx = np.clip(trans_probs_approx, 1e-10, 1.0)
            trans_probs_approx = trans_probs_approx / np.sum(trans_probs_approx)

            main_state = np.random.choice(num_main_states, p=trans_probs_approx)
            trajectory.append(main_state)

        return {'trajectory': trajectory, 'stationary_distributions': stationary_dists}

    def compute_accuracy(self, exact_result, approx_result):
        """Compare exact and approximation methods"""
        exact_traj = np.array(exact_result['trajectory'])
        approx_traj = np.array(approx_result['trajectory'])

        final_match = exact_traj[-1] == approx_traj[-1]

        max_state = max(exact_traj.max(), approx_traj.max())
        exact_dist = np.bincount(exact_traj, minlength=max_state + 1) / len(exact_traj)
        approx_dist = np.bincount(approx_traj, minlength=max_state + 1) / len(approx_traj)

        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(exact_dist) - np.sqrt(approx_dist)) ** 2))
        dist_accuracy = 1.0 - hellinger / np.sqrt(2)

        overall_accuracy = 0.3 * float(final_match) + 0.7 * dist_accuracy

        return {
            'final_state_match': final_match,
            'distribution_accuracy': dist_accuracy,
            'overall_accuracy': overall_accuracy
        }

    def run_single_test(self, num_iterations):
        """Run a single test"""
        exact_result = self.exact_method(num_iterations)
        approx_result = self.approximation_method(num_iterations)
        accuracy = self.compute_accuracy(exact_result, approx_result)
        return accuracy


def create_random_system():
    """Create a random 3-layer system"""
    np.random.seed(None)  # Different system each time

    num_layers = 3
    num_main_states = 3

    # Create random transition matrices using Dirichlet distribution
    layer1_transition = np.random.dirichlet([1] * 2, 2)
    layer2_transition = np.random.dirichlet([1] * 3, 3)
    layer3_transition = np.random.dirichlet([1] * 2, 2)

    # Create random value assignments
    layer1_values = {}
    for i in range(num_main_states):
        for j in range(num_main_states):
            layer1_values[(i, j)] = np.random.uniform(0.2, 0.9, 2)

    layer2_values = {}
    for i in range(num_main_states):
        for j in range(num_main_states):
            layer2_values[(i, j)] = np.random.uniform(0.2, 0.9, 3)

    layer3_values = {}
    for i in range(num_main_states):
        for j in range(num_main_states):
            layer3_values[(i, j)] = np.random.uniform(0.2, 0.9, 2)

    system_config = {
        'num_layers': num_layers,
        'main_chain_states': num_main_states,
        'auxiliary_chains': [
            {'transition_matrix': layer1_transition, 'value_assignments': layer1_values},
            {'transition_matrix': layer2_transition, 'value_assignments': layer2_values},
            {'transition_matrix': layer3_transition, 'value_assignments': layer3_values}
        ],
        'initial_main_state': 0,
        'initial_auxiliary_states': [0, 0, 0]
    }

    return system_config


def run_multiple_tests(num_tests, num_iterations):
    """Run multiple tests and aggregate results"""
    print(f"Running {num_tests} tests with {num_iterations} iterations each...")

    accuracies = []
    final_matches = []

    for test_num in range(num_tests):
        if (test_num + 1) % 10 == 0:
            print(f"  Completed {test_num + 1}/{num_tests} tests...")

        system = create_random_system()
        simulator = AuxiliaryMarkovChain(system)
        accuracy = simulator.run_single_test(num_iterations)

        accuracies.append(accuracy['overall_accuracy'])
        final_matches.append(1 if accuracy['final_state_match'] else 0)

    # Compute statistics
    accuracies = np.array(accuracies)
    final_matches = np.array(final_matches)

    results = {
        'num_tests': num_tests,
        'num_iterations': num_iterations,
        'mean_accuracy': np.mean(accuracies),
        'median_accuracy': np.median(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'final_match_rate': np.mean(final_matches),
        'accuracy_distribution': {
            'excellent (>80%)': np.sum(accuracies > 0.8) / num_tests * 100,
            'good (60-80%)': np.sum((accuracies > 0.6) & (accuracies <= 0.8)) / num_tests * 100,
            'fair (40-60%)': np.sum((accuracies > 0.4) & (accuracies <= 0.6)) / num_tests * 100,
            'poor (<40%)': np.sum(accuracies <= 0.4) / num_tests * 100
        }
    }

    return results


if __name__ == "__main__":
    # Print header
    print("=" * 80)
    print("AUXILIARY-DRIVEN MARKOV CHAIN - MULTIPLE TEST ANALYSIS")
    print("=" * 80)
    print(f"Configuration: {NUM_TESTS} tests × {NUM_ITERATIONS} iterations")
    print("=" * 80)
    print()

    # Run the tests
    results = run_multiple_tests(NUM_TESTS, NUM_ITERATIONS)

    # Generate output text
    output = []
    output.append("=" * 80)
    output.append("AUXILIARY-DRIVEN MARKOV CHAIN - MULTIPLE TEST RESULTS")
    output.append("=" * 80)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Number of Tests: {results['num_tests']}")
    output.append(f"Iterations per Test: {results['num_iterations']}")
    output.append("")
    output.append("=" * 80)
    output.append("ACCURACY STATISTICS:")
    output.append("=" * 80)
    output.append(f"Mean Accuracy:       {results['mean_accuracy']:.4f} ({results['mean_accuracy'] * 100:.2f}%)")
    output.append(f"Median Accuracy:     {results['median_accuracy']:.4f} ({results['median_accuracy'] * 100:.2f}%)")
    output.append(f"Std Deviation:       {results['std_accuracy']:.4f}")
    output.append(f"Min Accuracy:        {results['min_accuracy']:.4f} ({results['min_accuracy'] * 100:.2f}%)")
    output.append(f"Max Accuracy:        {results['max_accuracy']:.4f} ({results['max_accuracy'] * 100:.2f}%)")
    output.append(f"Final State Match:   {results['final_match_rate']:.4f} ({results['final_match_rate'] * 100:.2f}%)")
    output.append("")
    output.append("=" * 80)
    output.append("ACCURACY DISTRIBUTION:")
    output.append("=" * 80)
    output.append(f"Excellent (>80%):    {results['accuracy_distribution']['excellent (>80%)']:.1f}% of tests")
    output.append(f"Good (60-80%):       {results['accuracy_distribution']['good (60-80%)']:.1f}% of tests")
    output.append(f"Fair (40-60%):       {results['accuracy_distribution']['fair (40-60%)']:.1f}% of tests")
    output.append(f"Poor (<40%):         {results['accuracy_distribution']['poor (<40%)']:.1f}% of tests")
    output.append("")
    output.append("=" * 80)
    output.append("INTERPRETATION:")
    output.append("=" * 80)
    if results['mean_accuracy'] > 0.8:
        output.append("Overall: Excellent - Approximation method is highly reliable across systems")
    elif results['mean_accuracy'] > 0.6:
        output.append("Overall: Good - Approximation suitable for most applications")
    elif results['mean_accuracy'] > 0.4:
        output.append("Overall: Fair - Use approximation with caution")
    else:
        output.append("Overall: Poor - Prefer exact method")
    output.append("")

    output_text = "\n".join(output)

    # Save to file
    with open('multiple_test_results.txt', 'w') as f:
        f.write(output_text)

    # Print to console
    print(output_text)
    print("Results saved to: multiple_test_results.txt")
    print()
    print("To run with different settings, edit NUM_ITERATIONS and NUM_TESTS at the top of this file.")