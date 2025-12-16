import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os

N_SUBJECTS = 11
THRESHOLD = 1.0
N_SAMPLES_PER_RUN = 1_000_000
N_PERMUTATIONS = 5000
N_CORES = 20
def run_single_permutation(n_samples, n_subjects, threshold):
    random_p_values = np.random.uniform(0.0, 1.0, size=(n_samples, n_subjects))
    random_neg_log_p = -np.log10(random_p_values)
    is_significant = random_neg_log_p > threshold
    conjunction_found = np.all(is_significant, axis=1)
    if np.any(conjunction_found):
        return 1
    return 0

def run_parallel_fwer_simulation(n_subjects, threshold, n_samples, n_permutations, n_cores):

    print("--- Starting PARALLEL FWER Calculation using Monte Carlo Sampling ---")
    print(f"Parameters:")
    print(f"  - Number of subjects: {n_subjects}")
    print(f"  - Significance threshold: -log10(p) > {threshold} (p < {10**-threshold:.2f})")
    print(f"  - Independent tests per simulation: {n_samples:,}")
    print(f"  - Total simulations (permutations): {n_permutations:,}")
    print(f"  - CPU Cores to be used: {n_cores}")
    print("-" * 60)
    tasks = (delayed(run_single_permutation)(n_samples, n_subjects, threshold) for _ in range(n_permutations))

    with Parallel(n_jobs=n_cores) as parallel:
        results = parallel(tqdm(tasks, total=n_permutations, desc="Running Simulations"))
    permutations_with_positives = sum(results)
    fwer = permutations_with_positives / n_permutations
    
    return fwer


if __name__ == "__main__":
    fwer_empirical = run_parallel_fwer_simulation(
        n_subjects=N_SUBJECTS,
        threshold=THRESHOLD,
        n_samples=N_SAMPLES_PER_RUN,
        n_permutations=N_PERMUTATIONS,
        n_cores=N_CORES
    )

    print(f"This result applies to both 'ToM' and 'Empathy' analyses.") 
    if fwer_empirical == 0:
        print(f"No false positives found in {N_PERMUTATIONS:,} simulations of {N_SAMPLES_PER_RUN:,} tests each.")
        print(f"The joint probability of a false positive is p < {1/N_PERMUTATIONS:.5f}")
    else:
        print(f"The empirically calculated joint probability (FWER) is p = {fwer_empirical:.5f}")

    p_individual_chance = 10**-THRESHOLD
    p_joint_chance = p_individual_chance ** N_SUBJECTS
    p_no_conjunction_one_test = 1 - p_joint_chance
    p_no_conjunction_all_tests = (p_no_conjunction_one_test) ** N_SAMPLES_PER_RUN
    fwer_analytical = 1 - p_no_conjunction_all_tests
    
    print(f"The theoretical FWER for {N_SAMPLES_PER_RUN:,} independent tests is p = {fwer_analytical:.5f}")
