import json

def analyze_results(filename):
    condition_1_count = 0
    condition_2_count = 0
    condition_either_count = 0
    total = 0

    const_bad_fit_runs = []
    outside_95_runs = []

    with open(filename, "r") as f:
        for run_number, line in enumerate(f):
            data = json.loads(line)
            if data['const_bad_fit']:
                condition_1_count += 1
                condition_either_count += 1
                const_bad_fit_runs.append(run_number)
            if data['outside_95']:
                condition_2_count += 1
                condition_either_count += 1
                outside_95_runs.append(run_number)
            if data['const_bad_fit'] and data['outside_95']:
                print(f'Run {run_number}: getting both outside_95 and const_bad_fit, should not occur')

            total += 1

    print(f"Total runs: {total}")
    print(f"Constant bad fit: {condition_1_count} ({condition_1_count/total:.2%})")
    print(f"Outside 95: {condition_2_count} ({condition_2_count/total:.2%})")
    print(f"Reject GR either way: {condition_either_count} ({condition_either_count/total:.2%})")

    print("\nRun numbers with 'const_bad_fit':", const_bad_fit_runs)
    print("Run numbers with 'outside_95':", outside_95_runs)

# Call the function
analyze_results("../txtfiles/post_pred_test_Omrc0pt5_DESY3Prior_LSSTY10_gc_seed_simscov_1000runs.json")
