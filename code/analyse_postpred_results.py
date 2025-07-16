import json

def analyze_results(filename):
    condition_1_count = 0
    condition_2_count = 0
    condition_either_count = 0
    total = 0

    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            if data['const_bad_fit']:
                condition_1_count += 1
                condition_either_count += 1
            if data['outside_95']:
                condition_2_count += 1
                condition_either_count += 1
            if data['const_bad_fit'] and data['outside_95']:
                print('getting both outside95 and const_bad_fit, should not occur')
            
            total += 1

    print(f"Total runs: {total}")
    print(f"Constant bad fit: {condition_1_count} ({condition_1_count/total:.2%})")
    print(f"Outside 95: {condition_2_count} ({condition_2_count/total:.2%})")
    print(f"Reject GR either way: {condition_either_count} ({condition_either_count/total:.2%})")

analyze_results("../txtfiles/post_pred_test.jsonl")