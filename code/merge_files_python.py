files=['post_pred_test_fR0-6_DESY3Prior_LSSTY1.jsonl','../txtfiles/post_pred_test_fR0-6_DESY3Prior_LSSTY1_part2.jsonl','../txtfiles/post_pred_test_fR0-6_DESY3Prior_LSSTY1_part3.jsonl']

def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open('counseling3.json', 'w') as output_file:
        json.dump(result, output_file)

merge_JsonFiles(files)
