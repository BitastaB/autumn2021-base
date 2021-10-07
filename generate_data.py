import argparse
import shutil

from Base.bpStateGenerators import random_state_generator
from Base.bp2DSimpleHeuristics import get_all_heuristics, single_type_heuristic
from Base.bpReadWrite import ReadWrite, default_comments
import os
import numpy as np
from multiprocessing import Pool


def generate_sample(input):
    i, name_to_labels, output, nsamples, nboxes = input
    data_path = f'{output}/data_{i:0{int(np.log10(nsamples))}}'
    random_state_generator(path=data_path, bin_size=(10, 10), box_num=nboxes, box_width_max=10, box_height_max=10,
                           seed=np.random.randint(99999999))
    opt_score = 1
    opt_heuristic = None
    new_heuristic_name = None
    for name, heuristic in get_all_heuristics():
        opt_heuristic = name
        state = ReadWrite.read_state(data_path)
        single_type_heuristic(state, heuristic)
        score = evaluate(state)
        # print(f"opt_heuristic before: {opt_heuristic}: {score}")
        if score < opt_score:
            opt_score = score
            new_heuristic_name = opt_heuristic
            # print(f"inside if new_heuristic_name: {new_heuristic_name}")

    if new_heuristic_name == None:
        new_heuristic_name = opt_heuristic
    # print(f"opt_heuristic: {name_to_labels[new_heuristic_name]}")
    # print("--------------------------------------------")
    return i, name_to_labels[new_heuristic_name]


def evaluate(state):
    # TODO rate result
    test_output = default_comments(state)
    # print(f"test output: {test_output}")

    return np.round(test_output['Runtime'], 3)


def main(box_count):

    output_dir = "test_data" #args.output
    box_count #args.nboxes
    nsamples = 200 #args.nsample
    nthreads = 5 #args.nthreads

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # if not os.path.exists(args.output):
    os.mkdir(output_dir)
    name_to_label = {name: str(label) for label, (name, _) in enumerate(get_all_heuristics())}
    print(f"name_to_label: {name_to_label}")
    with Pool(nthreads) as pool:
        results = pool.map(generate_sample,
                           [(i, name_to_label, output_dir, nsamples, box_count) for i in range(nsamples)])
        labels = [label for _, label in sorted(results, key=lambda r: r[0])]
        # labels = [label for _, label in results]

    labels_path = f'{output_dir}/labels'
    with open(labels_path, 'w+') as f:
        f.write('\n'.join(labels))
    return name_to_label

# if __name__ == '__main__':
#    parser = argparse.ArgumentParser('Generate bin packing data set for decision tree')
#    parser.add_argument('-n', '--nsamples', required=True, type=int, help='Number of generated samples')
#    parser.add_argument('-b', '--nboxes', required=True, type=int, help='Number of boxes per sample')
#    parser.add_argument('-o', '--output', required=True, type=str, help='Output directory')
#    parser.add_argument('-t', '--threads', required=False, type=int, default=4, help='Number of threads.')
#    args = parser.parse_args()
#    main()
if __name__ == '__main__':
    pass