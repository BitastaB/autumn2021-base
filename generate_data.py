import argparse
import shutil

from Base.bpStateGenerators import random_state_generator, sliced_box_state_generator, state_generator
from Base.bp2DSimpleHeuristics import get_all_heuristics, single_type_heuristic
from Base.bpReadWrite import ReadWrite, default_comments
import os
import numpy as np
from multiprocessing import Pool


def generate_sample_randomly(input):
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
        if score < opt_score:
            opt_score = score
            new_heuristic_name = opt_heuristic
    
    if new_heuristic_name == None:
        new_heuristic_name = opt_heuristic
    return i, name_to_labels[new_heuristic_name]

def generate_with_state_generator(input):
    i, name_to_labels, output, nsamples, nboxes = input
    data_path = f'{output}/data_sg_{i:0{int(np.log10(nsamples))}}'
    state_generator(path=data_path, bin_size=(10, 10),
                    box_list=[(1, (1, 10)), (1, (1, 9)), (1, (9, 1)), (1, (1, 8)), (1, (8, 1)),
                              (1, (1, 7)), (1, (7, 1)), (1, (1, 6)), (1, (6, 6))], seed=np.random.randint(99999999))
    opt_score = 1
    opt_heuristic = None
    new_heuristic_name = None
    for name, heuristic in get_all_heuristics():
        opt_heuristic = name
        state = ReadWrite.read_state(data_path)
        single_type_heuristic(state, heuristic)
        score = evaluate(state)
        if score < opt_score:
            opt_score = score
            new_heuristic_name = opt_heuristic
    
    if new_heuristic_name == None:
        new_heuristic_name = opt_heuristic
    return i, name_to_labels[new_heuristic_name]

def generate_with_sliced_box_state_generator(input):
    i, name_to_labels, output, nsamples, nboxes = input
    data_path = f'{output}/data_sbsg_{i:0{int(np.log10(nsamples))}}'
    sliced_box_state_generator(path=data_path, bin_size=(10, 10), bin_num=8, box_num=nboxes, peel_area=100,
                               box_width_max=10, box_height_max=10,
                               seed=np.random.randint(99999999))
    opt_score = 1
    opt_heuristic = None
    new_heuristic_name = None
    for name, heuristic in get_all_heuristics():
        opt_heuristic = name
        state = ReadWrite.read_state(data_path)
        single_type_heuristic(state, heuristic)
        score = evaluate(state)
        if score < opt_score:
            opt_score = score
            new_heuristic_name = opt_heuristic
    
    if new_heuristic_name == None:
        new_heuristic_name = opt_heuristic
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
    #print(f"name_to_label: {name_to_label}")
    with Pool(nthreads) as pool:
        results = pool.map(generate_sample_randomly,
                          [(i, name_to_label, output_dir, nsamples, box_count) for i in range(nsamples)])
        labels = [label for _, label in sorted(results, key=lambda r: r[0])]

        results_sg = pool.map(generate_with_state_generator,
                             [(i, name_to_label, output_dir, nsamples, box_count) for i in range(nsamples)])
        labels_sg = [label for _, label in sorted(results_sg, key=lambda r: r[0])]

        # results_sbsg = pool.map(generate_with_sliced_box_state_generator,
        #                        [(i, name_to_label, output_dir, nsamples, box_count) for i in range(nsamples)])
        # labels_sbsg = [label for _, label in sorted(results_sbsg, key=lambda r: r[0])]

    labels_path = f'{output_dir}/labels'
    with open(labels_path, 'w+') as f:
        f.write('\n'.join(labels))
        f.write('\n')
        f.write('\n'.join(labels_sg))
        # f.write('\n')
        # f.write('\n'.join(labels_sbsg))
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