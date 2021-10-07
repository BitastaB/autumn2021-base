import argparse
import glob

import graphviz
import numpy as np
from sklearn import tree

from Base.bpReadWrite import ReadWrite
import generate_data


def generate_dataset():  # TODO: based on number of boxes in input file
    generate_data.main()


def generate_decision_tree(box_count):
    X = []
    for file in glob.glob("./test_data/data_*"):
        state = ReadWrite.read_state(path=file)
        boxes = np.empty([box_count * 2])
        for pos, box in enumerate(state.boxes_open):
            boxes[pos * 2 - 1] = box.get_w()
            boxes[pos * 2] = box.get_h()

        X.append(boxes)

    y = np.loadtxt("./test_data/labels", delimiter=", ")

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    # tree.plot_tree(clf)

    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("bin_packaging")

    return clf


def main():
    print("In predict heuristic")
    iFile = args.file
    print(f"input file: {iFile}")
    iState = ReadWrite.read_state(path=iFile)
    box_count = iState.get_box_count()
    print("box count : ", box_count)

    # generateDataset() in test_data TODO:

    # generate decision tree
    clf = generate_decision_tree(box_count)

    # predict heuristic
    boxes = np.empty([box_count * 2])
    for pos, box in enumerate(iState.boxes_open):
        boxes[pos * 2 - 1] = box.get_w()
        boxes[pos * 2] = box.get_h()
        # OR multiply width * height instead of storing individually ?

    heuristic = clf.predict(boxes.reshape(1, -1))
    print("PREDICTED HEURISTIC : ", heuristic)

    return heuristic


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Predict suitable heuristic for input state')
    parser.add_argument('-f', '--file', required=True, type=str, help='Input file location in the format '
                                                                      'path/to/file/filename.extension')
    args = parser.parse_args()
    main()
