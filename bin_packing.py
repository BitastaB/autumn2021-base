import argparse
import predict_heuristic


def main():
    print("I am here")
    predict_heuristic.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Predict suitable heuristic for input state')
    parser.add_argument('-f', '--file', required=True, type=str, help='Input file location in the format '
                                                                      'path/to/file/filename.extension')
    args = parser.parse_args()
    main()
