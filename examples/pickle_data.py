import os

from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file


def main():
    """
    Main function.
    """
    # get location of data
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cur_dir, "data")

    # read factor graph data
    data_file = os.path.join(data_dir, "factor_graph.pickle")
    factor_graph = parse_pickle_file(data_file)

    # write factor graph data
    write_file = os.path.join(data_dir, "factor_graph_out.pickle")
    factor_graph.save_to_file(write_file)


if __name__ == "__main__":
    main()
