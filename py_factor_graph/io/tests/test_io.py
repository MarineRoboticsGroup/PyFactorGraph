import os

from py_factor_graph.io.pyfg_text import read_from_pyfg_text, save_to_pyfg_text

# get current directory  and directory containing data
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir, "data")

# create temporary folder for saving factor graph to file
tmp_dir = os.path.join(cur_dir, "tmp")
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# read two text files and checks if each line in both files is identical
def _check_file_equality(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            if line1.strip() != line2.strip():
                return False
    return True


def test_pyfg_se3_file() -> None:
    # read factor graph data
    data_file = os.path.join(data_dir, "pyfg_text_se3_test_data.txt")
    factor_graph = read_from_pyfg_text(data_file)

    # write factor graph data
    write_file = os.path.join(tmp_dir, "pyfg_text_se3_test_tmp.txt")
    save_to_pyfg_text(factor_graph, write_file)

    # assert read and write files are equal
    assert _check_file_equality(data_file, write_file)
