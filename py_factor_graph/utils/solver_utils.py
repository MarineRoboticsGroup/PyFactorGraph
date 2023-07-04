from typing import Dict, Optional, Tuple, List
import pickle
from os.path import isfile, dirname, isdir
from os import makedirs
import numpy as np
import attr

from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_quat_from_rotation_matrix,
    get_translation_from_transformation_matrix,
    _check_transformation_matrix,
)
from py_factor_graph.utils.logging_utils import logger


@attr.s(frozen=True)
class VariableValues:
    dim: int = attr.ib(validator=attr.validators.instance_of(int))
    poses: Dict[str, np.ndarray] = attr.ib()
    landmarks: Dict[str, np.ndarray] = attr.ib()
    distances: Optional[Dict[Tuple[str, str], np.ndarray]] = attr.ib(default=None)

    @dim.validator
    def _check_dim(self, attribute, value: int):
        assert value in (2, 3)

    @poses.validator
    def _check_poses(self, attribute, value: Dict[str, np.ndarray]):
        for pose in value.values():
            _check_transformation_matrix(pose, dim=self.dim)

    @landmarks.validator
    def _check_landmarks(self, attribute, value: Dict[str, np.ndarray]):
        for landmark in value.values():
            assert landmark.shape == (self.dim,)

    @distances.validator
    def _check_distances(
        self, attribute, value: Optional[Dict[Tuple[str, str], np.ndarray]]
    ):
        if value is not None:
            for distance in value.values():
                assert distance.shape in [
                    (1,),
                    (self.dim,),
                ], f"Expected shape ({self.dim},) or (1,) but got {distance.shape} for distance"

    @property
    def rotations_theta(self) -> Dict[str, float]:
        return {
            key: get_theta_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }

    @property
    def rotations_matrix(self) -> Dict[str, np.ndarray]:
        return {
            key: get_rotation_matrix_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }

    @property
    def rotations_quat(self) -> Dict[str, np.ndarray]:
        return {
            key: get_quat_from_rotation_matrix(value)
            for key, value in self.rotations_matrix.items()
        }

    @property
    def translations(self) -> Dict[str, np.ndarray]:
        trans_vals = {
            key: get_translation_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }
        landmark_trans_vals = {key: value for key, value in self.landmarks.items()}
        trans_vals.update(landmark_trans_vals)
        return trans_vals

    @property
    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the x/y limits of the poses and landmarks

        Returns:
            (xmin, xmax), (ymin, ymax)
        """
        translations = self.translations.values()
        trans_as_array = np.array(list(translations))
        nrows, ncols = trans_as_array.shape
        assert ncols == self.dim
        max_vals = np.max(trans_as_array, axis=0)
        min_vals = np.min(trans_as_array, axis=0)
        return (min_vals[0], max_vals[0]), (min_vals[1], max_vals[1])


@attr.s(frozen=True)
class SolverResults:
    variables: VariableValues = attr.ib()
    total_time: float = attr.ib()
    solved: bool = attr.ib()
    pose_chain_names: Optional[list] = attr.ib(default=None)  # Default [[str]]
    solver_cost: Optional[float] = attr.ib(default=None)

    @property
    def dim(self) -> int:
        return self.variables.dim

    @property
    def poses(self):
        return self.variables.poses

    @property
    def translations(self):
        return self.variables.translations

    @property
    def rotations_quat(self):
        return self.variables.rotations_quat

    @property
    def rotations_theta(self):
        return self.variables.rotations_theta

    @property
    def landmarks(self):
        return self.variables.landmarks

    @property
    def distances(self):
        return self.variables.distances

    @property
    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the x/y limits of the poses and landmarks

        Returns:
            (xmin, xmax), (ymin, ymax)
        """
        return self.variables.limits


def save_results_to_file(
    solved_results: SolverResults,
    solved_cost: float,
    solve_success: bool,
    filepath: str,
):
    """
    Saves the results to a file

    Args:
        solved_results: The results of the solver
        solved_cost: The cost of the solved results
        solve_success: Whether the solver was successful
        filepath: The path to save the results to
    """
    data_dir = dirname(filepath)
    if not isdir(data_dir):
        makedirs(data_dir)

    if filepath.endswith(".pickle") or filepath.endswith(".pkl"):
        pickle_file = open(filepath, "wb")
        pickle.dump(solved_results, pickle_file)
        solve_info = {
            "success": solve_success,
            "optimal_cost": solved_cost,
        }
        pickle.dump(solve_info, pickle_file)
        pickle_file.close()

    elif filepath.endswith(".txt"):
        raise NotImplementedError(
            "Saving to txt not implemented yet since allowing for 3D"
        )
        with open(filepath, "w") as f:
            translations = solved_results.translations
            rot_thetas = solved_results.rotations_theta
            for pose_key in translations.keys():
                trans_solve = translations[pose_key]
                theta_solve = rot_thetas[pose_key]

                trans_string = np.array2string(
                    trans_solve, precision=1, floatmode="fixed"
                )
                status = (
                    f"State {pose_key}"
                    + f" | Translation: {trans_string}"
                    + f" | Rotation: {theta_solve:.2f}\n"
                )
                f.write(status)

            landmarks = solved_results.landmarks
            for landmark_key in landmarks.keys():
                landmark_solve = landmarks[landmark_key]

                landmark_string = np.array2string(
                    landmark_solve, precision=1, floatmode="fixed"
                )
                status = (
                    f"State {landmark_key}" + f" | Translation: {landmark_string}\n"
                )
                f.write(status)

            f.write(f"Is optimization successful? {solve_success}\n")
            f.write(f"optimal cost: {solved_cost}")

    # Outputs each posechain as a separate file with timestamp in TUM format
    elif filepath.endswith(".tum"):
        save_to_tum(solved_results, filepath)
    else:
        raise ValueError(
            f"The file extension {filepath.split('.')[-1]} is not supported. "
        )

    logger.debug(f"Results saved to: {filepath}\n")


def save_to_tum(
    solved_results: SolverResults,
    filepath: str,
    strip_extension: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Saves a given set of solver results to a number of TUM files, with one
    for each pose chain in the results.

    Args:
        solved_results (SolverResults): [description]
        filepath (str): the path to save the results to. The final files will
        have the pose chain letter appended to the end to indicate which pose chain.
        strip_extension (bool, optional): Whether to strip the file extension
        and replace with ".tum". This should be set to true if the file
        extension is not already ".tum". Defaults to False.

    Returns:
        List[str]: The list of filepaths that the results were saved to.
    """
    assert (
        solved_results.pose_chain_names is not None
    ), "Pose_chain_names must be provided for multi robot trajectories"
    acceptable_pose_chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".replace("L", "")
    # TODO: Add support for exporting without pose_chain_names

    save_files = []
    for pose_chain in solved_results.pose_chain_names:
        if len(pose_chain) == 0:
            continue
        pose_chain_letter = pose_chain[0][0]  # Get first letter of first pose in chain
        assert (
            pose_chain_letter in acceptable_pose_chain_letters
        ), "Pose chain letter must be uppercase letter and not L"

        # Removes extension from filepath to add tum extension
        if strip_extension:
            filepath = filepath.split(".")[0] + ".tum"

        assert filepath.endswith(".tum"), "File extension must be .tum"
        modified_path = filepath.replace(".tum", f"_{pose_chain_letter}.tum")

        # if file already exists we won't write over it
        if verbose and isfile(modified_path) and "/tmp/" not in modified_path:
            logger.warning(f"{modified_path} already exists, overwriting")

        if not isdir(dirname(modified_path)):
            makedirs(dirname(modified_path))

        with open(modified_path, "w") as f:
            translations = solved_results.translations
            quats = solved_results.rotations_quat
            for i, pose_key in enumerate(pose_chain):
                trans_solve = translations[pose_key]
                if len(trans_solve) == 2:
                    tx, ty = trans_solve
                    tz = 0.0
                elif len(trans_solve) == 3:
                    tx, ty, tz = trans_solve
                else:
                    raise ValueError(
                        f"Solved for translation of wrong dimension {len(trans_solve)}"
                    )

                quat_solve = quats[pose_key]
                qx, qy, qz, qw = quat_solve
                # TODO: Add actual timestamps
                f.write(f"{i} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

        if verbose and "/tmp/" not in modified_path:
            logger.info(f"Wrote: {modified_path}")
        save_files.append(modified_path)

    return save_files


def load_custom_init_file(file_path: str) -> VariableValues:
    """Loads the custom init file. Is either a pickled VariableValues object
    or a pickled SolverResults object.

    Args:
        file_path (str): path to the custom init file
    """

    assert isfile(file_path), f"File {file_path} does not exist"
    assert file_path.endswith(".pickle") or file_path.endswith(
        ".pkl"
    ), f"File {file_path} must end with '.pickle' or '.pkl'"

    logger.info(f"Loading custom init file: {file_path}")
    with open(file_path, "rb") as f:
        init_dict = pickle.load(f)
        if isinstance(init_dict, SolverResults):
            return init_dict.variables
        elif isinstance(init_dict, VariableValues):
            return init_dict
        else:
            raise ValueError(f"Unknown type: {type(init_dict)}")


def load_pickled_solution(pickled_solution_path: str) -> SolverResults:
    with open(pickled_solution_path, "rb") as f:
        return pickle.load(f)
