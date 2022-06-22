import os
import numpy as np


class DataStorage:
    """Allows the user to save data to be opened with gnuplot later."""

    def __init__(self, dir_path: str):
        self.folder = dir_path
        self._check_folder()

    def _check_folder(self) -> None:
        """Check if directory path exists and make it if it doesn't."""

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def _get_file_path(self, filename: str, extension: str) -> str:
        """Get file path."""

        return os.path.join(self.folder, filename) + extension

    def save_numpy(self, filename: str, data: np.ndarray,
                   mode: str = 'w', extension: str = '.txt') -> None:
        """Save a single numpy array into a file."""

        file_id = open(self._get_file_path(filename, extension), mode)
        np.savetxt(file_id, data)
        file_id.close()

    def save_numpies(self, filename: str, data_list: list,
                     mode: str = 'w', extension: str = '.txt') -> None:
        """Save multiples numpy arrays into a file."""

        # Get file id
        file_id = open(self._get_file_path(filename, extension), mode)

        # Get elements from the given list
        for data in data_list:

            # Store elements
            np.savetxt(file_id, data)

            # Add lines to split arrays
            file_id.write("\n\n")

        # Close file
        file_id.close()
