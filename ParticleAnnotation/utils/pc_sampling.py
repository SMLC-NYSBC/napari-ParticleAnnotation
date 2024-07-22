from typing import Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import KDTree


class DownSampling:
    """
    Base down sampling wrapper
    """

    def __init__(self, voxel=None, threshold=None, labels=True, KNN=False):
        if voxel is None:
            self.sample = threshold
        else:
            self.sample = voxel

        # If true downs sample with class ids.
        # Expect [ID x X x Y x (Z)] [[N, 3] or [N, 4]]
        self.labels = labels
        self.KNN = KNN

    @staticmethod
    def pc_down_sample(
        coord: np.ndarray,
        rgb=None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if rgb is not None:
            return coord, rgb
        return coord

    def __call__(
        self,
        coord: Optional[np.ndarray] = list,
        rgb: Optional[Union[np.ndarray, list]] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Compute voxel down sampling for entire point cloud at once or if coord is a list
        compute voxel down sampling for each list index.
        """
        ds_pc = []
        ds_rgb = []

        # Assert correct data structure
        if self.labels:
            if coord.shape[1] not in [3, 4]:
                return
        else:
            if coord.shape[1] not in [2, 3]:
                return

        """Down-sample each instance from list and combine"""
        if isinstance(coord, list):
            # Assert if RGB are of the same structure as coord
            if rgb is not None and not isinstance(rgb, list):
                return

            # Down sample
            id = 0
            for idx, i in enumerate(coord):
                coord_df = i
                if self.labels:
                    id = coord_df[0, 0]

                if rgb is not None:
                    rgb_df = rgb[idx]

                    coord_df, rgb_df = self.pc_down_sample(
                        coord=coord_df, rgb=rgb_df
                    )
                    ds_rgb.append(rgb_df)
                else:
                    coord_df = self.pc_down_sample(coord=coord_df)

                ds_pc.append(
                    np.hstack((np.repeat(id, len(coord_df)).reshape(-1, 1), coord_df))
                )
                if not self.labels:
                    id += 1
        else:
            """Down-sample entire point cloud at once"""
            if rgb is not None:
                return self.pc_down_sample(coord=coord, rgb=rgb)
            else:
                return self.pc_down_sample(coord=coord)


class VoxelDownSampling(DownSampling):
    """
    Wrapper for down sampling of the point cloud using voxel grid (Based on Open3d library)
    """

    def __init__(self, **kwargs):
        super(VoxelDownSampling, self).__init__(**kwargs)

    def pc_down_sample(
        self, coord: np.ndarray, sampling: float, rgb: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        This function takes a set of 3D points and a voxel size and returns the centroids
        of the voxels in which the points are located.

        Args:
            coord (np.ndarray): A numpy array of shape (N, 3) containing the 3D coordinates
            of the input points.
            rgb (np.ndarray): A numpy array of shape (N, 3) containing RGB values of each point.
            sampling (float): The size of each voxel in each dimension.

        Returns:
            voxel_centers (np.ndarray): A numpy array of shape (M, 3) containing the centroids
            of the voxels in which the points are located where M is the number
            of unique voxels.
        """
        if self.labels:
            coord_label = coord
            coord = coord[:, 1:]

        # Find the grid cell index for each point
        voxel_index = np.floor(coord / sampling).astype(np.int32)

        # Compute the unique set of voxel indices
        unique_voxel_index, inverse_index, voxel_counts = np.unique(
            voxel_index, axis=0, return_inverse=True, return_counts=True
        )

        # Compute the centroids of each voxel
        voxel_centers = np.zeros((len(unique_voxel_index), 3))
        np.add.at(voxel_centers, inverse_index, coord)
        voxel_centers /= voxel_counts[:, np.newaxis]

        # Retrieve ID value for down sampled point cloud
        if self.labels or rgb is not None or self.KNN:
            # Build a KDTree from the voxel_centers
            tree = KDTree(coord)

            # Query the KDTree to find the nearest voxel center for each coord point
            _, nearest_voxel_index = tree.query(voxel_centers)
            nearest_voxel_index = np.concatenate(nearest_voxel_index)

            if self.labels and not self.KNN:
                # Compute the color of the nearest voxel center for each down-sampled point
                voxel_centers = np.hstack(
                    (coord_label[nearest_voxel_index, 0].reshape(-1, 1), voxel_centers)
                )
            elif self.labels and self.KNN:
                # Compute the color of the nearest voxel center for each down-sampled point
                voxel_centers = np.hstack(
                    (
                        coord_label[nearest_voxel_index, 0].reshape(-1, 1),
                        coord[nearest_voxel_index, :],
                    )
                )
            elif not self.labels and self.KNN:
                # Compute the color of the nearest voxel center for each down-sampled point
                voxel_centers = coord[nearest_voxel_index, :]

        if rgb is not None:
            # Compute the color of the nearest voxel center for each down-sampled point
            return voxel_centers, rgb[nearest_voxel_index]
        return voxel_centers
