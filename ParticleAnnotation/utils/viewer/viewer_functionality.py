import numpy as np

from ParticleAnnotation.utils.model.utils import correct_coord


def draw_patch_and_scores(
    img: np.ndarray, scores: np.ndarray, patch_corner: np.ndarray, patch_size: int
):
    patch = np.array(())
    tm_score = np.array(())

    patch = img[
        patch_corner[0] : patch_corner[0] + patch_size,
        patch_corner[1] : patch_corner[1] + patch_size,
        patch_corner[2] : patch_corner[2] + patch_size,
    ]

    tm_score = scores[
        :,
        patch_corner[0] : patch_corner[0] + patch_size,
        patch_corner[1] : patch_corner[1] + patch_size,
        patch_corner[2] : patch_corner[2] + patch_size,
    ]

    # if patch.shape != (patch_size, patch_size, patch_size):
    #     df_patch = np.zeros((patch_size, patch_size, patch_size))
    #     df_tm_score = np.zeros((scores.shape[0], patch_size, patch_size, patch_size))

    #     shape_ = patch.shape
    #     df_patch[: shape_[0], : shape_[1], : shape_[2]] = patch
    #     df_tm_score[:, : shape_[0], : shape_[1], : shape_[2]] = tm_score

    #     return df_patch, df_tm_score

    return patch, tm_score


def build_gird_with_particles(
    patch_points: np.ndarray,
    patch_label: np.ndarray,
    patch_corner: tuple,
    img_process: np.ndarray,
    tm_scores: np.ndarray,
    tm_idx: int,
    box_size: int,
    correct=True,
):
    # Particles are in self.patch_points, self.patch_label
    crop_particles = []
    crop_tm_scores = []

    grid_particle_points = np.zeros_like(patch_points)
    grid_particle_labels = patch_label.copy()

    patch_size = box_size * 2
    crop_size = box_size * 4
    gap_size = 2

    for i in patch_points:
        if correct:
            i = correct_coord(np.array(i), patch_corner, True)
        else:
            i = np.array(i)

        i_min = np.max((i - patch_size, [0, 0, 0]), axis=0).astype(np.int16)
        i_max = np.max((i + patch_size, [0, 0, 0]), axis=0).astype(np.int16)

        crop_particle = img_process[
            i_min[0] : i_max[0], i_min[1] : i_max[1], i_min[2] : i_max[2]
        ]
        crop_tm_score = tm_scores[
            tm_idx,
            i_min[0] : i_max[0],
            i_min[1] : i_max[1],
            i_min[2] : i_max[2],
        ]
        crop_particles.append(crop_particle)
        crop_tm_scores.append(crop_tm_score)

    # Get empty grid
    if len(patch_points) < 50:
        grid = 10
    elif len(patch_points) < 100:
        grid = 20
    elif len(patch_points) < 250:
        grid = 30
    else:
        grid = 30

    n_x = np.min((grid, len(patch_points))).astype(np.int8)
    n_y = np.ceil(len(patch_points) / grid).astype(np.int8)

    if len(patch_points) < (grid + 1):
        crop_grid_img = np.zeros(
            (crop_size, crop_size, n_x * crop_size + (n_x - 1) * gap_size),
            dtype=img_process.dtype,
        )
        crop_grid_tm_scores = np.zeros(
            (crop_size, crop_size, n_x * crop_size + (n_x - 1) * gap_size),
            dtype=tm_scores.dtype,
        )
    else:
        crop_grid_img = np.zeros(
            (
                crop_size,
                n_y * crop_size + (n_y - 1) * gap_size,
                n_x * crop_size + (n_x - 1) * gap_size,
            ),
            dtype=img_process.dtype,
        )
        crop_grid_tm_scores = np.zeros(
            (
                crop_size,
                n_y * crop_size + (n_y - 1) * gap_size,
                n_x * crop_size + (n_x - 1) * gap_size,
            ),
            dtype=tm_scores.dtype,
        )

    # Build and display particle grid
    x_min = 0
    y_min = 0
    for idx, (i, j) in enumerate(zip(crop_particles, crop_tm_scores)):
        # Draw particles and place them in the right positions
        particle = np.asarray([patch_size, patch_size, patch_size])
        particle[0] -= 1
        particle[1] += y_min
        particle[2] += x_min
        grid_particle_points[idx, :] = particle

        # Draw image patches and place them in the right positions
        i_z, i_y, i_x = i.shape
        j_z, j_y, j_x = j.shape
        if i.shape == (crop_size, crop_size, crop_size):
            if crop_grid_img.shape[1] == crop_size:
                # Add crops
                crop_grid_img[0:i_z, 0:i_y, x_min : x_min + i_x] = i
                crop_grid_tm_scores[0:j_z, 0:j_y, x_min : x_min + j_x] = j
            else:
                # Add crops
                crop_grid_img[0:i_z, y_min : y_min + i_y, x_min : x_min + i_x] = i
                crop_grid_tm_scores[0:j_z, y_min : y_min + j_y, x_min : x_min + j_x] = j
        else:
            z_offset = int(np.floor((crop_size - i.shape[0]) / 2))

            if crop_grid_img.shape[1] == crop_size:
                # Add crops with z offset for centering
                crop_grid_img[z_offset : i_z + z_offset, 0:i_y, x_min : x_min + i_x] = i
                crop_grid_tm_scores[
                    z_offset : j_z + z_offset, 0:j_y, x_min : x_min + j_x
                ] = j
            else:
                # Add crops
                crop_grid_img[
                    z_offset : i_z + z_offset, y_min : y_min + i_y, x_min : x_min + i_x
                ] = i
                crop_grid_tm_scores[
                    z_offset : j_z + z_offset, y_min : y_min + j_y, x_min : x_min + j_x
                ] = j

        if (idx + 1) % grid == 0 and x_min != 0:
            x_min = 0
            y_min += crop_size + gap_size
        else:
            x_min += crop_size + gap_size

    return (
        crop_grid_img,
        crop_grid_tm_scores,
        grid_particle_points,
        grid_particle_labels,
    )
