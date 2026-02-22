"""
Convert LiDAR point cloud files to vertex (XYZ) and normal images.

Each input point cloud file (.bin, .npy, or .txt) is projected into a
spherical range image, and the resulting vertex (x, y, z coordinates) and
surface-normal images are saved as NumPy arrays (.npy).  Optionally a PNG
visualisation is written alongside each .npy file.

This script is fully self-contained and requires only NumPy (and Pillow when
``--save-png`` is used).  No other packages or local project imports are needed.

Usage
-----
    python convert_pointcloud_to_images.py -i <file_or_dir> [<file_or_dir> ...]
                                           -o <output_dir>
                                           [--H 64] [--W 1024]
                                           [--fov-up 3.0] [--fov-down -25.0]
                                           [--min-depth 1.0] [--max-depth 80.0]
                                           [--save-png]

Output
------
For every input scan ``<name>.bin`` (or .npy / .txt) two files are written to
``<output_dir>``:

* ``<name>_vertex.npy``  – float32 array of shape (H, W, 3) containing the
  projected x/y/z coordinates (vertex image).
* ``<name>_normal.npy``  – float32 array of shape (H, W, 3) containing the
  surface-normal vectors (nx, ny, nz) normalised to unit length.

When ``--save-png`` is given, additional PNG visualisations are saved:

* ``<name>_vertex.png``  – absolute-value of the vertex image, normalised to
  [0, 255].
* ``<name>_normal.png``  – normal image mapped from [-1, 1] to [0, 255].
"""

import argparse
import glob
import os

import numpy as np


# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

EXTENSIONS_SCAN = ['.bin', '.txt', '.npy']


# ---------------------------------------------------------------------------
# Point-cloud file loaders
# ---------------------------------------------------------------------------

def _load_velo_scan(filename):
    """Load a Velodyne scan file and return an (N, 4) float32 array (x,y,z,r)."""
    if filename.endswith('.bin'):
        scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    elif filename.endswith('.npy'):
        scan = np.load(filename).astype(np.float32)
    elif filename.endswith('.txt'):
        scan = np.genfromtxt(filename, dtype=np.float32)
    else:
        raise RuntimeError("Unsupported file extension: {}".format(filename))
    return scan


# ---------------------------------------------------------------------------
# Spherical range projection
# ---------------------------------------------------------------------------

def _range_projection(points, remissions, H, W, fov_up, fov_down, min_depth, max_depth):
    """Project 3-D points into a spherical range image.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    remissions : ndarray, shape (N,)
    H, W : int – image height / width
    fov_up, fov_down : float – vertical FOV limits in **degrees**
    min_depth, max_depth : float – depth clipping range in metres

    Returns
    -------
    proj_xyz : ndarray, shape (H, W, 3)  – vertex image
    proj_range : ndarray, shape (H, W)   – range image
    proj_remission : ndarray, shape (H, W)
    """
    fov_up_rad = fov_up / 180.0 * np.pi
    fov_down_rad = fov_down / 180.0 * np.pi
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    depth = np.linalg.norm(points, 2, axis=1)

    # Filter points outside the depth range
    valid = (depth >= min_depth) & (depth <= max_depth)
    points = points[valid]
    remissions = remissions[valid]
    depth = depth[valid]

    scan_x, scan_y, scan_z = points[:, 0], points[:, 1], points[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    proj_x = 0.5 * (yaw / np.pi + 1.0) * W
    proj_y = (1.0 - (pitch + abs(fov_down_rad)) / fov) * H

    proj_x = np.clip(np.floor(proj_x).astype(np.int32), 0, W - 1)
    proj_y = np.clip(np.floor(proj_y).astype(np.int32), 0, H - 1)

    # Paint in decreasing-depth order so closer points overwrite farther ones
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    points = points[order]
    remissions = remissions[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    proj_xyz = np.zeros((H, W, 3), dtype=np.float32)
    proj_range = np.zeros((H, W), dtype=np.float32)
    proj_remission = np.zeros((H, W), dtype=np.float32)

    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    proj_remission[proj_y, proj_x] = remissions

    return proj_xyz, proj_range, proj_remission


# ---------------------------------------------------------------------------
# Normal estimation
# ---------------------------------------------------------------------------

def _normal_projection(proj_xyz, proj_range):
    """Compute a surface-normal image from a vertex + range image.

    Uses weighted cross-products of 4 neighbouring difference vectors.

    Parameters
    ----------
    proj_xyz : ndarray, shape (H, W, 3)
    proj_range : ndarray, shape (H, W)

    Returns
    -------
    proj_normal : ndarray, shape (H, W, 3) – unit-length normals
    """
    img = np.dstack((proj_xyz, proj_range))

    def calc_weights(x, alpha=-0.8):
        return np.exp(alpha * np.abs(x))

    diff_vertical = img[:-1, :, :] - img[1:, :, :]
    diff_horizontal = img[:, :-1, :] - img[:, 1:, :]

    x_diff_top = diff_vertical[:-1, 1:-1, :]
    x_diff_bottom = -diff_vertical[1:, 1:-1, :]
    x_diff_left = diff_horizontal[1:-1, :-1, :]
    x_diff_right = -diff_horizontal[1:-1, 1:, :]

    x_range_diffs = np.stack(
        (x_diff_top[:, :, -1], x_diff_left[:, :, -1],
         x_diff_bottom[:, :, -1], x_diff_right[:, :, -1]),
        axis=2,
    )
    weights = calc_weights(x_range_diffs)

    x_norm_tl = np.cross(weights[..., 0:1] * x_diff_top[..., :3],
                         weights[..., 1:2] * x_diff_left[..., :3])
    x_norm_lb = np.cross(weights[..., 1, None] * x_diff_left[..., :3],
                         weights[..., 2, None] * x_diff_bottom[..., :3])
    x_norm_br = np.cross(weights[..., 2, None] * x_diff_bottom[..., :3],
                         weights[..., 3, None] * x_diff_right[..., :3])
    x_norm_rt = np.cross(weights[..., 3, None] * x_diff_right[..., :3],
                         weights[..., 0, None] * x_diff_top[..., :3])

    proj_normal = np.sum(
        np.stack((x_norm_tl, x_norm_lb, x_norm_br, x_norm_rt)), axis=0
    )
    proj_normal /= (np.linalg.norm(proj_normal, axis=2, keepdims=True) + 1e-8)
    proj_normal = np.pad(proj_normal, ((1, 1), (1, 1), (0, 0)))
    return proj_normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_scan_files(paths):
    """Return a sorted list of point-cloud file paths from *paths*.

    Each entry in *paths* may be a single file or a directory.  Directories
    are searched (non-recursively) for files whose extension is in
    ``EXTENSIONS_SCAN``.
    """
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for ext in EXTENSIONS_SCAN:
                files.extend(glob.glob(os.path.join(p, "*{}".format(ext))))
        else:
            raise FileNotFoundError("Path not found: {}".format(p))
    return sorted(set(files))


def _save_png(array, path):
    """Save a (H, W, 3) float32 array as a uint8 PNG file."""
    from PIL import Image  # imported lazily – only needed with --save-png
    arr = np.clip(array, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_scan(scan_file, output_dir, H, W, fov_up, fov_down,
                 min_depth, max_depth, save_png=False):
    """Project one LiDAR scan file and write vertex / normal images.

    Parameters
    ----------
    scan_file : str
        Path to the input point-cloud file (.bin, .npy, or .txt).
    output_dir : str
        Directory where output files will be written.
    H, W : int – projection image height / width
    fov_up, fov_down : float – vertical FOV in degrees
    min_depth, max_depth : float – depth clipping in metres
    save_png : bool
        When ``True``, also write PNG visualisations.
    """
    scan = _load_velo_scan(scan_file)
    points = scan[:, :3]
    remissions = scan[:, 3] if scan.shape[1] > 3 else np.zeros(len(scan), dtype=np.float32)

    proj_xyz, proj_range, _ = _range_projection(
        points, remissions, H, W, fov_up, fov_down, min_depth, max_depth
    )
    proj_normal = _normal_projection(proj_xyz, proj_range)

    base = os.path.splitext(os.path.basename(scan_file))[0]

    # --- vertex image (H, W, 3) ---
    vertex_path = os.path.join(output_dir, "{}_vertex.npy".format(base))
    np.save(vertex_path, proj_xyz)

    # --- normal image (H, W, 3) ---
    normal_path = os.path.join(output_dir, "{}_normal.npy".format(base))
    np.save(normal_path, proj_normal)

    if save_png:
        # Vertex: map absolute values to [0, 1]
        v_abs = np.abs(proj_xyz)
        v_max = v_abs.max()
        v_vis = v_abs / v_max if v_max > 0 else v_abs
        _save_png(v_vis, os.path.join(output_dir, "{}_vertex.png".format(base)))

        # Normal: map from [-1, 1] to [0, 1]
        n_vis = (proj_normal + 1.0) / 2.0
        _save_png(n_vis, os.path.join(output_dir, "{}_normal.png".format(base)))

    return vertex_path, normal_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args):
    scan_files = _collect_scan_files(args["input"])

    if not scan_files:
        print("No point-cloud files found for the given input paths.")
        return

    os.makedirs(args["output"], exist_ok=True)

    print("Converting {} scan file(s) → {}".format(len(scan_files), args["output"]))
    for scan_file in scan_files:
        vertex_path, normal_path = convert_scan(
            scan_file,
            args["output"],
            H=args["H"],
            W=args["W"],
            fov_up=args["fov_up"],
            fov_down=args["fov_down"],
            min_depth=args["min_depth"],
            max_depth=args["max_depth"],
            save_png=args["save_png"],
        )
        print("  {} → vertex: {}, normal: {}".format(
            os.path.basename(scan_file),
            os.path.basename(vertex_path),
            os.path.basename(normal_path),
        ))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LiDAR point cloud files to vertex and normal images."
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="One or more point-cloud files (.bin/.npy/.txt) or directories.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory where .npy (and optional .png) files are saved.",
    )
    parser.add_argument("--H", type=int, default=64,
                        help="Height of the projection image (default: 64).")
    parser.add_argument("--W", type=int, default=1024,
                        help="Width of the projection image (default: 1024).")
    parser.add_argument("--fov-up", type=float, default=3.0, dest="fov_up",
                        help="Vertical field of view upward in degrees (default: 3.0).")
    parser.add_argument("--fov-down", type=float, default=-25.0, dest="fov_down",
                        help="Vertical field of view downward in degrees (default: -25.0).")
    parser.add_argument("--min-depth", type=float, default=1.0, dest="min_depth",
                        help="Minimum valid depth in metres (default: 1.0).")
    parser.add_argument("--max-depth", type=float, default=80.0, dest="max_depth",
                        help="Maximum valid depth in metres (default: 80.0).")
    parser.add_argument("--save-png", action="store_true",
                        help="Also save PNG visualisations of the output images.")

    args = vars(parser.parse_args())
    main(args)
