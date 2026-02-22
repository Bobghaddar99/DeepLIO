"""
Convert LiDAR point cloud files to vertex (XYZ) and normal images.

Each input point cloud file (.bin, .npy, or .txt) is projected into a
spherical range image, and the resulting vertex (x, y, z coordinates) and
surface-normal images are saved as NumPy arrays (.npy).  Optionally a PNG
visualisation is written alongside each .npy file.

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
import sys

import numpy as np

dname = os.path.dirname(os.path.realpath(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.insert(0, content_dir)

from deeplio.common.laserscan import LaserScan  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_scan_files(paths):
    """Return a sorted list of point-cloud file paths from *paths*.

    Each entry in *paths* may be a single file or a directory.  Directories
    are searched (non-recursively) for files with the extensions supported by
    :class:`~deeplio.common.laserscan.LaserScan`.
    """
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for ext in LaserScan.EXTENSIONS_SCAN:
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

def convert_scan(scan_file, output_dir, scanner, save_png=False):
    """Project one LiDAR scan file and write vertex / normal images.

    Parameters
    ----------
    scan_file : str
        Path to the input point-cloud file (.bin, .npy, or .txt).
    output_dir : str
        Directory where output files will be written.
    scanner : LaserScan
        Configured :class:`~deeplio.common.laserscan.LaserScan` instance (will
        be reused across calls for efficiency).
    save_png : bool
        When ``True``, also write PNG visualisations.
    """
    scanner.open_scan(scan_file)
    scanner.do_range_projection()
    scanner.do_normal_projection()

    base = os.path.splitext(os.path.basename(scan_file))[0]

    # --- vertex image (H, W, 3) ---
    vertex_img = scanner.proj_xyz.astype(np.float32)
    vertex_path = os.path.join(output_dir, "{}_vertex.npy".format(base))
    np.save(vertex_path, vertex_img)

    # --- normal image (H, W, 3) ---
    normal_img = scanner.proj_normal.astype(np.float32)
    normal_path = os.path.join(output_dir, "{}_normal.npy".format(base))
    np.save(normal_path, normal_img)

    if save_png:
        # Vertex: map absolute values to [0, 1]
        v_abs = np.abs(vertex_img)
        v_max = v_abs.max()
        v_vis = v_abs / v_max if v_max > 0 else v_abs
        _save_png(v_vis, os.path.join(output_dir, "{}_vertex.png".format(base)))

        # Normal: map from [-1, 1] to [0, 1]
        n_vis = (normal_img + 1.0) / 2.0
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

    scanner = LaserScan(
        project=False,
        H=args["H"],
        W=args["W"],
        fov_up=args["fov_up"],
        fov_down=args["fov_down"],
        min_depth=args["min_depth"],
        max_depth=args["max_depth"],
    )

    print("Converting {} scan file(s) → {}".format(len(scan_files), args["output"]))
    for scan_file in scan_files:
        vertex_path, normal_path = convert_scan(
            scan_file, args["output"], scanner, save_png=args["save_png"]
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
