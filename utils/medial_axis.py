import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree


def compute_medial_axis_pts(vertices, faces, grid_size=32, min_surface_dist=0.02, verbose=False):
    """
    Extract approximate medial axis points without mesh2sdf.

    Samples the mesh surface with trimesh, builds a KDTree, then evaluates the
    unsigned distance-to-surface on a regular grid.  Local maxima of that
    distance field that exceed min_surface_dist are the medial axis candidates.

    Using unsigned distance (rather than signed) avoids all watertightness
    requirements — the NPZ meshes are original (non-watertight) geometry.

    Args:
        vertices        : (V, 3) float — mesh vertices in joint coordinate space.
        faces           : (F, 3) int   — triangle vertex indices.
        grid_size       : grid resolution per axis (32 ~ 1 voxel per 3% of bbox).
        min_surface_dist: minimum distance-to-surface to be a medial candidate.
        verbose         : print diagnostics.

    Returns:
        (N, 3) float of medial axis candidate points in the same coordinate
        space as vertices, or None if extraction fails.
    """
    try:
        import trimesh
    except ImportError:
        if verbose:
            print("[medial_axis] trimesh not installed — returning None")
        return None

    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        surface_pts, _ = trimesh.sample.sample_surface(mesh, 4096)
    except Exception as e:
        if verbose:
            print(f"[medial_axis] mesh sampling failed: {e}")
        return None

    if verbose:
        print(f"[medial_axis] vertices range: {vertices.min(axis=0)} to {vertices.max(axis=0)}")
        print(f"[medial_axis] sampled {len(surface_pts)} surface points")

    tree = KDTree(surface_pts)

    # Build a grid over the bounding box of the mesh
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    axes = [np.linspace(vmin[i], vmax[i], grid_size) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    grid_pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    dists, _ = tree.query(grid_pts)
    dist_grid = dists.reshape(grid_size, grid_size, grid_size)

    if verbose:
        print(f"[medial_axis] distance field range: {dists.min():.4f} to {dists.max():.4f}")

    # Local maxima of the distance field above the structural threshold
    dist_max_filt = maximum_filter(dist_grid, size=3)
    is_local_max = dist_grid == dist_max_filt
    is_structural = dist_grid > min_surface_dist
    medial_mask = is_local_max & is_structural

    grid_idx = np.argwhere(medial_mask)
    if verbose:
        print(f"[medial_axis] structural voxels: {is_structural.sum()}, "
              f"local maxima: {is_local_max.sum()}, "
              f"medial candidates: {len(grid_idx)}")
    if len(grid_idx) == 0:
        return None

    # Convert grid indices back to vertex coordinate space
    pts = np.stack([
        axes[i][grid_idx[:, i]] for i in range(3)
    ], axis=1)
    return pts.astype(np.float32)


def snap_joints_to_medial_axis(joints, medial_pts, max_dist=0.05):
    """
    Snap each predicted joint to the nearest medial axis point within max_dist.

    Args:
        joints     : (N, 3) float — predicted joint positions.
        medial_pts : (M, 3) float — medial axis candidates, or None.
        max_dist   : snap radius in the same units as joints.

    Returns:
        (N, 3) float of refined joint positions.
    """
    if medial_pts is None or len(medial_pts) == 0:
        return joints

    tree = KDTree(medial_pts)
    dists, idxs = tree.query(joints)

    refined = joints.copy()
    snap_mask = dists < max_dist
    refined[snap_mask] = medial_pts[idxs[snap_mask]]
    return refined
