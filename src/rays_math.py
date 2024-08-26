from typing import Tuple
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.

    Args:
        v (np.ndarray): A vector to be normalized.

    Returns:
        np.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalize_batch(v: np.ndarray) -> np.ndarray:
    """
    Normalize a batch of vectors.

    Args:
        v (np.ndarray): An array of shape (n, 3) representing n vectors.

    Returns:
        np.ndarray: An array of shape (n, 3) representing the normalized vectors.
    """
    magnitudes = np.linalg.norm(v, axis=1, keepdims=True)
    normalized_vectors = v / magnitudes
    return normalized_vectors


def rotation_matrix_from_vectors(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix that aligns vector v to vector w.

    Args:
        v (np.ndarray): Source vector.
        w (np.ndarray): Target vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    v = normalize(v)
    w = normalize(w)

    # Calculate the cross product and the sine of the angle
    cross_product = np.cross(v, w)
    sin_angle = np.linalg.norm(cross_product)

    # Calculate the dot product and the cosine of the angle
    dot_product = np.dot(v, w)
    cos_angle = dot_product

    # If vectors are collinear or opposite, handle these special cases
    if sin_angle == 0:
        if cos_angle > 0:
            # Vectors are collinear and in the same direction
            return np.eye(3)
        else:
            # Vectors are collinear and in opposite directions
            return -np.eye(3)

    # Normalize the cross product to get the rotation axis
    axis = cross_product / sin_angle

    # Compute the skew-symmetric cross-product matrix of the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Use Rodrigues' rotation formula to compute the rotation matrix
    I = np.eye(3)
    R = I + K * sin_angle + np.dot(K, K) * (1 - cos_angle)

    return R


def align_rays(rays: np.ndarray, from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """
    Align rays from a source direction to a target direction.

    Args:
        rays (np.ndarray): Array of shape (n, 3) representing n rays.
        from_vec (np.ndarray): Source direction vector.
        to_vec (np.ndarray): Target direction vector.

    Returns:
        np.ndarray: Array of shape (n, 3) representing the aligned rays.
    """
    R = rotation_matrix_from_vectors(from_vec, to_vec)
    return np.dot(rays, R.T)


def moller_trumbore_batch(ray_origins: np.ndarray, ray_directions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Perform the Möller–Trumbore intersection algorithm for a batch of rays and triangles.

    Args:
        ray_origins (np.ndarray): Array of shape (n, 3) representing the origins of n rays.
        ray_directions (np.ndarray): Array of shape (n, 3) representing the directions of n rays.
        triangles (np.ndarray): Array of shape (m, 3, 3) representing m triangles.

    Returns:
        np.ndarray: Array of shape (n, m) representing intersection distances.
    """
    EPSILON = 1e-8

    # Extract vertices of triangles
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute determinants
    h = np.cross(ray_directions[:, np.newaxis, :], edge2[np.newaxis, :, :])
    a = np.einsum('ijk,jk->ij', h, edge1)

    # Parallel rays and triangles
    parallel_mask = np.abs(a) < EPSILON

    with np.errstate(divide='ignore', invalid='ignore'):
        f = 1.0 / a
        f[parallel_mask] = np.inf

        # Compute intersection
        s = ray_origins[:, np.newaxis, :] - v0[np.newaxis, :, :]
        u = f * np.einsum('ijk,ijk->ij', s, h)
        u_mask = (u < 0.0) | (u > 1.0)

        q = np.cross(s, edge1[np.newaxis, :, :])
        v = f * np.einsum('ijk,ijk->ij', ray_directions[:, np.newaxis, :], q)
        v_mask = (v < 0.0) | (u + v > 1.0)

        t = f * np.einsum('ijk,ijk->ij', edge2[np.newaxis, :, :], q)
        t_mask = t < EPSILON

    # Combine masks
    mask = parallel_mask | u_mask | v_mask | t_mask

    t[mask] = np.inf

    return t


def find_first_collision(ray_origins: np.ndarray, ray_directions: np.ndarray, triangles: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the first collision of rays with triangles.

    Args:
        ray_origins (np.ndarray): Array of shape (n, 3) representing the origins of n rays.
        ray_directions (np.ndarray): Array of shape (n, 3) representing the directions of n rays.
        triangles (np.ndarray): Array of shape (m, 3, 3) representing m triangles.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - An array of shape (n, ) representing the indices of the closest triangles.
            - An array of shape (n, ) representing the distances to the closest triangles.
    """
    t = moller_trumbore_batch(ray_origins, ray_directions, triangles)

    closest_distances = np.min(t, axis=1)
    closest_tri_indices = np.argmin(t, axis=1)

    # Set indices to -1 where there are no intersections
    closest_tri_indices[closest_distances == np.inf] = -1

    return closest_tri_indices, closest_distances
