"""
This module contains methods for generating random points in a hemisphere
"""
import numpy as np


def normalize_hemisphere(_points: np.ndarray):
    """
    Make the z coordinate of the points positive and normalize the points.

    Args:
        _points (np.ndarray): Array of points (Nx3)
    """
    # Ensure all z coordinates are positive
    _points[:, 2] = np.abs(_points[:, 2])
    # Calculate the norm of each point
    norm = np.linalg.norm(_points, axis=1)
    # Normalize each coordinate
    _points[:, 0] /= norm
    _points[:, 1] /= norm
    _points[:, 2] /= norm


def generate_points_uniform_angle(num_points: int) -> np.ndarray:
    """
    Generate random points on a hemisphere using uniform angle distribution.
    Points are in the form of Cartesian coordinates.

    Args:
        num_points (int): The number of points to be generated.

    Returns:
        np.ndarray: An array of points (Nx3)
    """
    # Generate angle using uniform distribution
    theta = np.random.uniform(0, np.pi / 2, size=num_points)
    phi = np.random.uniform(0, 2 * np.pi, size=num_points)

    # Convert spherical coordinate to cartesian coordinate
    rpoints = np.ndarray((num_points, 3))
    rpoints[:, 0] = np.sin(theta) * np.cos(phi)
    rpoints[:, 1] = np.sin(theta) * np.sin(phi)
    rpoints[:, 2] = np.cos(theta)
    return rpoints


def generate_points_uniform_cube(num_points: int) -> np.ndarray:
    """
    Generate random points on a hemisphere using a uniform distribution within a cube,
    then normalize the points to lie on the hemisphere.

    Args:
        num_points (int): The number of points to be generated.

    Returns:
        np.ndarray: An array of points (Nx3)
    """
    # Generate random points within a cube
    rpoints = np.random.uniform(low=-1, high=1, size=(num_points, 3))
    # Normalize the points to lie on the hemisphere
    normalize_hemisphere(rpoints)
    return rpoints


def generate_points_normal(num_points: int) -> np.ndarray:
    """
    Generate random points on a hemisphere using normal distribution.

    Args:
        num_points (int): The number of points to be generated.

    Returns:
        np.ndarray: An array of points (Nx3)

    Source:
     - https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
     - https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
    """
    # Generate random points using normal distribution
    rpoints = np.random.normal(size=(num_points, 3))
    # Normalize the points to lie on the hemisphere
    normalize_hemisphere(rpoints)
    return rpoints


def get_angle(_points: np.ndarray) -> np.ndarray:
    """
    Convert points from Cartesian coordinates to spherical coordinates.

    Args:
        _points (np.ndarray): Array of points to be converted (Nx3)

    Returns:
        np.ndarray: Array of angles defined by theta and phi (Nx2)
    """
    # Initialize an array for the angles
    theta_phi = np.empty((_points.shape[0], 2))
    # Compute theta (polar angle)
    theta_phi[:, 0] = np.arccos(_points[:, 2])
    # Compute phi (azimuthal angle)
    theta_phi[:, 1] = np.arctan2(_points[:, 1], _points[:, 0])
    return theta_phi
