from typing import List, Tuple
import numpy as np
from src.random_vector import generate_points_normal
from src.rays_math import align_rays, find_first_collision


def compute_meshgrid(received_plane_size: Tuple[int, int, int, int], grid_size: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the X and Y meshgrid for the given hit positions and grid size.

    Args:
        received_plane_size: Tuple containing x and y positions where rays hit
            (x min, x max, y min, y max)
        grid_size: Number of grid cells along each axis. The grid will be grid_size x grid_size.

    Returns:
        X and Y mesh grids, and x and y edges of the bins.
    """
    # Define the bin edges
    x_edges = np.linspace(received_plane_size[0], received_plane_size[1], grid_size + 1)
    y_edges = np.linspace(received_plane_size[2], received_plane_size[3], grid_size + 1)

    # Compute the center of each bin
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Create a meshgrid for the X and Y axes
    X, Y = np.meshgrid(x_centers, y_centers)

    return X, Y, x_edges, y_edges


def compute_mean_power_histogram(hit_positions: np.ndarray, hit_powers: np.ndarray,
                                 x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """
    Computes the mean power histogram for a grid.

    Args:
        hit_positions: Array of shape (n, 2) containing x and y positions where rays hit.
        hit_powers: Array of shape (n, ) containing the power of rays at each hit position.
        x_edges: The edges of the bins along the x-axis.
        y_edges: The edges of the bins along the y-axis.

    Returns:
        Mean power histogram Z.
    """
    x_positions, y_positions = hit_positions[:, 0], hit_positions[:, 1]

    # Compute the 2D histogram for total power
    total_power_hist, _, _ = np.histogram2d(x_positions, y_positions, bins=[x_edges, y_edges], weights=hit_powers)

    # Compute the 2D histogram for counts
    counts_hist, _, _ = np.histogram2d(x_positions, y_positions, bins=[x_edges, y_edges])

    # Avoid division by zero by setting zero counts to one (since zero divided by anything is still zero)
    counts_hist[counts_hist == 0] = 1

    # Compute the mean power
    mean_power_hist = total_power_hist / counts_hist

    return mean_power_hist.T


def compute_median_power_histogram(hit_positions: np.ndarray, hit_powers: np.ndarray,
                                   x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """
    Computes the median power histogram for a grid.

    Args:
        hit_positions: Array of shape (n, 2) containing x and y positions where rays hit.
        hit_powers: Array of shape (n, ) containing the power of rays at each hit position.
        x_edges: The edges of the bins along the x-axis.
        y_edges: The edges of the bins along the y-axis.

    Returns:
        Median power histogram Z.
    """
    x_positions, y_positions = hit_positions[:, 0], hit_positions[:, 1]

    # Initialize an empty list to hold the power values for each bin
    power_bins = [[[] for _ in range(len(y_edges) - 1)] for _ in range(len(x_edges) - 1)]

    # Assign each power value to the appropriate bin
    for i in range(len(hit_powers)):
        x_idx = np.searchsorted(x_edges, x_positions[i], side='right') - 1
        y_idx = np.searchsorted(y_edges, y_positions[i], side='right') - 1
        if 0 <= x_idx < len(x_edges) - 1 and 0 <= y_idx < len(y_edges) - 1:
            power_bins[x_idx][y_idx].append(hit_powers[i])

    # Compute the median for each bin
    median_power_hist = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            if power_bins[i][j]:  # Check if the bin has any values
                median_power_hist[i, j] = np.median(power_bins[i][j])
            else:
                median_power_hist[i, j] = 0  # Assign zero if the bin is empty

    return median_power_hist.T


class Triangle:
    def __init__(self, _vertices: List[np.array], _normal: np.array, _rho: float):
        """
        Initializes a Triangle.

        Args:
            _vertices: List of 3 arrays representing the vertices of the triangle
            _normal: A vector that represents the normal of the triangle
        """
        self.vertices = _vertices
        self.normal = _normal / np.linalg.norm(_normal)
        self.rho = _rho


def get_triangles_values(triangles: List[Triangle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts vertices and normals from a list of triangles.

    Args:
        triangles: List of Triangle objects.

    Returns:
        Tuple containing an array of vertices and an array of normals.
    """
    vertices = []
    normals = []
    rho = []
    for tri in triangles:
        vertices.append(tri.vertices)
        normals.append(tri.normal)
        rho.append(tri.rho)
    return np.array(vertices), np.array(normals), np.array(rho)


class Emitter:
    def __init__(self, _position: np.array, _direction: np.array, _power: float, _angle_emission: float):
        """
            Initializes an Emitter.

            Args:
                _position: Position of the emitter as a 3D array.
                _direction: Direction of the emitter as a 3D array.
                _power: Power of the emitter.
                _angle_emission: Emission angle of the emitter in degrees.
            """
        self.position = _position
        self.direction = _direction / np.linalg.norm(_direction)  # Normalize the normal vector
        self.power = _power
        self.ml = -np.log(2) / np.log(np.cos(np.radians(_angle_emission)))


class Receiver:
    def __init__(self, _height: float, _fov: float, _adet: float, _index: float, _received_plane_size):
        """
        Initializes a Receiver.

        Args:
            _height: Height of the receiver.
            _fov: Field of view of the receiver in degrees.
            _adet: Area of the detector.
            _index: Refractive index.
            _received_plane_size: Size of the received plane as a tuple (x min, x max, y min, y max).
        """
        self.height = _height
        self.fov = _fov
        self.Adet = _adet
        self.G_Con = (_index ** 2) / (np.sin(np.radians(_fov)) ** 2)
        self.received_plane_size = _received_plane_size


class VLCSimulator:
    def __init__(self, _emitters: List[Emitter], _receiver: Receiver, _triangles: List[Triangle]):
        """
        Initializes a VLC Simulator.

        Args:
            _emitters: List of Emitter objects.
            _receiver: Receiver object.
            _triangles: List of Triangle objects.
        """
        self.emitters = _emitters
        self.receiver = _receiver

        vertices, normals, rho = get_triangles_values(_triangles)
        h = _receiver.height
        receiver_vertices = np.array([
            [[-100, 100, h], [-100, -100, h], [100, -100, h]],
            [[-100, 100, h], [100, 100, h], [100, -100, h]]
        ])
        receiver_normals = np.array([
            [0, 0, 1],
            [0, 0, 1]
        ])
        receiver_rho = np.array([0, 0])

        self.triangle_vertices = np.concatenate((receiver_vertices, vertices), axis=0)
        self.triangle_normals = np.concatenate((receiver_normals, normals), axis=0)
        self.triangle_rho = np.concatenate((receiver_rho, rho), axis=0)

    def compute_ray_powers(self, ray_directions, ray_distances, emitter: Emitter) -> np.ndarray:
        """
        Computes the power of rays at their hit positions.

        Args:
            ray_directions: Array of ray direction vectors.
            ray_distances: Array of distances each ray travels before hitting.
            emitter: Emitter object.

        Returns:
            Array of powers received by the rays.
        """
        cos_theta = np.dot(ray_directions, emitter.direction)
        cos_phi = np.dot(-ray_directions, np.array([0, 0, 1]))
        phi = np.degrees(np.arccos(cos_phi))

        radiation_pattern = (emitter.ml + 1) * emitter.power * np.power(cos_theta, emitter.ml) / (2 * np.pi)
        power_received = radiation_pattern * self.receiver.Adet * cos_phi * self.receiver.G_Con / np.power(
            ray_distances, 2)
        power_received[np.where(np.abs(phi) > self.receiver.fov)] = 0
        return power_received

    def compute_nlos_ray_powers(self, ray1_directions, ray2_directions, wall_indices, distance1, distance2,
                                emitter: Emitter) -> np.ndarray:
        """
        Computes the power of rays for non-line-of-sight (NLOS) scenarios

        Args:
            ray1_directions: Directions of the first set of rays before reflection.
            ray2_directions: Directions of the second set of rays after reflection.
            wall_indices: Indices of the walls where reflections occur.
            distance1: Distances traveled by the first set of rays before hitting the wall.
            distance2: Distances traveled by the second set of rays after reflection.
            emitter: Emitter object.

        Returns:
            Array of powers received by the rays after reflection.
        """
        cos_theta = np.dot(ray1_directions, emitter.direction)
        cos_phi = np.dot(-ray2_directions, np.array([0, 0, 1]))
        phi = np.degrees(np.abs(np.arccos(cos_phi)))

        cos_alpha = np.einsum("ij,ij->i", -ray1_directions, self.triangle_normals[wall_indices])
        cos_beta = np.einsum("ij,ij->i", self.triangle_normals[wall_indices], ray2_directions)

        radiation_pattern = (emitter.ml + 1) * np.power(cos_theta, emitter.ml) / (2 * np.square(np.pi))
        power_received = (radiation_pattern * self.receiver.Adet * self.triangle_rho[wall_indices] * cos_alpha *
                          cos_beta * cos_phi / (np.square(distance1) * np.square(distance2)))

        power_received[np.where(phi > self.receiver.fov)] = -1
        return power_received * emitter.power * self.receiver.G_Con

    def compute_reflexion(self, ray1_directions: np.ndarray, distances_1: np.ndarray, tri_indices: np.ndarray,
                          hit_positions: np.ndarray, emitter: Emitter, num_rays: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the reflections of rays

        Args:
            ray1_directions: Directions of the first set of rays before reflection.
            distances_1: Distances traveled by the first set of rays before hitting the wall.
            tri_indices: Indices of the triangles (walls) where reflections occur.
            hit_positions: Positions where the first set of rays hit the wall.
            emitter: Emitter object.
            num_rays: Number of rays reflected from each hit position.

        Returns:
            Tuple containing:
            - New hit positions of the reflected rays.
            - Powers of the reflected rays.
            - Triangle indices for the reflected rays.
        """
        wall_normals = self.triangle_normals[tri_indices]

        ray2_directions = np.zeros((len(ray1_directions) * num_rays, 3))
        ray2_origins = np.repeat(hit_positions, num_rays, axis=0)
        all_tri_indices = np.repeat(tri_indices, num_rays, axis=0)

        for i in range(len(ray1_directions)):
            new_ray = generate_points_normal(num_rays)
            new_rays_aligned = align_rays(new_ray, np.array([0, 0, 1]), wall_normals[i])
            start_idx = i * num_rays
            ray2_directions[start_idx:start_idx + num_rays] = new_rays_aligned

        closest_tri_indices, closest_distances = (
            find_first_collision(ray2_origins, ray2_directions, self.triangle_vertices))

        hit_indices = np.isin(closest_tri_indices, [0, 1])

        # Compute hit positions
        new_hit_positions = (ray2_origins[hit_indices] + ray2_directions[hit_indices] *
                             closest_distances[hit_indices][:, np.newaxis])

        all_tri_indices = all_tri_indices[hit_indices]

        # Calculate the power of each ray at the hit position
        new_ray_powers = self.compute_nlos_ray_powers(
            np.repeat(ray1_directions, num_rays, axis=0)[hit_indices],
            ray2_directions[hit_indices],
            np.repeat(tri_indices, num_rays)[hit_indices],
            np.repeat(distances_1, num_rays)[hit_indices],
            closest_distances[hit_indices], emitter)

        return (new_hit_positions[new_ray_powers >= 0], new_ray_powers[new_ray_powers >= 0],
                all_tri_indices[new_ray_powers >= 0])

    def do_simulation(self, _number_rays: int, _number_batch: int, _number_reflections_rays: int, grid_size: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the VLC simulation.

        Args:
            _number_rays: Number of rays to generate per batch.
            _number_batch: Number of batches of rays to generate.
            _number_reflections_rays: Number of rays that are compute for a specific reflection (0 means no reflection).
            grid_size: Size of the grid for the receiver plane.

        Returns:
            X, Y meshgrid, total mean power over the grid for both los and nlos (first reflection).
        """
        X, Y, x_edges, y_edges = compute_meshgrid(self.receiver.received_plane_size, grid_size)
        total_los_mean_power = np.zeros((grid_size, grid_size))
        total_nlos_mean_power = np.zeros((grid_size, grid_size))

        for emitter in self.emitters:
            los_hit_positions_list = []
            los_ray_powers_list = []
            nlos_hit_positions_list = []
            nlos_ray_powers_list = []
            nlos_wall_indices_list = []

            for batch in range(_number_batch):
                # Generate random ray directions
                ray_directions = generate_points_normal(_number_rays)

                # Align the ray directions with the emitter direction
                ray_directions = align_rays(ray_directions, np.array([0, 0, 1]), emitter.direction)

                ray_origins = np.array([emitter.position] * _number_rays)

                # Find the first collision of the rays with the triangles
                closest_tri_indices, closest_distances = (
                    find_first_collision(ray_origins, ray_directions, self.triangle_vertices))

                los_indices = np.isin(closest_tri_indices, [0, 1])

                # Compute hit positions
                hit_positions = ray_origins + ray_directions * closest_distances[:, np.newaxis]
                los_hit_positions_list.append(hit_positions[los_indices])

                # Calculate the power of each ray at the hit position
                ray_powers = self.compute_ray_powers(ray_directions, closest_distances, emitter)
                los_ray_powers_list.append(ray_powers[los_indices])

                if _number_reflections_rays > 0:
                    nlos_indices = closest_tri_indices > 1
                    nlos_hit_positions, nlos_ray_powers, nlos_wall_indices = self.compute_reflexion(
                        ray_directions[nlos_indices],
                        closest_distances[nlos_indices],
                        closest_tri_indices[nlos_indices],
                        hit_positions[nlos_indices],
                        emitter,
                        _number_reflections_rays)
                    nlos_hit_positions_list.append(nlos_hit_positions)
                    nlos_ray_powers_list.append(nlos_ray_powers)
                    nlos_wall_indices_list.append(nlos_wall_indices)

            los_hit_positions = np.vstack(los_hit_positions_list)
            los_ray_powers = np.hstack(los_ray_powers_list)
            los_mean_power = compute_mean_power_histogram(los_hit_positions[:, :2],
                                                          los_ray_powers, x_edges, y_edges)
            total_los_mean_power += los_mean_power

            if _number_reflections_rays > 0:
                nlos_hit_positions = np.vstack(nlos_hit_positions_list)
                nlos_ray_powers = np.hstack(nlos_ray_powers_list)
                nlos_wall_indices = np.hstack(nlos_wall_indices_list)

                for i in range(2, len(self.triangle_normals), 2):
                    mask = np.logical_or(nlos_wall_indices == i, nlos_wall_indices == i+1)
                    nlos_mean_power = compute_median_power_histogram(nlos_hit_positions[mask][:, :2],
                                                                     nlos_ray_powers[mask],
                                                                     x_edges, y_edges)
                    total_nlos_mean_power += nlos_mean_power

        return X, Y, total_los_mean_power, total_nlos_mean_power
