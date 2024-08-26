from typing import List
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.random_vector import *
from src.simulator import Triangle


def plot_points_on_sphere(_points: np.ndarray, _title: str, view, with_sphere: bool = False):
    """
    Plot a table of points on a sphere

    Args:
        _points (np.ndarray): A table of points to be plotted (Nx3)
        _title (str): The title of the plot
        view (tuple): The view of the plot (elevation, azimuth, roll)
        with_sphere (bool): Whether to plot a sphere or not

    Source :
     - https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere
    """
    # Create a hemisphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if with_sphere:
        # Create sphere data
        phi, theta = np.mgrid[0.0:np.pi / 2:100j, 0.0:2.0 * np.pi:100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        # Plot the surface of the sphere
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax.view_init(elev=view[0], azim=view[1], roll=view[2])

    xx = _points[:, 0]
    yy = _points[:, 1]
    zz = _points[:, 2]
    ax.scatter(xx, yy, zz, color="k", s=0.1)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_aspect("equal")
    ax.set_title(_title)
    plt.tight_layout()
    plt.show()


def plot_hist_sphere_projection(_points: np.ndarray, _title: str):
    """
    Plot a histogram of a top projection of points on a sphere

    Args:
        _points (np.ndarray): A table of points to be plotted (Nx3)
        _title (str): The title of the plot
    """
    x = _points[:, 0]
    y = _points[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(x, y)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal")
    ax.set_title(_title)
    plt.show()


def plot_sphere_projection(_points: np.ndarray, _title: str):
    """
    Plot a top and side projection of points on a sphere

    Args:
        _points (np.ndarray): A table of points to be plotted (Nx3)
        _title (str): The title of the plot
    """
    x = _points[:, 0]
    y = _points[:, 1]
    z = _points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x, y, s=0.1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal")
    ax.set_title(_title + " top projection")

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x, z, s=0.1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal")
    ax.set_title(_title + " side projection")
    plt.show()


def plot_random_vectors(gen: int, n: int):
    """
    Generate and plot random vectors using different generation methods.

    Args:
        gen (int): Generation method (1: uniform with angle, 2: uniform in cube, 3: normal)
        n (int): Number of points to generate
    """
    if gen == 1:
        # uniform with angle
        title = "uniform with angle"
        points = generate_points_uniform_angle(n)
    elif gen == 2:
        # uniform in cube
        title = "uniform in cube"
        points = generate_points_uniform_cube(n)
    elif gen == 3:
        # normal
        title = "normal"
        points = generate_points_normal(n)
    else:
        print("Wrong generation mode")
        exit(1)

    plot_points_on_sphere(points, title, (None, None, None))
    plot_points_on_sphere(points, title,  (90, -90, 0))
    plot_hist_sphere_projection(points, title)
    plot_sphere_projection(points, title)


def plot_rays(rays: np.ndarray, view=(None, None, None)):
    """
    Plot rays in 3D space.

    Args:
        rays (np.ndarray): Array of ray directions (Nx3)
        view (tuple): View angles (elevation, azimuth, roll)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=view[0], azim=view[1], roll=view[2])

    # Plot original rays
    for ray in rays:
        ax.plot([0, ray[0]], [0, ray[1]], [0, ray[2]], color='blue', alpha=0.5, linewidth=1)

    # Setting the axes properties
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xticks([])

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()


def plot_triangles_rays(ray_origins: np.ndarray, ray_directions: np.ndarray, triangles: np.ndarray, distances: np.ndarray):
    """
    Plot rays and triangles, and indicate collision points.

    Args:
        ray_origins (np.ndarray): Origins of the rays (Nx3)
        ray_directions (np.ndarray): Directions of the rays (Nx3)
        triangles (np.ndarray): Array of triangles (Mx3x3)
        distances (np.ndarray): Distances to collision points (Nx1)
    """
    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(0, -90, 0)

    # Plot triangles
    for tri in triangles:
        poly3d = [tri]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='blue', linewidths=1, edgecolors='r', alpha=.25))

    # Plot rays
    for origin, direction, distance in zip(ray_origins, ray_directions, distances):
        # Extend the ray to make it visible
        end_point = origin + direction * 0.2
        ax.plot([origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]], 'k-', lw=2)

        # If there's a collision, plot the collision point
        if distance < np.inf:
            collision_point = origin + direction * distance
            ax.scatter(collision_point[0], collision_point[1], collision_point[2], color='orange', s=50)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Show plot
    plt.show()


def plot_triangles(triangles: List[Triangle]):
    """
    Plot a list of triangles in 3D space.

    Args:
        triangles (List[Triangle]): List of Triangle objects to plot
    """
    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot triangles
    for tri in triangles:
        poly3d = [tri.vertices]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='blue', linewidths=1, edgecolors='r', alpha=.25))

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the limits (these can be adjusted as needed)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])

    # Show plot
    plt.show()


def plot_collisions(ray_origins: np.ndarray, ray_directions: np.ndarray, triangles: List[Triangle],
                    closest_tri_indices: np.ndarray, closest_distances: np.ndarray):
    """
    Plot collisions between rays and triangles.

    Args:
        ray_origins (np.ndarray): Origins of the rays (Nx3)
        ray_directions (np.ndarray): Directions of the rays (Nx3)
        triangles (List[Triangle]): List of Triangle objects
        closest_tri_indices (np.ndarray): Indices of closest triangles for each ray (Nx1)
        closest_distances (np.ndarray): Distances to the closest collision points (Nx1)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot triangles
    for tri in triangles:
        poly3d = [tri.vertices]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='blue', linewidths=1, edgecolors='r', alpha=.25))

    # Plot rays and collision points
    for origin, direction, tri_idx, dist in zip(ray_origins, ray_directions, closest_tri_indices, closest_distances):
        if tri_idx != -1:
            # Compute the collision point
            collision_point = origin + direction * dist
            tri = triangles[tri_idx]

            # Plot ray
            end_point = collision_point  # Extend the ray for visualization
            ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]], 'k-', lw=1)

            # Plot collision point
            ax.scatter(collision_point[0], collision_point[1], collision_point[2], color='orange', s=35)

            # Plot normal at collision point
            normal_end = collision_point + tri.normal * 0.2
            ax.plot([collision_point[0], normal_end[0]],
                    [collision_point[1], normal_end[1]],
                    [collision_point[2], normal_end[2]], 'g-', lw=2)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])

    ax.set_zticks([])

    ax.set_proj_type("ortho")
    ax.view_init(elev=90, azim=0)

    # Show plot
    plt.show()


def plot_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, view=(None, None, None), z_axis: str = 'Mean Power', name: str = "default"):
    """
    Plots a 3D surface plot where the height of each cell represents the mean power of rays falling into that cell.

    Args:
        X (np.ndarray): Meshgrid array for X-axis positions.
        Y (np.ndarray): Meshgrid array for Y-axis positions.
        Z (np.ndarray): Array of mean power values for each grid cell.
        view (tuple): View angles (elevation, azimuth, roll)
        z_axis (str): Text on the z axis
        name (str): Name of the plot
    """
    # Plot the surface
    fig =  plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

    # Add color bar
    fig.colorbar(surf, ax=ax)

    # Set labels and title
    ax.view_init(elev=view[0], azim=view[1], roll=view[2])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel(z_axis)

    if view == (90, -90, 0):
        ax.set_zticks([])

    plt.show()


def plot_2d_histogram(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, z_axis: str = 'Puissance moyenne (dBm)', name: str = "default"):
    """
    Plots a 2D histogram (heatmap) where the color intensity of each cell represents
    the mean power of rays falling into that cell.

    Args:
        X (np.ndarray): Meshgrid array for X-axis positions.
        Y (np.ndarray): Meshgrid array for Y-axis positions.
        Z (np.ndarray): Array of mean power values for each grid cell.
        z_axis (str): Text on the z axis
        name (str): Name of the plot
    """
    # Create the 2D histogram
    fig = plt.figure(figsize=(10, 7))

    # Using pcolormesh to plot the heatmap
    heatmap = plt.pcolormesh(X, Y, Z, shading='auto', cmap='coolwarm')

    # Add color bar
    cbar = plt.colorbar(heatmap)
    cbar.set_label(z_axis)

    # Set labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    plt.show()