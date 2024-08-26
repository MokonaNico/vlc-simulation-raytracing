from typing import List
from stl import mesh
from simulator import Triangle


def load_triangles_from_file(filename: str, rho: float = 1) -> List[Triangle]:
    """
    Load a list of triangles that represent a room from a 3D STL file.

    Args:
        rho (int): The surface reflection coefficient for all triangles.
        filename (str): The name of the STL file (without the extension).

    Returns:
        List[Triangle]: A list of Triangle objects representing the room.
    """
    # Load the STL file
    try:
        my_mesh = mesh.Mesh.from_file('../resource/' + filename + '.stl')
    except FileNotFoundError:
        my_mesh = mesh.Mesh.from_file('resource/' + filename + '.stl')

    triangles = []
    for data in my_mesh.data:
        triangles.append(Triangle(data[1], data[0], rho))

    return triangles
