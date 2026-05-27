import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev

from segment import segment_fossil


NUM_POINTS = 300
NUM_LAYERS = 40
MAX_HEIGHT = 60


def generate_fossil_mesh(img_path):

    result = segment_fossil(img_path)

    contour = result["contour"]

    contour = contour.squeeze().astype(np.float32)

    # =====================================================
    # Smooth contour
    # =====================================================

    x = contour[:, 0]
    y = contour[:, 1]

    tck, u = splprep([x, y], s=5.0, per=True)

    u_new = np.linspace(0, 1, NUM_POINTS)

    x_new, y_new = splev(u_new, tck)

    smooth_contour = np.stack([x_new, y_new], axis=1)

    # =====================================================
    # Generate shell layers
    # =====================================================

    center = smooth_contour.mean(axis=0)

    vertices = []
    triangles = []

    for layer in range(NUM_LAYERS):

        z = (layer / NUM_LAYERS) * MAX_HEIGHT

        scale = 1.0 - 0.5 * (layer / NUM_LAYERS)

        for pt in smooth_contour:

            direction = pt - center

            new_pt = center + direction * scale

            vertices.append([
                new_pt[0],
                new_pt[1],
                z
            ])

    # =====================================================
    # Mesh connectivity
    # =====================================================

    for layer in range(NUM_LAYERS - 1):

        for i in range(NUM_POINTS - 1):

            curr = layer * NUM_POINTS + i
            next_i = curr + 1

            upper = curr + NUM_POINTS
            upper_next = upper + 1

            triangles.append([curr, next_i, upper])
            triangles.append([next_i, upper_next, upper])

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.compute_vertex_normals()

    mesh = mesh.filter_smooth_simple(
        number_of_iterations=5
    )

    return mesh