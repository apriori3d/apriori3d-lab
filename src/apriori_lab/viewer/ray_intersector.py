import torch
import vedo

from apriori_lab.geometry.surface_mapping import RayTriangleIntersector

if __name__ == "__main__":
    vedo.settings.default_backend = "k3d"

    # Set ray to direcrion -z to follow right-hand rule:
    # front faces will have -z normal
    origins = torch.tensor(
        [
            [0, 0, 1],
            [0, 0.5, -1],
        ],
        dtype=torch.float32,
    )
    dirs = torch.tensor(
        [
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    sample = torch.tensor(
        [
            [1, -1, 0],
            [0, 1, 0.5],
            [-1, -1, 0],
        ],
        dtype=torch.float32,
    )
    sample_faces = torch.tensor(
        [
            [0, 1, 2],
        ],
        dtype=torch.long,
    )

    sample_faces_inv = torch.tensor(
        [
            [0, 2, 1],
        ],
        dtype=torch.long,
    )
    vertices = torch.cat(
        [
            sample,
            sample + torch.tensor([0, 0, -1]),
            sample + torch.tensor([0, 1.2, -1.2]),
        ],
        dim=0,
    )
    faces = torch.cat(
        [
            sample_faces,
            sample_faces + 3,
            sample_faces + 6,
        ],
        dim=0,
    )

    intersector = RayTriangleIntersector(faces, vertices)
    intersector.prepare()
    result = intersector(origins, dirs, back_culling=False)

    plt = vedo.Plotter()
    plt += vedo.Mesh([vertices, faces]).wireframe()

    for i in range(origins.shape[0]):
        origin = origins[i]
        dir = dirs[i]
        t = result.distances[i]
        point = result.points[i]

        plt += vedo.Sphere(origin, r=0.05)
        plt += vedo.Line([origin, origin + dir * 2], lw=0.05)
        if result.ray_hit[i].item():
            plt += vedo.Sphere(origin + dir * t, r=0.05, c="green")

    plt.show()
