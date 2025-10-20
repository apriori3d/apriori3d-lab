import math
from dataclasses import dataclass
from typing import Any

import torch
from apriori.flow.progress.console import ConsoleProgress, ProgressProtocol
from apriori.flow.progress.rich.utils import create_progress
from apriori.geometry.mesh.barycentrics2d import BarycentricMapper2D
from apriori.geometry.utils import triangle_local_frame
from torch.nn import functional as F


def ray_plane_intersection(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    eps: float = 1e-6,
    min_t: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect rays with a plane.

    Solves for t in: ray_origin + t * ray_dir  ∈  plane { x | (x - plane_point) · n = 0 }.
    Works with broadcasting/batches.

    Args:
      ray_origins: (..., 3) Ray origins O.
      ray_dirs: (..., 3) Ray directions D (need not be unit-length).
      plane_points: (..., 3) Any point V0 on the plane (e.g., a triangle vertex).
      plane_normals: (..., 3) Plane normal n (need not be unit-length if normalize_normal=True).
      eps_parallel: Threshold for |D·n| below which the ray is treated as parallel. Defaults to 1e-9.
      min_t: Minimum acceptable t (e.g., 0 for ray, >0 to ignore hits behind origin or too close). Defaults to 0.0.

    Returns:
      hit: (...,) bool. True where a valid intersection occurs (not parallel, t >= min_t).
      t: (...,) Intersection distance along the ray (O + t D). Undefined where hit=False.
      p: (..., 3) Intersection point. Undefined where hit=False.

    Notes:
      - If rays or planes are batched, standard PyTorch broadcasting applies.
      - For triangles, call this once with (plane_point=V0, plane_normal=tri_normal), then
        project the hit point into a local 2D basis of the plane and compute barycentrics.
    """
    device = plane_normals.device
    zero = torch.tensor(0, dtype=torch.float32, device=device)

    # Make calculation stable by set small values to zero
    plane_normals = F.normalize(plane_normals, dim=-1)
    plane_normals = plane_normals.where(plane_normals.abs() > eps, zero)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    ray_dirs = ray_dirs.where(ray_dirs.abs() > eps, zero)

    # Numerator, distance to plain: (V0 - O)·n
    delta = plane_points - ray_origins
    delta = delta.where(delta.abs() > eps, zero)

    plane_distance = (delta * plane_normals).sum(dim=-1)
    # Denominator, cosine between ray direction and plane normal: D·n
    alpha = (ray_dirs * plane_normals).sum(dim=-1)

    # Parallel if |D·n| is too small.
    not_parallel = alpha.abs() > eps
    t = torch.zeros_like(alpha)
    t[not_parallel] = plane_distance[not_parallel] / alpha[not_parallel]

    # Accept only hits in front of the origin (t >= min_t).
    front = t >= min_t
    hit = not_parallel & front

    # Intersection points
    p = ray_origins + t.unsqueeze(-1) * ray_dirs

    return hit, t, p


def select_closest_face_3d(
    barycentrics: torch.Tensor,
    inside: torch.Tensor,
    *,
    distances: torch.Tensor,
    front_faces: torch.Tensor,
    back_culling: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Select the closest valid triangle for each query based on intersection distances.

    Args:
        barycentrics: Tensor of shape (num_queries, num_triangles, 3).
            Barycentric coordinates of intersection candidates.
        inside: Boolean tensor of shape (num_queries, num_triangles).
            Mask of triangles where the intersection lies inside.
        distances: Tensor of shape (num_queries, num_triangles).
            Ray-triangle intersection distances (e.g., along ray direction).
        front_faces: Boolean tensor of shape (num_queries, num_triangles).
            Mask of triangles facing the ray (for backface culling).
        back_culling: If True, only front-facing triangles are considered.

    Returns:
        Tensor of shape (num_queries, 1) with the selected triangle index
        (argmin of distance among valid candidates).
    """
    if distances.shape != inside.shape:
        raise ValueError(
            "Distances shape does not match query_inside (num_query, num_triangles)."
        )
    if front_faces.shape != inside.shape:
        raise ValueError(
            "Front_faces shape does not match query_inside (num_query, num_triangles)."
        )
    if barycentrics.ndim != 3:
        raise ValueError(
            "Query_barycentrics shape should be (num_query, num_triangles, 3)."
        )
    if barycentrics.shape[:2] != inside.shape:
        raise ValueError(
            "Query_barycentrics shape does not match query_inside (num_query, num_triangles)."
        )

    device = barycentrics.device

    # Mask invalid triangles with +inf distance
    inf_value = torch.tensor(float("inf"), device=device)
    score = distances.masked_fill(~inside, inf_value)
    score = score.masked_fill(distances <= 1e-6, inf_value)

    if back_culling:
        score = score.masked_fill(~front_faces, inf_value)

    # Select closest triangle index per query
    selected_score, selected = score.min(dim=1, keepdim=True)  # (chunk, 1)
    found = selected_score < inf_value

    return found, selected


@dataclass(slots=True, frozen=True)
class RayTriangleIntersectorResult:
    points: torch.Tensor
    distances: torch.Tensor
    barycentrics: torch.Tensor
    ray_hit: torch.Tensor
    ray_to_face: torch.Tensor

    @property
    def distances_hit(self) -> torch.Tensor:
        return self.distances[self.ray_hit]

    @property
    def barycentrics_hit(self) -> torch.Tensor:
        return self.barycentrics[self.ray_hit]

    @property
    def ray_to_face_hit(self) -> torch.Tensor:
        return self.ray_to_face[self.ray_hit]

    @property
    def points_hit(self) -> torch.Tensor:
        return self.points[self.ray_hit]

    def reshape(self, shape: tuple[int, ...]) -> "RayTriangleIntersectorResult":
        return RayTriangleIntersectorResult(
            self.points.reshape(*shape, 3),
            self.distances.reshape(*shape),
            self.barycentrics.reshape(*shape, 3),
            self.ray_hit.reshape(*shape),
            self.ray_to_face.reshape(*shape),
        )


class RayTriangleIntersector:
    def __init__(
        self,
        faces: torch.Tensor,
        vertices: torch.Tensor,
        eps_zero=1e-6,
        eps_in=1e-6,
        query_chunk_size: int = 1000,
    ):
        self.eps_zero = eps_zero
        self.eps_in = eps_in
        self.query_chunk_size = query_chunk_size
        self.faces = faces  # (tri, 3)
        self.vertices = vertices.double()  # (tri, 3)
        self.progress = create_progress("Ray-plane intersection..")

        self.valid: torch.Tensor | None = None
        self.local_frames: torch.Tensor | None = None
        self.edge_local: torch.Tensor | None = None
        self.barycentric_mapper: BarycentricMapper2D | None = None
        self.prepared = False

    def prepare(self):
        # Local orthonormal local_frames (b0, b1, b2) per triangle
        triangles = self.vertices[self.faces]  # (poly, 3, 2)
        self.valid, self.local_frames = triangle_local_frame(triangles)

        # Keep only XY projection (first two axes of local frame)
        local_frames_2d = self.local_frames[..., :2]

        v0_world = triangles[:, 0]  # (poly, 3)
        v1_world = triangles[:, 1]
        v2_world = triangles[:, 2]
        e0_world = v1_world - v0_world  # (poly, 3)
        e1_world = v2_world - v0_world

        # Project world edges into local 2D space
        edge0_local = torch.einsum("nij,ni->nj", local_frames_2d, e0_world)  # (poly, 2)
        edge1_local = torch.einsum("nij,ni->nj", local_frames_2d, e1_world)
        origin_local = torch.zeros_like(edge0_local)

        triangles_local = torch.stack(
            [
                origin_local,
                edge0_local,
                edge1_local,
            ],
            dim=1,
        )  # (tri, 3, 2)

        # Update faces and triangles structures
        faces_flatten = torch.arange(
            0, self.faces.shape[0] * 3, device=self.faces.device
        ).reshape(-1, 3)
        vertices_flatten = triangles_local.reshape(-1, 2)

        # Prepare mapper to use with 2D data
        self.barycentric_mapper = BarycentricMapper2D(
            faces=faces_flatten,
            vertices=vertices_flatten,
            query_chunk_size=self.query_chunk_size,
        )
        self.barycentric_mapper.prepare()
        self.prepared = True

    def __call__(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        back_culling: bool = True,
        progress: ProgressProtocol | None = None,
    ) -> RayTriangleIntersectorResult:
        triangles = self.vertices[self.faces]  # (tri, 3, 2)

        v0_world = triangles[:, 0]  # (poly, 3)
        v1_world = triangles[:, 1]
        v2_world = triangles[:, 2]
        e0_world = v1_world - v0_world  # (poly, 3)
        e1_world = v2_world - v0_world

        # Add new dimension to broadcast along queries
        plane_points_world = triangles[:, 0, :].unsqueeze(0)  # (1, tri, 3)
        plane_normals_world = torch.cross(e0_world, e1_world, dim=-1).unsqueeze(
            0
        )  # (1, tri, 3)

        # Keep only XY projection (first two axes of local frame)
        local_frames_2d = self.local_frames[..., :2]

        ray_origin = ray_origin.double()
        ray_dir = ray_dir.double()

        device = ray_origin.device
        num_tri = self.faces.shape[0]
        num_query = ray_origin.shape[0]
        num_chunks = math.ceil(float(num_query) / self.query_chunk_size)

        # Allocate tensors for the result
        barycentrics = torch.zeros((num_query, 3), dtype=torch.double, device=device)
        distances = torch.zeros((num_query,), dtype=torch.double, device=device)
        ray_hit = torch.zeros((num_query,), dtype=torch.bool, device=device)
        ray_to_face = torch.zeros((num_query,), dtype=torch.long, device=device)

        progress = progress or ConsoleProgress()
        task = progress.add_task("Finding ray intersections", total=num_chunks)

        for i in range(num_chunks):
            start = i * self.query_chunk_size
            end = min((i + 1) * self.query_chunk_size, num_query)
            chunk_size = end - start

            # Chunking rays for processing
            ray_origin_chunk = ray_origin[start:end].unsqueeze(1)  # (chunk, 1, 3)
            ray_dirs_chunk = ray_dir[start:end].unsqueeze(1)  # (chunk, 1, 3)

            # Find intersection point for each ray/triangle combination
            _, chunk_distances, point = ray_plane_intersection(
                ray_origins=ray_origin_chunk,
                ray_dirs=ray_dirs_chunk,
                plane_points=plane_points_world,
                plane_normals=plane_normals_world,
            )

            # Project point into local 2D space
            point_world = point - plane_points_world  # (chunk, tri, 3)
            point_local = torch.einsum(
                "mij,nmi->nmj",
                local_frames_2d,
                point_world,
            )  # (chunk, tri, 2)

            if back_culling:
                front_faces = (
                    (ray_dirs_chunk * plane_normals_world).sum(dim=-1)
                    < 0  # (chunk, tri)
                )
            else:
                front_faces = (
                    torch.tensor([True], device=device)
                    .reshape(1, 1)
                    .expand(chunk_size, num_tri)
                )

            # Find the closest intersected face for each ray.
            # Use barycentrics in 2D to find faces being hit
            # and chunk_distances to select the closest one.
            mapper_result = self.barycentric_mapper(
                point_local,
                selector=select_closest_face_3d,
                distances=chunk_distances,
                front_faces=front_faces,
                back_culling=back_culling,
            )

            chunk_indices = torch.arange(start, end, device=device)
            barycentrics[chunk_indices] = mapper_result.barycentrics
            ray_to_face[chunk_indices] = mapper_result.query_to_face

            query_chunk_distances = torch.gather(
                chunk_distances,
                1,
                mapper_result.query_to_face.unsqueeze(-1),
            )
            distances[chunk_indices] = query_chunk_distances.squeeze()
            ray_hit[chunk_indices] = mapper_result.query_inside
            progress.advance(task)

        ray_triangles = triangles[ray_to_face]
        v0 = ray_triangles[:, 0, :]
        v1 = ray_triangles[:, 1, :]
        v2 = ray_triangles[:, 2, :]
        w0 = barycentrics[:, 0:1]
        w1 = barycentrics[:, 1:2]
        w2 = barycentrics[:, 2:3]
        hit_points = v0 * w0 + v1 * w1 + v2 * w2

        return RayTriangleIntersectorResult(
            hit_points,
            distances,
            barycentrics,
            ray_hit,
            ray_to_face,
        )


@dataclass
class PointInsideTraceInfo:
    ray_origins: torch.Tensor
    ray_dirs: torch.Tensor
    ray_hits: torch.Tensor
    ray_hit_back_faces: torch.Tensor
    ray_distances: torch.Tensor


def detect_points_inside(
    points: torch.Tensor,
    faces: torch.Tensor,
    vertices: torch.Tensor,
    num_rays: int,
    num_rays_min: int | None = None,
    eps: float = 1e-6,
    return_trace_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PointInsideTraceInfo]:
    num_rays_min = num_rays_min if num_rays_min is not None else num_rays
    num_points = points.shape[0]
    device = points.device

    ray_origins = points.unsqueeze(1).expand(
        num_points, num_rays, 3
    )  # (num_keypoints, num_rays, 3)

    ray_dirs = F.normalize(
        torch.randn((num_points, num_rays, 3), dtype=torch.float32, device=device),
        dim=-1,
    )  # (num_keypoints, num_rays, 3)

    # Find closest ray/face intersection
    intersector = RayTriangleIntersector(faces, vertices)
    intersector.prepare()

    result = intersector(
        ray_origins.reshape(-1, 3),  # Reshape tensors for intersection calculation
        ray_dirs.reshape(-1, 3),
        back_culling=False,
    )

    # Reshape result back
    ray_hits = result.ray_hit.view(num_points, num_rays)  # (num_keypoints, num_rays)
    ray_to_face = result.ray_to_face.view(
        num_points, num_rays
    )  # (num_keypoints, num_rays)

    # Determine two conditions whether the keypoint inside the surface:
    # 1. all rays hit faces
    rays_all_hit = ray_hits.sum(dim=-1) == num_rays  # (num_keypoints,)

    # 2. Face normal and ray have opposite direction
    ray_to_face_vertices = vertices[
        faces[ray_to_face]
    ]  # (num_keypoints, num_rays, 3, 3)
    # Compute face normals and determine back faces
    # using the sign of the dot product with ray direction
    v1, v2, v3 = ray_to_face_vertices.unbind(dim=2)  # (num_keypoints, num_rays, 3)

    # Calculate in double precision to get correct result for near orthogonal cases
    e1 = (v2 - v1).double()
    e2 = (v3 - v1).double()
    norm = F.normalize(
        torch.cross(e1, e2, dim=-1)
    ).double()  # (num_keypoints, num_ray, 3)
    flat_batch_size = num_points * num_rays

    zero = torch.tensor(0, dtype=torch.double, device=device)
    safe_norm = norm.where(norm.abs() > eps, zero).double()
    safe_ray_dirs = ray_dirs.where(ray_dirs.abs() > eps, zero).double()

    faces_orientation = (
        torch.bmm(
            safe_norm.view(flat_batch_size, 1, 3),  # clamp for numerical stability
            safe_ray_dirs.view(flat_batch_size, 3, 1),
        ).view(num_points, num_rays)  # (num_keypoints, num_rays)
    )
    rays_hit_back_faces = faces_orientation > -eps  # (num_keypoints, num_rays)
    all_rays_hit_back_faces = (
        rays_hit_back_faces.sum(dim=-1) >= num_rays_min
    )  # (num_keypoints,)

    points_inside = rays_all_hit & all_rays_hit_back_faces

    if return_trace_info:
        return points_inside, PointInsideTraceInfo(
            ray_origins=ray_origins,
            ray_dirs=ray_dirs,
            ray_hits=ray_hits,
            ray_hit_back_faces=rays_hit_back_faces,
            ray_distances=result.distances.reshape(num_points, num_rays),
        )

    return points_inside
