import torch
from torch.nn import functional as F

from apriori_lab.core.progress import ConsoleProgress, ProgressProtocol


def find_closest_vertex(
    target: torch.Tensor,  # (batch_size, 3) or (3,)
    vertices: torch.Tensor,  # (batch_size, num_vertices, 3) or (num_vertices, 3)
) -> torch.Tensor:
    if target.dim() == 1:
        target = target.unsqueeze(0)  # (1, 3)
        squeeze_batch = True
    else:
        squeeze_batch = False

    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)  # (1, num_vertices, 3)

    target = target.unsqueeze(1)  # (batch_size, 1, 3)
    distances = (target - vertices).norm(dim=-1)  # (batch_size, num_vertices)
    closest = distances.argmin(dim=-1)  # (batch_size,)
    return closest if not squeeze_batch else closest[0]


def create_adjacency_matrix(
    faces: torch.Tensor,  # (num_faces, 3)
    num_vertices: int,
) -> torch.Tensor:  # (num_vertices, num_vertices)
    # Collect edges from faces
    faces_src = faces[:, [0, 1, 2]].reshape(-1)
    faces_dst = faces[:, [1, 2, 0]].reshape(-1)

    # Combine forward and backward edge direction
    edges_src = torch.cat([faces_src, faces_dst])
    edges_dst = torch.cat([faces_dst, faces_src])

    # Build sparce index
    indices = torch.stack([edges_src, edges_dst])  # (2, num_edges)
    values = torch.ones_like(edges_src, dtype=torch.float32)

    return torch.sparse_coo_tensor(
        indices, values, size=(num_vertices, num_vertices)
    ).coalesce()


def cartesian_to_spherical(
    directions: torch.Tensor,  # (batch_size, num_directions, 3) or (num_directions, 3)
) -> torch.Tensor:  # (batch_size, num_directions, 2) or (num_directions, 2)
    if directions.dim() == 2:
        directions = directions.unsqueeze(0)  # (1, num_directions, 3)
        squeeze_batch = True
    else:
        squeeze_batch = False

    x, y, z = directions.unbind(dim=-1)  # (batch_size, num_directions)
    # convert to azimuth [0, 2pi] and elevation [0, pi] for easier binning
    azimuth = (torch.atan2(y, x) + torch.pi) % (2 * torch.pi)  # [0, 2pi]
    elevation = torch.acos(z.clamp(-1, 1))  # [0, pi]
    spherical = torch.stack(
        [azimuth, elevation], dim=-1
    )  # (batch_size, num_directions, 2)

    return spherical if not squeeze_batch else spherical[0]


def assign_to_spherical_bins(
    directions: torch.Tensor,  # (batch_size, num_directions, 3) or (num_directions, 3)
    num_azimuth_bins: int,
    num_elevation_bins: int,
) -> torch.Tensor:  # (batch_size, num_directions) or (num_directions)
    if directions.dim() == 2:
        directions = directions.unsqueeze(0)  # (1, num_directions, 3)
        squeeze_batch = True
    else:
        squeeze_batch = False

    spherical = cartesian_to_spherical(directions)  # (batch_size, num_directions, 2)
    azimuth, elevation = spherical.unbind(dim=-1)  # (batch_size, num_directions)
    azimuth_bins = (
        (azimuth / (2 * torch.pi) * num_azimuth_bins)
        .clamp(max=num_azimuth_bins - 1)
        .long()
    )  # (batch_size, num_directions)
    elevation_bins = (
        (elevation / torch.pi * num_elevation_bins)
        .clamp(max=num_elevation_bins - 1)
        .long()
    )  # (batch_size, num_directions)
    spherical_bins = (
        azimuth_bins * num_elevation_bins + elevation_bins
    )  # (batch_size, num_directions)

    return spherical_bins if not squeeze_batch else spherical_bins[0]


def find_surface_hull(
    target_vertex: torch.Tensor,  # (batch_size, 3) or (3,)
    vertices: torch.Tensor,  # (num_vertices, 3)
    vertices_bins: torch.Tensor,  # (batch_size, num_vertices) or (num_vertices,)
    vertices_select_mask: torch.Tensor,  #  (num_vertices, ) or (batch_size, num_vertices)
    num_bins: int,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # vertices indices and distances, (batch_size, num_bins) or (num_bins,)
    if target_vertex.dim() == 1:
        target_vertex = target_vertex.unsqueeze(0)  # (1, 3)
        squeeze_batch = True
    else:
        squeeze_batch = False

    if vertices_bins.dim() == 1:
        if not squeeze_batch:
            raise ValueError(
                "If vertices_bins is (num_vertices,), target_vertex must be (3,)"
            )
        vertices_bins = vertices_bins.unsqueeze(0)  # (1, num_vertices)

    if vertices_select_mask.dim() == 1:
        if not squeeze_batch:
            raise ValueError(
                "If vertices_select_mask is (num_vertices,), target_vertex must be (3,)"
            )
        vertices_select_mask = vertices_select_mask.unsqueeze(0)  # (1, num_vertices)

    batch_size = target_vertex.shape[0]
    num_vertices = vertices.shape[0]

    # Make matrix (batch, num_bins, num_vertices) to find argmin in vectorized way
    hull_map = (
        torch.arange(0, num_bins, device=vertices.device, dtype=torch.long)
        .view(1, num_bins, 1)
        .expand(batch_size, num_bins, num_vertices)
    )  # (batch_size, num_bins, num_vertices)

    # Expand tensors to match dimensions
    vertices = vertices.unsqueeze(
        0
    )  # (1, num_vertices, 3) -> broadcast to (batch, num_vertices, 3)
    target_vertex = target_vertex.unsqueeze(1)  # (batch, 1, 3)
    vertices_select_mask = vertices_select_mask.unsqueeze(
        1
    )  # (batch, 1, num_vertices))

    # Use distance to target vertex as metric for argmin
    vertices_distances = (
        (vertices - target_vertex)
        .norm(dim=-1)  # (batch, num_vertices)
        .unsqueeze(1)
    )  # (batch, 1, num_vertices) -> broadcast to (batch, num_bins, num_vertices)

    # Create mask to select only vertices in each bin
    hull_mask = (
        vertices_bins.unsqueeze(1) == hull_map
    )  # (batch, 1, num_vertices) -> broadcast to (batch, num_bins, num_vertices)
    hull_distances = torch.full_like(hull_map, float("inf"), dtype=torch.float32)
    hull_distances = vertices_distances.where(
        hull_mask & vertices_select_mask, hull_distances
    )  # (batch, num_bins, num_vertices)

    hull_vertex_indices = hull_distances.argmin(dim=-1)  # (batch, num_bins)
    hull_vertex_distances = (
        torch.gather(hull_distances, 2, hull_vertex_indices.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, num_bins)
    )
    if squeeze_batch:
        return hull_vertex_indices[0], hull_vertex_distances[0]
    return hull_vertex_indices, hull_vertex_distances


def find_surface_convex_hull(
    target_vertices: torch.Tensor,  # (3,) or (batch_size, 3)
    vertices: torch.Tensor,  # (num_vertices, 3)
    faces: torch.Tensor,  # (num_faces, 3)
    max_steps: int = 20,
    num_azimuth_bins: int = 12,
    num_elevation_bins: int = 6,
    convex_bins_ratio: float = 0.7,
    return_all_hulls: bool = True,
    verbose: int = 0,
    progress: ProgressProtocol = None,
) -> torch.Tensor:  # (num_hull_vertices,)
    if target_vertices.dim() == 1:
        target_vertices = target_vertices.unsqueeze(0)  # (1, 3)

    num_bins = num_azimuth_bins * num_elevation_bins
    bins_convex_threshold = int(num_bins * convex_bins_ratio)
    num_vertices = vertices.shape[0]
    batch_size = target_vertices.shape[0]
    device = vertices.device

    # Find directions and assign to spherical bins
    directions = F.normalize(
        vertices.unsqueeze(0) - target_vertices.unsqueeze(1),
        dim=-1,
    )  # (batch_size, num_vertices, 3)
    vertices_bins = assign_to_spherical_bins(
        directions,
        num_azimuth_bins,
        num_elevation_bins,
    )  # (batch_size, num_vertices)

    # Storge for convex hull vertex indices
    convex_hull_vertices = torch.zeros(
        (batch_size, num_bins),
        dtype=torch.long,
        device=device,  # (batch_size, num_bins)
    )
    convex_hull_bins_included = torch.zeros(
        (batch_size, num_bins),
        dtype=torch.bool,
        device=device,  # (batch_size, num_bins)
    )
    convex_hull_found = torch.zeros(
        (batch_size,),
        dtype=torch.bool,
        device=device,  # (batch_size,)
    )

    # Create adjencity matrix for finding neighbors
    adjacency = create_adjacency_matrix(
        faces, num_vertices
    )  # (num_vertices, num_vertices)
    adjacency = adjacency.to(device=device)

    # Initialize neighborhood with closest vertex
    batch_indices = torch.arange(batch_size, device=device)
    closest_vertices = find_closest_vertex(target_vertices, vertices)  # (batch_size,)
    neighborhood = torch.zeros(
        (batch_size, num_vertices),
        dtype=torch.bool,
        device=device,
    )  # (batch_size, num_vertices)
    neighborhood[batch_indices, closest_vertices] = True

    progress = progress or ConsoleProgress()
    task = progress.add_task("Finding convex hull", total=max_steps)

    for step in range(max_steps):
        # Find neighbors
        frontier = torch.sparse.mm(
            adjacency, neighborhood.float().T
        ).T  # (batch_size, num_vertices)
        neighborhood = (frontier > 0) | (neighborhood > 0)

        # Find hull vertices
        hull_vertices, hull_distances = find_surface_hull(
            target_vertex=target_vertices,
            vertices=vertices,
            vertices_bins=vertices_bins,
            vertices_select_mask=neighborhood,
            num_bins=num_bins,
        )  # (step_batch_size, num_bins)

        # Find bins included in hull
        hull_bins_included = torch.isfinite(
            hull_distances
        )  # (step_batch_size, num_bins)
        hull_bins_hit = hull_bins_included.sum(dim=-1)  # (step_batch_size, )
        hulls_convex = hull_bins_hit >= bins_convex_threshold  # (step_batch_size, )
        hulls_num_convex = hulls_convex.sum().item()

        # Update result storage
        if hulls_num_convex > 0:
            # Find indices in global batch
            step_mask = ~convex_hull_found  # (step_batch_size,)
            step_hull_indices = step_mask.nonzero(as_tuple=True)[
                0
            ]  # (step_batch_size,)
            step_hull_convex_indices = step_hull_indices[
                hulls_convex
            ]  # (step_num_convex,)

            # Update found convex items
            convex_hull_vertices[step_hull_convex_indices] = hull_vertices[hulls_convex]
            convex_hull_bins_included[step_hull_convex_indices] = hull_bins_included[
                hulls_convex
            ]
            # Mark found items
            convex_hull_found[step_hull_convex_indices] = True

            # Keep only non convex targets in neighborhood
            hulls_non_convex_indices = (~hulls_convex).nonzero(as_tuple=True)[0]
            neighborhood = neighborhood[hulls_non_convex_indices]
            target_vertices = target_vertices[hulls_non_convex_indices]
            vertices_bins = vertices_bins[hulls_non_convex_indices]

        progress.advance(task)

        if verbose == 2 and convex_hull_found.all():
            progress.print(rf"✅ All hulls found by step {step}.")
            progress.advance(task, max_steps - step - 1)
            break

        if verbose == 3:
            remaining = (~convex_hull_found).sum().item()
            mean_coverage = (hull_bins_hit.float() / num_bins).mean().item()
            progress.print(
                rf"\[Step {step:02d}] new convex: {hulls_num_convex}"
                rf", remaining: {remaining}, avg coverage: {mean_coverage:.2f}"
            )

    if return_all_hulls and not convex_hull_found.all():
        # Add remaining hulls to result
        # Find indices in global batch
        step_mask = ~convex_hull_found  # (step_batch_size,)
        step_hull_indices = step_mask.nonzero(as_tuple=True)[0]  # (step_batch_size,)

        convex_hull_vertices[step_hull_indices] = hull_vertices
        convex_hull_bins_included[step_hull_indices] = hull_bins_included

    return [
        convex_hull_vertices[i, convex_hull_bins_included[i]] for i in range(batch_size)
    ]
