import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from apriori.flow.progress import ConsoleProgress, ProgressProtocol
from apriori.geometry.utils import area2d, make_faces_ccw, swap_with_mask


def select_any_inside(
    barycentrics: torch.Tensor,
    inside: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Select one triangle index per query using min(|w0|+|w1|+|w2|) among inside candidates.

    If a query has no inside triangles, an arbitrary index is returned
    (because non-inside candidates are masked with +inf and argmin is applied).

    Args:
        barycentrics: Tensor of shape (num_queries, num_triangles, 3)
            with barycentric weights (w0, w1, w2).
        inside: Boolean tensor of shape (num_queries, num_triangles)
            or (num_queries, num_triangles, 1) — mask of valid triangles.

    Returns:
        Tensor of shape (num_queries, 1) with the selected triangle indices.
    """
    if barycentrics.ndim != 3:
        raise ValueError(
            "Query_barycentrics shape should be (num_query, num_triangles, 3)."
        )
    if barycentrics.shape[:2] != inside.shape:
        raise ValueError(
            "Query_barycentrics shape does not match query_inside (num_query, num_triangles)."
        )

    device = barycentrics.device

    # Compute L1-like score in barycentric space
    score = barycentrics.abs().sum(dim=-1)  # (num_queries, num_triangles)

    # Mask out invalid candidates with +inf
    inf_value = torch.tensor(float("inf"), device=device)
    score = score.masked_fill(~inside, inf_value)

    # Select the best triangle per query
    selected_score, selected = score.min(dim=1, keepdim=True)  # (chunk, 1)
    found = selected_score < inf_value
    return found, selected


@runtime_checkable
class FaceSelector(Protocol):
    def __call__(
        self,
        barycentrics: torch.Tensor,  # (num_queries, num_triangles, 3)
        inside: torch.Tensor,  # (num_queries, num_triangles) or (num_queries, num_triangles, 1)
        /,
        **kwargs: Any,  # extra per-call data (e.g., distances, front_faces, ...)
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (num_queries, 1) found flag and indices of faces
        ...


@dataclass(slots=True, frozen=True)
class BarycentricMapperResult:
    barycentrics: torch.Tensor
    query_inside: torch.Tensor
    query_to_face: torch.Tensor

    @property
    def query_barycentrics_inside(self) -> torch.Tensor:
        return self.barycentrics[self.query_inside]

    @property
    def query_to_face_inside(self) -> torch.Tensor:
        return self.query_to_face[self.query_inside]

    def reshape(self, shape: tuple[int, ...]) -> "BarycentricMapperResult":
        return BarycentricMapperResult(
            self.barycentrics.reshape(*shape, 3),
            self.query_inside.reshape(*shape),
            self.query_to_face.reshape(*shape),
        )


class BarycentricMapper2D:
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
        self.vertices = vertices.double()  # (tri, 2)

        self.area_inv: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None
        self.prepared = False

    def prepare(self):
        self.valid, area, self.faces, self.cw_mask = make_faces_ccw(
            self.faces,
            self.vertices,
            eps_zero=self.eps_zero,
        )
        # Keep +inf for degenerate triangles to use
        # in argmin optimization with query points
        self.area_inv = 1 / area
        self.prepared = True

    def __call__(
        self,
        query: torch.Tensor,
        progress: ProgressProtocol | None = None,
        selector: FaceSelector = select_any_inside,
        **selector_kwargs,
    ) -> BarycentricMapperResult:
        if not self.prepared:
            raise RuntimeError("Call prepare() before calling BarycentricMapper2D()")

        if query.ndim not in [2, 3]:
            raise ValueError(
                "Query dimension must be 2: (num_points, 2),"
                " or 3 (num_points, num_triangles, 2).",
            )
        if query.ndim == 3 and query.shape[1] != self.faces.shape[0]:
            raise ValueError(
                "Query needs to have shape (num_points, num_triangles, 2)."
            )

        query_needs_broadcast = query.ndim == 2

        device = query.device
        num_query = query.shape[0]
        num_chunks = math.ceil(float(num_query) / self.query_chunk_size)
        query = query.double()

        triangles = self.vertices[self.faces]
        # Add new dimension to broadcast along queries
        triangles = triangles.unsqueeze(0)  # (1, poly, 3, 2)
        v0 = triangles[:, :, 0]  # (1, poly, 2)
        v1 = triangles[:, :, 1]
        v2 = triangles[:, :, 2]
        area_inv = self.area_inv.unsqueeze(0)
        valid = self.valid.unsqueeze(0)

        # Allocate tensors for result
        barycentrics = torch.zeros((num_query, 3), dtype=torch.double, device=device)
        query_inside = torch.zeros((num_query,), dtype=torch.bool, device=device)
        query_to_face = torch.zeros((num_query,), dtype=torch.long, device=device)

        progress = progress or ConsoleProgress()
        task = progress.add_task("Finding 2D barycentrics", total=num_chunks)

        for i in range(num_chunks):
            start = i * self.query_chunk_size
            end = min((i + 1) * self.query_chunk_size, num_query)
            query_chunk = query[start:end]  # (chunk, 2)

            if query_needs_broadcast:
                query_chunk = query_chunk.unsqueeze(1)  # (chunk, 1, 2)

            # Find barycentrics using triangle area method
            w0 = area2d(query_chunk, v1, v2) * area_inv  # (chunk, tri)
            w1 = area2d(query_chunk, v2, v0) * area_inv  # keep CCW order
            w2 = 1 - w0 - w1
            chunk_barycentrics = torch.stack([w0, w1, w2], dim=-1)  # (chunk, tri, 3)

            # Found points inside triangles
            inside = (
                valid & (w0 > -self.eps_in) & (w1 > -self.eps_in) & (w2 > -self.eps_in)
            )  # (chunk, poly)

            # Select face for each query
            query_chunk_found, query_chunk_to_face = selector(
                chunk_barycentrics,
                inside,
                **selector_kwargs,
            )  # (chunk, 1)

            # Align with barycentrics shape (chunk, tri, 3)
            query_chunk_to_face_for_bary = query_chunk_to_face.unsqueeze(-1).expand(
                -1, 1, 3
            )

            # Collect found face barycentrics for each query
            query_chunk_barycentrics = torch.gather(
                chunk_barycentrics,
                1,
                query_chunk_to_face_for_bary,
            )  # (chunk, 1, 3)

            # Collect found face's inside flag for each query
            query_chunk_inside = torch.gather(inside, 1, query_chunk_to_face)
            query_chunk_inside &= query_chunk_found

            # Store chunk's result
            chunk_indices = torch.arange(start, end, device=device)
            barycentrics[chunk_indices] = query_chunk_barycentrics.squeeze()
            query_inside[chunk_indices] = query_chunk_inside.squeeze()
            query_to_face[chunk_indices] = query_chunk_to_face.squeeze()

            progress.advance(task)

        # Unflip barycentrics for faces that were originally CW.
        if self.cw_mask.any():
            swap_mask = self.cw_mask[query_to_face[query_inside]]
            w0, w1, w2 = barycentrics[query_inside].unbind(dim=-1)  # (num_inside,)
            swap_with_mask(swap_mask, w1, w2)
            barycentrics[query_inside] = torch.stack([w0, w1, w2], dim=-1)

        return BarycentricMapperResult(
            barycentrics,
            query_inside,
            query_to_face,
        )
