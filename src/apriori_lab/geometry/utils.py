import torch
from torch.nn import functional as F


def area2d(
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    """Oriented 2D triangle area * 2 (z-component of cross)."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    area = edge1[..., 0] * edge2[..., 1] - edge1[..., 1] * edge2[..., 0]
    return area.unsqueeze(dim=-1) if keepdim else area


def swap_with_mask(
    mask: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    """Swap a and b in-place where mask==True (1D/ND-safe broadcasting)."""
    new_a = b.where(mask, a)
    new_b = a.where(mask, b)
    a.copy_(new_a)
    b.copy_(new_b)


def make_faces_ccw(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    eps_zero: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder triangle faces to CCW in 2D and return validity + (non-negative) areas.

    The function treats `vertices` as 2D (UV/screen) or 3D but only XY is used.
    Faces that are clockwise are flipped (swap v1<->v2) to become CCW.
    Degenerate faces (|area| < eps_zero) are marked invalid and area set to 0.

    Args:
        faces (torch.Tensor): (F, 3) long indices of triangle vertices.
        vertices (torch.Tensor): (V, 2|3) vertex positions. Only XY are used.
        eps_zero (float): Degeneracy threshold for area magnitude.

    Returns:
        tuple:
            valid (torch.Tensor): (F,) bool — True for non-degenerate triangles.
            area (torch.Tensor): (F,) non-negative 2*area for CCW-ordered faces (0 for degenerate).
            ccw_faces (torch.Tensor): (F, 3) reordered faces in CCW order.

    Notes:
        - If `vertices` is 3D, only XY components are used to determine orientation.
    """
    if faces.dtype != torch.long:
        raise TypeError(f"faces must be torch.long, got {faces.dtype}")
    if not vertices.dtype.is_floating_point:
        raise TypeError(f"vertices must be float tensor, got {vertices.dtype}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError(
            f"vertices must have shape (V,2) or (V,3), got {vertices.shape}"
        )

    vertices_xy = vertices[:, :2]

    vi0 = faces[:, 0]  # (tri, )
    vi1 = faces[:, 1]
    vi2 = faces[:, 2]
    v0 = vertices_xy[vi0]  # (tri, 2)
    v1 = vertices_xy[vi1]
    v2 = vertices_xy[vi2]

    # Calculate signed area and find negative values
    area_signed = area2d(v0, v1, v2)  # (tri, )
    negative = area_signed < -eps_zero

    # Reorder CW faces
    if negative.any():
        swap_with_mask(negative, vi1, vi2)

    # Make area positive or zero
    area = area_signed.abs()
    valid = area > eps_zero
    zero_area = torch.zeros_like(area_signed)
    valid_area = area.where(valid, zero_area)

    # Stack faces in CCW order
    ccw_faces = torch.stack([vi0, vi1, vi2], dim=-1)  # (tri, 3)

    return valid, valid_area, ccw_faces


def triangle_local_frame(
    triangle: torch.Tensor,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a local orthonormal frame (b0, b1, b2) for each triangle.

    Constructs a right-handed ONB using Gram–Schmidt:
      - b0 is the normalized edge (v1 - v0)
      - b1 is the normalized component of (v2 - v0) orthogonal to b0
      - b2 = b0 × b1

    If the triangle is degenerate (edges are collinear or near zero area),
    `not_collinear` is False for that triangle and (b0, b1, b2) become arbitrary but
    well-defined (no NaNs). You can use the mask to skip such triangles.

    Args:
        triangle (torch.Tensor): (..., 3, 3) triangle vertices (v0, v1, v2) in 3D.
        eps_collinear (float): Threshold to detect degeneracy (near-collinearity).

    Returns:
        tuple:
            not_collinear (torch.Tensor): (...,) bool mask where the triangle has non-zero area.
            frame (torch.Tensor): (..., 3, 3) matrix whose columns are [b0, b1, b2].

    Notes:
        - The frame is right-handed: b2 = b0 × b1.
        - For degenerate triangles, `not_collinear=False`; the returned frame is [b0, 0, 0].
    """
    device = triangle.device
    zero = torch.tensor(0, dtype=torch.float32, device=device)

    v0 = triangle[:, 0, :]  # (..., 3)
    v1 = triangle[:, 1, :]
    v2 = triangle[:, 2, :]
    e0 = v1 - v0  # (..., 3)
    e1 = v2 - v0

    # First basis vector along e0
    b0 = F.normalize(e0, dim=-1)
    b0 = b0.where(b0.abs() > eps, zero)

    # Second basis:
    # orthogonalize e1 against b0 → component of e1 perpendicular to b0
    proj_coeff = (e1 * b0).sum(dim=-1, keepdim=True)  # (..., 1)
    e1_orpho = e1 - proj_coeff * b0  # (..., 3)

    # Detect degeneracy: if e1_ortho is too small -> triangle is nearly collinear
    not_collinear = e1_orpho.norm(dim=-1, keepdim=True) > eps  # (..., 1)
    zeros = torch.zeros_like(e1)
    # Choose between e1_ortho and zero
    b1 = F.normalize(e1_orpho, dim=-1).where(not_collinear, zeros)
    b1 = b1.where(b1.abs() > eps, zero)

    # Third basis vector to make the frame right-handed
    b1_vec = torch.cross(b0, b1, dim=-1)
    b2 = F.normalize(b1_vec, dim=-1).where(not_collinear, zeros)
    b2 = b2.where(b2.abs() > eps, zero)

    frame = torch.stack([b0, b1, b2], dim=-1)  # (..., 3, 3)

    return not_collinear.squeeze(), frame

