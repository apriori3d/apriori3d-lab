import numpy as np
import torch
import vtk


def get_vtk_camera_to_world(camera: vtk.vtkCamera) -> torch.Tensor:
    vtm = camera.GetViewTransformMatrix()
    arr = [vtm.GetElement(i, j) for i in range(4) for j in range(4)]
    return torch.tensor(arr, dtype=torch.float32).reshape(4, 4)


def get_vtk_camera_focal(
    camera: vtk.vtkCamera,
    width: int,
    height: int,
) -> tuple[float, float]:
    fovy = np.deg2rad(camera.GetViewAngle())
    fy = (height / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    return fx, fy


def get_vtk_camera_center(
    camera: vtk.vtkCamera,
    width: int,
    height: int,
) -> tuple[float, float]:
    wx, wy = camera.GetWindowCenter()
    # follow nerf studio convention to have center in the middle of a pixel
    cx = width / 2 - wx * (width / 2)
    cy = height / 2 - wy * (height / 2)
    return cx, cy
