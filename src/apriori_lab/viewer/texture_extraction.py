import argparse
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
import vedo
from pytorch3d.io import load_obj
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify

from apriori_lab.geometry.surface_mapping import (
    BarycentricMapper2D,
    BarycentricMapperResult,
    RayTriangleIntersector,
    RayTriangleIntersectorResult,
)
from apriori_lab.utils.vedo_utils import create_camera_frustum
from apriori_lab.viewer.widgets.overlay import Overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "smpl_model",
        type=Path,
        help="Path to smplx model",
    )
    parser.add_argument(
        "uv_map",
        type=Path,
        help="Path to smplx uv mapping",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to run visualization on",
        default="0.0.0.0",
    )

    return parser.parse_args()


def _draw_uv_map(
    image: np.ndarray,
    uvs: np.ndarray,
    width: int,
    height: int,
    thickness: int = 1,
    color=(0, 255, 0, 255),
) -> np.ndarray:
    image_uvs = ((uvs * np.array([width, height])) + 0.5).astype(np.int32)

    for p1, p2, p3 in image_uvs:
        cv2.line(image, p1, p2, color, thickness)
        cv2.line(image, p1, p3, color, thickness)
        cv2.line(image, p2, p3, color, thickness)

    return image


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    radius: int = 1,
    thickness: int = -1,
    color=(0, 255, 0, 255),
) -> np.ndarray:
    for point in points:
        cv2.circle(
            image,
            point,
            color=color,
            radius=radius,
            thickness=thickness,
        )

    return image


class Viewer:
    def __init__(
        self,
        smpl_model_file: str,
        uv_map_file: str,
        texture_size: tuple[int, int] = (512, 512),
        eps_zero: float = 1e-12,
        eps_in: float = 1e-6,
    ):
        self.smpl_model_file = smpl_model_file
        self.uv_map_file = uv_map_file
        self.eps_zero = eps_zero
        self.eps_in = eps_in
        self.texture_size = texture_size

        self.plt = vedo.Plotter()
        self.overlay = Overlay(self.plt.renderer)
        self.view: vtk_widgets.VtkRemoteView | None = None

    def __call__(self):
        device = torch.cuda.current_device()
        body_model = smplx.create(
            model_path=str(args.smpl_model),
            model_type="smplx",
            gender="male",
            use_pca=False,
            batch_size=1,
            num_betas=10,
        ).to(device=device)
        body_model.requires_grad_(False)

        body = body_model()
        vertices = body.vertices[0].to(device=device)
        faces = torch.tensor(
            body_model.faces.astype(np.int64),
            dtype=torch.int64,
            device=device,
        )

        _, uv_faces_props, props = load_obj(self.uv_map_file, load_textures=False)
        uv_faces = uv_faces_props.textures_idx.to(device)
        uvs = props.verts_uvs.to(device)

        self.attach_resize_handlers()
        self.show_body(faces, vertices)
        self.show_uv_map(uv_faces, uvs)

        uv_mapping = self.show_texture_points_on_surface(faces, vertices, uv_faces, uvs)
        ray_mapping = self.show_camera_rays_on_surface(faces, vertices)

        uv_faces = torch.unique(uv_mapping.query_to_face_inside)
        ray_faces = torch.unique(ray_mapping.ray_to_face_hit)
        uv_pixels_in_tri = torch.bincount(uv_mapping.query_to_face_inside)
        ray_pixels_in_tri = torch.bincount(ray_mapping.ray_to_face_hit)

        mask = torch.isin(uv_faces, ray_faces)
        shared_faces = uv_faces[mask]
        mapping_ratio = (
            ray_pixels_in_tri[shared_faces] / uv_pixels_in_tri[shared_faces]
        ).mean()
        print(f"mapping ratio:{mapping_ratio:.02f}")

        self._start_server()

    def _start_server(self):
        server = get_server(client_type="vue2")
        with (
            SinglePageLayout(server) as layout,
            layout.content,
            vuetify.VContainer(
                fluid=True,
                classes="pa-0 ma-0 fill-height",
                style="height: 100%; width: 100%;",
            ),
        ):
            layout.title.set_text("Sampling Methods on Mesh Surfaces")

            self.plt.reset_camera()
            self.view = vtk_widgets.VtkRemoteView(
                self.plt.window,
                interactive_ratio=1,
                still_ratio=1,
                interactive_quality=90,
                still_quality=100,
            )
            server.controller.on_server_ready.add(self.view.update)

        server.start()

    def attach_resize_handlers(self):
        window = self.plt.window  # vtkRenderWindow
        interactor = window.GetInteractor()  # vtkRenderWindowInteractor
        window.AddObserver("WindowResizeEvent", self.on_resize)
        interactor.AddObserver("WindowResizeEvent", self.on_resize)

    def on_resize(self, *args, **kw_args):
        self.overlay.resize_to_viewport(self.plt.renderer)

    def show_body(
        self,
        faces: torch.Tensor,
        vertices: torch.Tensor,
    ):
        self.plt.clear()

        body_mesh = vedo.Mesh([vertices.cpu(), faces.cpu()])
        self.plt += body_mesh.wireframe()
        axes = vedo.Axes(body_mesh)
        self.plt += axes.unpack()

        self.plt.render()
        if self.view:
            self.view.update()

    def show_uv_map(self, uv_faces: torch.Tensor, uvs: torch.Tensor):
        width, height = self.texture_size
        uv_coords = uvs[uv_faces]

        image = np.zeros((height, width, 4), dtype=np.uint8)
        uv_map_image = _draw_uv_map(image, uv_coords.cpu().numpy(), width, height)

        screen_width, screen_height = self.plt.renderer.GetSize()
        aspect = screen_width / screen_height
        w_norm = 0.4
        h_norm = w_norm / aspect

        # left upper corner
        x0 = 0.0
        y0 = 1.0 - h_norm
        x1 = x0 + w_norm
        y1 = 1.0

        self.overlay.set_size((width, height))
        self.overlay.set_bounds_norm((x0, y0), (x1, y1))
        self.overlay.set_image(uv_map_image)

    def show_texture_points_on_surface(
        self,
        faces: torch.Tensor,
        vertices: torch.Tensor,
        uv_faces: torch.Tensor,
        uvs: torch.Tensor,
    ) -> BarycentricMapperResult:
        width, height = self.texture_size
        device = uv_faces.device

        xs = torch.linspace(0, 1, steps=width, device=device)
        ys = torch.linspace(0, 1, steps=height, device=device)
        x, y = torch.meshgrid(xs, ys, indexing="xy")
        pixel_coords = torch.stack([x, y], dim=-1).reshape(-1, 2)

        mapper = BarycentricMapper2D(uv_faces, uvs)
        mapper.prepare()
        result = mapper(pixel_coords)
        uv_faces.copy_(mapper.faces)

        found_faces = faces[result.query_to_face_inside]
        found_barycentrics = result.query_barycentrics_inside

        v0 = vertices[found_faces[:, 0]]
        v1 = vertices[found_faces[:, 1]]
        v2 = vertices[found_faces[:, 2]]
        w0 = found_barycentrics[:, 0:1]
        w1 = found_barycentrics[:, 1:2]
        w2 = found_barycentrics[:, 2:3]
        texture_points_3d = v0 * w0 + v1 * w1 + v2 * w2

        cloud = vedo.Points(texture_points_3d.cpu(), r=3, c="green")
        self.plt += cloud

        self.plt.render()
        if self.view:
            self.view.update()

        return result

    def show_camera_rays_on_surface(
        self,
        faces: torch.Tensor,
        vertices: torch.Tensor,
    ) -> RayTriangleIntersectorResult:
        distance_threshold = 5
        device = faces.device

        camera_to_world_transform = torch.eye(4)
        camera_to_world_transform[2, 3] = 2

        scale = 2.0
        ratio = 1.0
        cam_w, cam_h = 720.0 / scale, 1280 / scale
        cx = cam_w / 2 / ratio
        cy = cam_h / 2 / ratio
        fx = ratio * 645 / scale
        fy = ratio * 645 / scale
        # cx = 20.0
        # cy = 20.0
        # fx = 20.0
        # fy = 20.0
        width = int(cx * 2)
        height = int(cy * 2)

        camera = create_camera_frustum(
            camera_focal=(fx, fy),
            camera_center=(cx, cy),
            image_size=(width, height),
            camera_to_world_transform=camera_to_world_transform.numpy(),
            label="Camera",
        )
        self.plt += camera

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world_transform[None, :3],
            camera_type=CameraType.PERSPECTIVE,
        )
        cameras = cameras.to(device)
        rays = cameras.generate_rays(camera_indices=0)

        ray_origins = rays.origins.view(-1, 3)  # (rays, 3)
        ray_dirs = rays.directions.view(-1, 3)

        intersector = RayTriangleIntersector(faces, vertices)
        intersector.prepare()
        result = intersector(ray_origins, ray_dirs)

        points = result.points_hit
        points = points[result.distances_hit.abs() < distance_threshold]
        cloud = vedo.Points(points.cpu(), r=3, c="blue")
        self.plt += cloud

        self.plt.render()
        if self.view:
            self.view.update()

        return result


if __name__ == "__main__":
    args = parse_args()

    if not args.smpl_model.exists():
        raise FileNotFoundError(args.smpl_model)

    viewer = Viewer(
        smpl_model_file=str(args.smpl_model),
        uv_map_file=str(args.uv_map),
        texture_size=(256, 256),
    )
    viewer()
