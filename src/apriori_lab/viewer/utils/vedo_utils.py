import numpy as np
import vedo
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify
from vedo.applications import Plotter

vedo_text_transform = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def setup_remote_scene(plotter: Plotter, title: str = ""):
    server = get_server(client_type="vue2")

    with (
        SinglePageLayout(server) as layout,
        layout.content,
        vuetify.VContainer(fluid=True, classes="pa-0 fill-height"),
    ):
        layout.title.set_text(title)
        plotter.reset_camera()
        vtk.VtkLocalView(plotter.window)

    return server


def create_camera_frustum(
    camera_focal: tuple[float, float],
    camera_center: tuple[float, float],
    image_size: tuple[int, int],
    camera_to_world_transform: np.ndarray,
    scale: float = 0.1,
    color: str = "black",
    label: str = "",
    font_scale=0.25,
    right_direction: tuple[float, float, float] = (1, 0, 0),
    up_direction: tuple[float, float, float] = (0, 1, 0),
    forward_direction: tuple[float, float, float] = (0, 0, -1),
) -> list[vedo.Mesh]:
    camera_orientation_world = np.stack(
        [
            right_direction,
            up_direction,
            forward_direction,
        ]
    ).T

    # calculate frustum's corners rays
    w, h = image_size
    corners = np.array(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]
    )

    camera_center = np.array(camera_center)[None, :]
    camera_focal = np.array(camera_focal)[None, :]
    rays_xy = (corners - camera_center) / camera_focal
    rays_xyz_hg = np.hstack([rays_xy, np.ones((rays_xy.shape[0], 2))])
    rays_xyz_hg[:, :3] @= camera_orientation_world.T
    rays_xyz_hg[:, :3] *= scale

    rays_world = rays_xyz_hg @ camera_to_world_transform.T
    rays_world = rays_world[:, :3]

    figures = []
    cam_center_world = camera_to_world_transform[:3, 3]

    # draw frustum rays
    figures += [vedo.Line(cam_center_world, pt, c=color) for pt in rays_world]

    # draw frustum rectangle
    frustum_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ]
    figures += [
        vedo.Line(rays_world[i], rays_world[j], c=color) for i, j in frustum_edges
    ]

    # draw camera label
    frustum_up = rays_world[2:4].mean(axis=0) - cam_center_world
    frustum_up[2] += 1e-2  # set offset from frustum
    text_in_camera_frame_transform = camera_to_world_transform  # @ vedo_text_transform

    label_transform = text_in_camera_frame_transform.copy()
    label_transform[:3, 3] += frustum_up

    camera_label = vedo.Text3D(label, pos=(0, 0, 0), c="black", s=scale * font_scale)
    camera_label.apply_transform(label_transform)
    figures.append(camera_label)

    # draw camera axes
    axes = camera_to_world_transform[:3, :3] * scale

    for axis, axes_label in zip(axes.T, ("x", "y", "z"), strict=False):
        figures.append(vedo.Line(cam_center_world, cam_center_world + axis, c="blue"))

        axis_label_transform = text_in_camera_frame_transform.copy()
        axis_label_transform[:3, 3] += axis

        axis_label_fig = vedo.Text3D(
            axes_label, pos=(0, 0, 0), c="blue", s=scale * font_scale
        )
        axis_label_fig.apply_transform(axis_label_transform)
        figures.append(axis_label_fig)

    return figures


def create_world_origin(
    scale: float = 1.0,
    color: str = "black",
    font_scale=0.05,
):
    figures = []
    axes = np.eye(3) * scale

    for axis, label in zip(axes, ("x", "y", "z"), strict=False):
        figures.append(vedo.Line((0, 0, 0), axis, c=color))

        axis_label_transform = vedo_text_transform.copy()
        axis_label_transform[:3, 3] += axis

        axis_label = vedo.Text3D(label, pos=(0, 0, 0), c=color, s=scale * font_scale)
        axis_label.apply_transform(axis_label_transform)
        figures.append(axis_label)

    return figures
