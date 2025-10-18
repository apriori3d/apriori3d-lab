import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


class Overlay:
    def __init__(
        self,
        main_renderer: vtk.vtkRenderer,
        alpha: float = 1,
    ) -> None:
        self.alpha = alpha
        self.input_image = None

        renderer = vtk.vtkRenderer()
        renderer.SetLayer(1)
        renderer.SetInteractive(0)
        self.renderer = renderer

        camera = renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        self.camera = camera

        window = main_renderer.GetRenderWindow()
        window.SetNumberOfLayers(2)
        window.AddRenderer(renderer)

        self.image = vtk.vtkImageData()

        self.actor = vtk.vtkImageActor()
        self.actor.SetInputData(self.image)
        self.renderer.AddActor(self.actor)

    def set_bounds_norm(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        render_w, render_h = self.renderer.GetSize()
        self._config_camera_to_pixels(render_w, render_h)

        # coordinate in view port pixel space
        x0 = int(round(start[0] * render_w))
        y0 = int(round(start[1] * render_h))  # (0,0) — bottom left corner in VTK
        x1 = int(round(end[0] * render_w))
        y1 = int(round(end[1] * render_h))

        target_w = max(1, x1 - x0)
        target_h = max(1, y1 - y0)
        image_w, image_h, _ = self.image.GetDimensions()  # image real size

        self.actor.SetPosition(float(x0), float(y0), 0.0)

        sx = target_w / max(1, float(image_w))
        sy = target_h / max(1, float(image_h))
        self.actor.SetScale(sx, sy, 1.0)

        self.renderer.ResetCameraClippingRange()

    def set_size(self, size: tuple[int, int]):
        w, h = size
        self.image.SetDimensions(w, h, 1)
        self.image.SetOrigin(0, 0, 0)
        self.image.SetSpacing(1, 1, 1)
        self.image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
        self.image.Modified()

    def resize_to_viewport(self, main_renderer: vtk.vtkRenderer):
        self.renderer.SetViewport(main_renderer.GetViewport())
        w, h = main_renderer.GetSize()

        self._config_camera_to_pixels(w, h)
        self.renderer.ResetCameraClippingRange()

    def _config_camera_to_pixels(self, width, height):
        self.camera.SetViewUp(0, 1, 0)
        self.camera.SetFocalPoint(width * 0.5, height * 0.5, 0.0)
        self.camera.SetPosition(width * 0.5, height * 0.5, 1.0)
        self.camera.SetParallelScale(height * 0.5)

    def clear(self):
        w, h, _ = self.image.GetDimensions()
        empty_image = np.full((h, w, 4), 0, dtype=np.uint8)
        self.set_image(empty_image)

    def update_alpha(self, alpha: float):
        if self.input_image is not None:
            self.alpha = alpha
            self.set_image(self.input_image[:, :, :3])

    def set_image(self, image: np.ndarray):
        w, h, _ = self.image.GetDimensions()

        if image.shape[:2] != (h, w):
            raise ValueError(
                f"Image shape for overlay does not match:{image.shape[:2]} != {(h, w)}"
            )
        if image.shape[2] not in [3, 4]:
            raise ValueError(
                f"Image should have 3 or 4 channels, not {image.shape[2]}."
            )
        if image.dtype not in [np.float32, np.uint8]:
            raise ValueError("Image data type should be torch.float32 or torch.uint8.")

        if image.dtype == np.float32:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        if image.shape[2] == 3:
            # add alpha channel
            default_alpha = int(self.alpha * 255)
            alpha_channel = np.full(
                (h, w, 1),
                default_alpha,
                dtype=np.uint8,
            )
            image = np.concatenate([image, alpha_channel], axis=-1)

        self.input_image = image

        vtk_image_data = numpy_to_vtk(
            num_array=image.reshape(-1, order="C"),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        vtk_image_data.SetNumberOfComponents(4)

        self.image.GetPointData().SetScalars(vtk_image_data)
        self.image.Modified()
