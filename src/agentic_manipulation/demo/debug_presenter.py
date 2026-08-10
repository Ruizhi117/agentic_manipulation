"""Non-blocking 2D and 3D windows for Agent perception debugging."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def gripper_wireframe(
    camera_from_gripper: object,
    *,
    width_m: float,
    depth_m: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a small parallel-jaw wireframe in the predicted 6-DoF pose."""

    pose = np.asarray(camera_from_gripper, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_from_gripper must be a finite 4x4 matrix")
    if not np.isfinite(width_m) or width_m <= 0:
        raise ValueError("width_m must be positive and finite")
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be positive and finite")
    half_width = float(width_m) / 2.0
    local = np.array(
        [
            [-depth_m, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -half_width, 0.0],
            [0.0, half_width, 0.0],
            [depth_m, -half_width, 0.0],
            [depth_m, half_width, 0.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((local, np.ones(len(local))))
    vertices = (pose @ homogeneous.T).T[:, :3]
    edges = np.array(
        [[0, 1], [1, 2], [1, 3], [2, 3], [2, 4], [3, 5]],
        dtype=np.int32,
    )
    return vertices, edges


class OpenCVDebugPresenter:
    """Keep reusable grounding/depth/grasp panels alive during control."""

    _TITLE = "Agent debug: grounding + depth + GraspNet prediction"

    def __init__(self) -> None:
        self._opened = False
        self._grounding: np.ndarray | None = None
        self._depth: np.ndarray | None = None

    def __call__(
        self, kind: str, rgb: np.ndarray, artifact_path: Path | None
    ) -> None:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("debug popup image must be a uint8 RGB array")
        if kind == "grounding":
            self._grounding = image.copy()
        elif kind == "depth":
            self._depth = image.copy()
        displayed = image
        panels: list[np.ndarray] = []
        if kind in {"depth", "graspnet"} and self._grounding is not None:
            panels.append(self._grounding)
        if kind == "graspnet" and self._depth is not None:
            panels.append(self._depth)
        if panels:
            panels.append(image)
            resized = [
                panel
                if panel.shape[:2] == image.shape[:2]
                else cv2.resize(panel, (image.shape[1], image.shape[0]))
                for panel in panels
            ]
            displayed = np.concatenate(resized, axis=1)
        cv2.namedWindow(self._TITLE, cv2.WINDOW_NORMAL)
        cv2.imshow(self._TITLE, cv2.cvtColor(displayed, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        self._opened = True
        if artifact_path is not None:
            print(f"[panda-agent] {kind} image: {artifact_path}")

    def pump(self) -> None:
        if self._opened:
            cv2.waitKey(1)

    def close(self) -> None:
        if self._opened:
            try:
                cv2.destroyWindow(self._TITLE)
            except cv2.error:
                pass
        self._opened = False
        self._grounding = None
        self._depth = None
        cv2.waitKey(1)


class Open3DGraspPresenter:
    """Show GraspNet input clouds and its selected gripper in a live 3D window."""

    _TITLE = "GraspNet 3D: point cloud + predicted gripper"

    def __init__(self, *, max_workspace_points: int = 12000) -> None:
        if max_workspace_points <= 0:
            raise ValueError("max_workspace_points must be positive")
        self.max_workspace_points = int(max_workspace_points)
        self._visualizer = None
        self._open3d = None

    @staticmethod
    def _points(value: object, field: str) -> np.ndarray:
        points = np.asarray(value, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"{field} must be a finite (N, 3) array")
        return points

    def _ensure_window(self):
        if self._visualizer is not None:
            return self._visualizer
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Open3D is required for GraspNet 3D display") from exc
        visualizer = o3d.visualization.Visualizer()
        if not visualizer.create_window(
            window_name=self._TITLE, width=960, height=720, visible=True
        ):
            raise RuntimeError("failed to create the Open3D GraspNet window")
        self._open3d = o3d
        self._visualizer = visualizer
        return visualizer

    def __call__(
        self,
        target_points: np.ndarray,
        workspace_points: np.ndarray,
        candidates: object,
        selected: object,
        artifact_path: Path | None,
    ) -> None:
        target = self._points(target_points, "target_points")
        workspace = self._points(workspace_points, "workspace_points")
        if len(workspace) > self.max_workspace_points:
            stride = int(np.ceil(len(workspace) / self.max_workspace_points))
            workspace = workspace[::stride]
        candidate_list = tuple(candidates)
        visualizer = self._ensure_window()
        o3d = self._open3d
        visualizer.clear_geometries()

        workspace_cloud = o3d.geometry.PointCloud()
        workspace_cloud.points = o3d.utility.Vector3dVector(workspace)
        workspace_cloud.paint_uniform_color([0.35, 0.55, 0.85])
        visualizer.add_geometry(workspace_cloud, reset_bounding_box=True)

        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(target)
        target_cloud.paint_uniform_color([0.15, 0.95, 0.25])
        visualizer.add_geometry(target_cloud, reset_bounding_box=False)

        centers = np.asarray(
            [candidate.world_from_gripper[:3, 3] for candidate in candidate_list],
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(centers):
            candidate_cloud = o3d.geometry.PointCloud()
            candidate_cloud.points = o3d.utility.Vector3dVector(centers)
            candidate_cloud.paint_uniform_color([1.0, 0.45, 0.05])
            visualizer.add_geometry(candidate_cloud, reset_bounding_box=False)

        vertices, edges = gripper_wireframe(
            selected.world_from_gripper, width_m=float(selected.width_m)
        )
        gripper = o3d.geometry.LineSet()
        gripper.points = o3d.utility.Vector3dVector(vertices)
        gripper.lines = o3d.utility.Vector2iVector(edges)
        gripper.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.8], (len(edges), 1))
        )
        visualizer.add_geometry(gripper, reset_bounding_box=False)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        axes.transform(np.asarray(selected.world_from_gripper, dtype=np.float64))
        visualizer.add_geometry(axes, reset_bounding_box=False)

        render_option = visualizer.get_render_option()
        render_option.background_color = np.asarray([0.03, 0.03, 0.03])
        render_option.point_size = 3.0
        controller = visualizer.get_view_control()
        controller.set_lookat(np.mean(target, axis=0))
        controller.set_front([0.0, 0.0, -1.0])
        controller.set_up([0.0, -1.0, 0.0])
        controller.set_zoom(0.7)
        visualizer.poll_events()
        visualizer.update_renderer()
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            visualizer.capture_screen_image(str(artifact_path), do_render=True)
            print(f"[panda-agent] graspnet 3D image: {artifact_path}")

    def pump(self) -> None:
        if self._visualizer is not None:
            self._visualizer.poll_events()
            self._visualizer.update_renderer()

    def close(self) -> None:
        if self._visualizer is not None:
            self._visualizer.destroy_window()
        self._visualizer = None
        self._open3d = None
