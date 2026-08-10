"""Independent scene, point-cloud, grasp, and Panda pick demo stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agentic_manipulation.control.panda_atomic import (
    PandaAtomicPickPlaceSkill,
    PandaDeltaPoseBackend,
)
from agentic_manipulation.demo.grasp_component import run_grasp_request
from agentic_manipulation.demo.panda_calibration import (
    CalibratedTopDownPandaGraspProvider,
    PANDA_GRASP_FROM_EE,
    calibrated_camera_from_grasp,
    compose_panda_world_ee,
)
from agentic_manipulation.demo.pointcloud_artifacts import write_point_cloud_bundle
from agentic_manipulation.demo.protocol import (
    atomic_write_json,
    read_json,
    resolve_project_path,
)
from agentic_manipulation.demo.stage_recorder import StageRecorder
from agentic_manipulation.envs.ee_camera_scene import (
    DESTINATION_INSTANCE_IDS,
    GRASPABLE_INSTANCE_IDS,
)
from agentic_manipulation.errors import (
    AgenticManipulationError,
    ConfigurationError,
    ExecutionError,
)
from agentic_manipulation.perception.camera import CameraAdapter
from agentic_manipulation.types import BBox, CameraFrame


STAGES = ("scene", "pointcloud", "grasp", "pick", "place")
_PREDECESSOR = {
    "pointcloud": "scene",
    "grasp": "pointcloud",
    "pick": "grasp",
    "place": "pick",
}
_MOCK_IDS = {
    instance_id: index
    for index, instance_id in enumerate(
        GRASPABLE_INSTANCE_IDS + DESTINATION_INSTANCE_IDS, start=1
    )
}


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _status_path(output: Path, stage: str) -> Path:
    return output / stage / "status.json"


def _require_predecessor(
    output: Path, stage: str, mode: str, target: str
) -> dict[str, object]:
    predecessor = _PREDECESSOR[stage]
    path = _status_path(output, predecessor)
    if not path.is_file():
        raise ConfigurationError(f"{predecessor} stage status is missing: {path}")
    status = read_json(path)
    if status.get("status") != "ok":
        raise ConfigurationError(f"{predecessor} stage status is not ok")
    if status.get("mode") != mode or status.get("target") != target:
        raise ConfigurationError(
            f"{predecessor} stage mode/target does not match this request"
        )
    return status


def _bbox_for_segmentation(segmentation: np.ndarray, segmentation_id: int) -> BBox:
    rows, columns = np.nonzero(segmentation == segmentation_id)
    if len(rows) == 0:
        raise ConfigurationError(
            f"segmentation id {segmentation_id} is not visible in the wrist camera"
        )
    height, width = segmentation.shape
    return BBox(
        float(columns.min()) / width,
        float(rows.min()) / height,
        float(columns.max() + 1) / width,
        float(rows.max() + 1) / height,
    )


def _mock_observation() -> tuple[dict[str, object], dict[str, int]]:
    height = width = 96
    rgb = np.full((1, height, width, 3), 32, dtype=np.uint8)
    depth = np.full((1, height, width, 1), 0.48, dtype=np.float32)
    segmentation = np.zeros((1, height, width, 1), dtype=np.int32)
    palette = (
        (220, 40, 40),
        (40, 80, 220),
        (230, 200, 30),
        (150, 50, 190),
        (30, 180, 70),
        (235, 110, 25),
        (230, 230, 230),
        (245, 100, 160),
    )
    for index, (instance_id, segmentation_id) in enumerate(_MOCK_IDS.items()):
        row, column = divmod(index, 4)
        y1, y2 = 12 + row * 38, 30 + row * 38
        x1, x2 = 6 + column * 23, 22 + column * 23
        rgb[0, y1:y2, x1:x2] = palette[index]
        depth[0, y1:y2, x1:x2, 0] = 0.40 + index * 0.005
        segmentation[0, y1:y2, x1:x2, 0] = segmentation_id
    intrinsic = np.array(
        [[[110.0, 0.0, 48.0], [0.0, 110.0, 48.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    scene_extrinsic = np.eye(4, dtype=np.float32)[:3][None]
    hand_world = np.eye(4, dtype=np.float32)
    hand_world[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    hand_world[:3, 3] = [0.0, 0.0, 0.85]
    hand_extrinsic = np.linalg.inv(hand_world)[:3][None]
    observation: dict[str, object] = {
        "sensor_data": {
            "scene_camera": {"rgb": rgb.copy()},
            "hand_camera": {
                "rgb": rgb,
                "depth": depth,
                "segmentation": segmentation,
            },
        },
        "sensor_param": {
            "scene_camera": {
                "intrinsic_cv": intrinsic,
                "extrinsic_cv": scene_extrinsic,
            },
            "hand_camera": {
                "intrinsic_cv": intrinsic,
                "extrinsic_cv": hand_extrinsic,
            },
        },
    }
    return observation, _MOCK_IDS.copy()


def _make_real_env(render_backend: str):
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import agentic_manipulation.envs  # noqa: F401
    except ImportError as exc:
        raise ConfigurationError(f"ManiSkill simulation dependencies are missing: {exc}") from exc
    return gym.make(
        "EECameraScene-v1",
        robot_uids="panda_wristcam",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb+depth+segmentation",
        render_backend=render_backend,
        num_envs=1,
        sensor_configs={
            "hand_camera": {"width": 320, "height": 240},
            "scene_camera": {"width": 320, "height": 240},
        },
    )


def _capture_real_scene(render_backend: str):
    env = _make_real_env(render_backend)
    try:
        observation, _ = env.reset(seed=0, options={"reconfigure": True})
        base = env.unwrapped
        ids = {
            instance_id: int(actor.per_scene_id[0].detach().cpu().item())
            for instance_id, actor in base.semantic_actors.items()
        }
        frame = CameraAdapter().capture(
            observation, observation["sensor_param"], "hand_camera", timestamp=0.0
        )
        overview = np.asarray(
            observation["sensor_data"]["scene_camera"]["rgb"][0]
            .detach()
            .cpu()
            .numpy()
        )
        return frame, overview, ids
    finally:
        env.close()


def _write_scene_arrays(
    directory: Path,
    root: Path,
    frame: CameraFrame,
    overview: np.ndarray,
    ids: dict[str, int],
    mode: str,
    target: str,
) -> dict[str, object]:
    if frame.segmentation is None:
        raise ConfigurationError("wrist frame must contain instance segmentation")
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "rgb_path": directory / "hand_rgb.png",
        "overview_path": directory / "scene_overview.png",
        "depth_path": directory / "hand_depth_m.npy",
        "segmentation_path": directory / "hand_segmentation.npy",
        "intrinsic_path": directory / "hand_intrinsic.npy",
        "world_from_camera_path": directory / "world_from_hand_camera.npy",
    }
    Image.fromarray(frame.rgb).save(files["rgb_path"])
    Image.fromarray(np.asarray(overview, dtype=np.uint8)).save(files["overview_path"])
    np.save(files["depth_path"], frame.depth_m.astype(np.float32, copy=False))
    np.save(files["segmentation_path"], frame.segmentation.astype(np.int32, copy=False))
    np.save(files["intrinsic_path"], frame.intrinsic.astype(np.float32, copy=False))
    np.save(
        files["world_from_camera_path"],
        frame.world_from_camera.astype(np.float32, copy=False),
    )
    visible = {
        instance_id: int(np.count_nonzero(frame.segmentation == segmentation_id))
        for instance_id, segmentation_id in ids.items()
    }
    missing = [name for name in _MOCK_IDS if visible.get(name, 0) < 4]
    if missing:
        raise ConfigurationError(f"wrist camera does not see all scene components: {missing}")
    payload: dict[str, object] = {
        "stage": "scene",
        "status": "ok",
        "mode": mode,
        "is_mock": mode == "mock",
        "target": target,
        "robot": "panda_wristcam",
        "visible_pixel_counts": visible,
        "segmentation_ids": ids,
        **{name: _relative(path, root) for name, path in files.items()},
    }
    atomic_write_json(directory / "status.json", payload)
    return payload


def _load_scene_frame(root: Path, status: dict[str, object]) -> CameraFrame:
    def load(field: str) -> np.ndarray:
        value = status.get(field)
        if not isinstance(value, str):
            raise ConfigurationError(f"scene status is missing {field}")
        path = resolve_project_path(root, value)
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"failed to load {field}: {exc}") from exc

    rgb_value = status.get("rgb_path")
    if not isinstance(rgb_value, str):
        raise ConfigurationError("scene status is missing rgb_path")
    rgb_path = resolve_project_path(root, rgb_value)
    try:
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise ConfigurationError(f"failed to load scene RGB: {exc}") from exc
    return CameraFrame(
        rgb=rgb,
        depth_m=load("depth_path").astype(np.float32, copy=False),
        intrinsic=load("intrinsic_path").astype(np.float32, copy=False),
        world_from_camera=load("world_from_camera_path").astype(np.float32, copy=False),
        segmentation=load("segmentation_path").astype(np.int32, copy=False),
        timestamp=0.0,
    )


def _run_scene(
    output: Path, root: Path, mode: str, target: str, render_backend: str
) -> dict[str, object]:
    if mode == "mock":
        observation, ids = _mock_observation()
        frame = CameraAdapter().capture(
            observation, observation["sensor_param"], "hand_camera", timestamp=0.0
        )
        overview = np.asarray(observation["sensor_data"]["scene_camera"]["rgb"])[0]
    else:
        frame, overview, ids = _capture_real_scene(render_backend)
    return _write_scene_arrays(output / "scene", root, frame, overview, ids, mode, target)


def _run_pointcloud(
    output: Path, root: Path, mode: str, target: str
) -> dict[str, object]:
    scene = _require_predecessor(output, "pointcloud", mode, target)
    frame = _load_scene_frame(root, scene)
    ids = scene.get("segmentation_ids")
    if not isinstance(ids, dict) or target not in ids:
        raise ConfigurationError(f"scene status has no segmentation id for {target}")
    segmentation_id = int(ids[target])
    if frame.segmentation is None:
        raise ConfigurationError("scene frame has no segmentation")
    bbox = _bbox_for_segmentation(frame.segmentation, segmentation_id)
    artifacts = write_point_cloud_bundle(
        output / "pointcloud", frame, bbox, segmentation_id
    )
    payload: dict[str, object] = {
        "stage": "pointcloud",
        "status": "ok",
        "mode": mode,
        "is_mock": mode == "mock",
        "target": target,
        "target_bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        "segmentation_id": segmentation_id,
        "target_point_count": artifacts.target_point_count,
        "workspace_point_count": artifacts.workspace_point_count,
        "target_points_path": _relative(artifacts.target_npy, root),
        "target_ply_path": _relative(artifacts.target_ply, root),
        "workspace_points_path": _relative(artifacts.workspace_npy, root),
        "overlay_path": _relative(artifacts.overlay_png, root),
        "world_from_camera": frame.world_from_camera.tolist(),
    }
    atomic_write_json(output / "pointcloud" / "status.json", payload)
    return payload


def _grasp_request(
    pointcloud: dict[str, object], checkpoint_path: str, device: str
) -> dict[str, object]:
    return {
        "request_id": 1,
        "target_points_path": pointcloud["target_points_path"],
        "workspace_points_path": pointcloud["workspace_points_path"],
        "world_from_camera": pointcloud["world_from_camera"],
        "grasp_from_ee": PANDA_GRASP_FROM_EE.tolist(),
        "max_width_m": 0.081,
        "checkpoint_path": checkpoint_path,
        "device": device,
    }


def _run_grasp(
    output: Path,
    root: Path,
    mode: str,
    target: str,
    checkpoint_path: str,
    device: str,
) -> dict[str, object]:
    pointcloud = _require_predecessor(output, "grasp", mode, target)
    response = run_grasp_request(
        _grasp_request(pointcloud, checkpoint_path, device), root, mode
    )
    payload = {
        "stage": "grasp",
        "status": "ok",
        "mode": mode,
        "is_mock": mode == "mock",
        "target": target,
        **response,
    }
    directory = output / "grasp"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "status.json", payload)
    return payload


class _MockPickBackend:
    def __init__(self, recorder: StageRecorder, observation: dict[str, object]) -> None:
        self.recorder = recorder
        self.observation = observation
        self.home_pose = np.eye(4, dtype=np.float64)
        self.home_pose[2, 3] = 0.45
        self.gripper = 1.0

    def _record(self, stage: str, pose: np.ndarray) -> None:
        self.recorder.capture(stage, self.observation)
        self.recorder.record_motion(stage, pose, self.gripper)

    def move_ee(self, target_pose, steps, stage) -> None:
        del steps
        self._record(stage, np.asarray(target_pose, dtype=np.float64))

    def set_gripper(self, value, steps, stage) -> None:
        del steps
        self.gripper = float(value)
        self._record(stage, self.home_pose)

    def is_grasping(self, instance_id) -> bool:
        del instance_id
        return True

    def bin_inner_aabb(self, bin_id):
        del bin_id
        return np.array([0.0, 0.0, 0.01]), np.array([0.1, 0.1, 0.10])

    def object_half_height(self, instance_id):
        del instance_id
        return 0.02

    def settle(self, steps) -> None:
        del steps
        self._record("settle", self.home_pose)


def _run_mock_pick(
    output: Path, root: Path, target: str, grasp: dict[str, object]
) -> dict[str, object]:
    observation, _ = _mock_observation()
    recorder = StageRecorder()
    recorder.capture("home_observe", observation)
    recorder.record_motion("home_observe", np.eye(4), 1.0)
    backend = _MockPickBackend(recorder, observation)
    report = PandaAtomicPickPlaceSkill(backend).pick(target, grasp["world_from_ee"])
    directory = output / "pick"
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "pick.mp4"
    motion = directory / "motion.json"
    recorder.write_mp4(video)
    recorder.write_motion_json(motion)
    payload: dict[str, object] = {
        "stage": "pick",
        "status": "ok" if report.success else "error",
        "mode": "mock",
        "is_mock": True,
        "target": target,
        "success": report.success,
        "simulator_grasped": report.success,
        "stages": list(report.stages),
        "failure_reason": report.failure_reason,
        "gripper_semantics": {"close": -1.0, "open": 1.0},
        "video_path": _relative(video, root),
        "motion_path": _relative(motion, root),
    }
    atomic_write_json(directory / "status.json", payload)
    return payload


def _run_real_pick(
    output: Path,
    root: Path,
    target: str,
    render_backend: str,
    motion_steps: int,
) -> dict[str, object]:
    env = _make_real_env(render_backend)
    directory = output / "pick"
    directory.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder()
    try:
        observation, _ = env.reset(seed=0, options={"reconfigure": True})
        base = env.unwrapped
        frame = CameraAdapter().capture(
            observation, observation["sensor_param"], "hand_camera", timestamp=0.0
        )
        segmentation_id = int(
            base.semantic_actors[target].per_scene_id[0].detach().cpu().item()
        )
        if frame.segmentation is None:
            raise ConfigurationError("real wrist frame has no segmentation")
        bbox = _bbox_for_segmentation(frame.segmentation, segmentation_id)
        cloud = write_point_cloud_bundle(directory / "recomputed_cloud", frame, bbox, segmentation_id)
        camera_position = np.mean(
            np.load(cloud.target_npy, allow_pickle=False), axis=0
        )
        home_rotation = np.asarray(base.observation_home_pose, dtype=np.float64)[:3, :3]
        camera_from_grasp = calibrated_camera_from_grasp(
            frame.world_from_camera,
            camera_position,
            home_rotation,
        )
        world_from_ee = compose_panda_world_ee(
            frame.world_from_camera, camera_from_grasp
        )
        recorder.capture("home_observe", observation)
        recorder.record_motion(
            "home_observe",
            base.agent.tcp_pose.to_transformation_matrix()[0].detach().cpu().numpy(),
            1.0,
        )
        backend = PandaDeltaPoseBackend(env, recorder)
        report = PandaAtomicPickPlaceSkill(
            backend, motion_steps=motion_steps, gripper_steps=20
        ).pick(target, world_from_ee)
        simulator_grasped = backend.is_grasping(target)
        video = directory / "pick.mp4"
        motion = directory / "motion.json"
        recorder.write_mp4(video)
        recorder.write_motion_json(motion)
        payload: dict[str, object] = {
            "stage": "pick",
            "status": "ok" if report.success and simulator_grasped else "error",
            "mode": "real",
            "is_mock": False,
            "target": target,
            "success": bool(report.success and simulator_grasped),
            "simulator_grasped": simulator_grasped,
            "grasp_source": "deterministic_calibrated_safety_fixture",
            "model_is_mock": True,
            "camera_from_grasp": camera_from_grasp.tolist(),
            "world_from_ee": world_from_ee.tolist(),
            "stages": list(report.stages),
            "failure_reason": report.failure_reason,
            "gripper_semantics": {"close": -1.0, "open": 1.0},
            "video_path": _relative(video, root),
            "motion_path": _relative(motion, root),
        }
        atomic_write_json(directory / "status.json", payload)
        if not payload["success"]:
            raise ExecutionError(report.failure_reason or "real Panda did not hold target after lift")
        return payload
    finally:
        env.close()


def _run_mock_place(
    output: Path,
    root: Path,
    target: str,
    destination: str,
    grasp: dict[str, object],
) -> dict[str, object]:
    observation, _ = _mock_observation()
    recorder = StageRecorder()
    recorder.capture("home_observe", observation)
    recorder.record_motion("home_observe", np.eye(4), 1.0)
    backend = _MockPickBackend(recorder, observation)
    skill = PandaAtomicPickPlaceSkill(backend)
    pick = skill.pick(target, grasp["world_from_ee"])
    place = skill.place(target, destination) if pick.success else None
    success = bool(pick.success and place is not None and place.success)
    directory = output / "place"
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "place.mp4"
    motion = directory / "motion.json"
    recorder.write_mp4(video)
    recorder.write_motion_json(motion)
    payload: dict[str, object] = {
        "stage": "place",
        "status": "ok" if success else "error",
        "mode": "mock",
        "is_mock": True,
        "target": target,
        "destination": destination,
        "success": success,
        "inside_destination": success,
        "released": success,
        "stable": success,
        "pick_stages": list(pick.stages),
        "place_stages": [] if place is None else list(place.stages),
        "gripper_semantics": {"close": -1.0, "open": 1.0},
        "video_path": _relative(video, root),
        "motion_path": _relative(motion, root),
    }
    atomic_write_json(directory / "status.json", payload)
    return payload


def _run_real_place(
    output: Path,
    root: Path,
    target: str,
    destination: str,
    render_backend: str,
    motion_steps: int,
) -> dict[str, object]:
    from agentic_manipulation.envs.panda_runtime_scene import PandaSortingScene

    env = _make_real_env(render_backend)
    directory = output / "place"
    directory.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder()
    try:
        observation, _ = env.reset(seed=0, options={"reconfigure": True})
        base = env.unwrapped
        scene = PandaSortingScene(env)
        frame = CameraAdapter().capture(
            observation, observation["sensor_param"], "hand_camera", timestamp=0.0
        )
        segmentation_id = int(
            base.semantic_actors[target].per_scene_id[0].detach().cpu().item()
        )
        if frame.segmentation is None:
            raise ConfigurationError("real wrist frame has no segmentation")
        bbox = _bbox_for_segmentation(frame.segmentation, segmentation_id)
        cloud = write_point_cloud_bundle(
            directory / "recomputed_cloud", frame, bbox, segmentation_id
        )
        provider = CalibratedTopDownPandaGraspProvider(
            frame.world_from_camera,
            np.asarray(base.observation_home_pose)[:3, :3],
        )
        selected = provider.predict(
            np.load(cloud.target_npy, allow_pickle=False),
            np.load(cloud.workspace_npy, allow_pickle=False),
        )[0]
        world_from_ee = compose_panda_world_ee(
            frame.world_from_camera, selected.world_from_gripper
        )
        recorder.capture("home_observe", observation)
        recorder.record_motion(
            "home_observe",
            base.agent.tcp_pose.to_transformation_matrix()[0].detach().cpu().numpy(),
            1.0,
        )
        backend = PandaDeltaPoseBackend(env, recorder)
        skill = PandaAtomicPickPlaceSkill(
            backend,
            motion_steps=motion_steps,
            gripper_steps=20,
            settle_steps=30,
        )
        pick = skill.pick(target, world_from_ee)
        if not pick.success:
            skill.recover_after_failed_pick()
            place = None
        else:
            place = skill.place(target, destination)
        inside = scene.is_in_bin(target, destination)
        released = scene.is_released(target)
        stable = scene.is_stable(target)
        success = bool(
            pick.success
            and place is not None
            and place.success
            and inside
            and released
            and stable
        )
        video = directory / "place.mp4"
        motion = directory / "motion.json"
        recorder.write_mp4(video)
        recorder.write_motion_json(motion)
        payload: dict[str, object] = {
            "stage": "place",
            "status": "ok" if success else "error",
            "mode": "real",
            "is_mock": False,
            "model_is_mock": True,
            "grasp_source": "deterministic_calibrated_safety_fixture",
            "target": target,
            "destination": destination,
            "success": success,
            "inside_destination": inside,
            "released": released,
            "stable": stable,
            "camera_from_grasp": selected.world_from_gripper.tolist(),
            "world_from_ee": world_from_ee.tolist(),
            "pick_stages": list(pick.stages),
            "place_stages": [] if place is None else list(place.stages),
            "gripper_semantics": {"close": -1.0, "open": 1.0},
            "video_path": _relative(video, root),
            "motion_path": _relative(motion, root),
        }
        atomic_write_json(directory / "status.json", payload)
        if not success:
            raise ExecutionError(
                "real Panda place truth failed: "
                f"inside={inside}, released={released}, stable={stable}"
            )
        return payload
    finally:
        env.close()


def run_stage(
    *,
    stage: str,
    mode: str,
    target: str,
    output: str | Path,
    project_root: str | Path,
    checkpoint_path: str = "graspnet-baseline/checkpoint-rs.tar",
    device: str = "cuda",
    render_backend: str = "cpu",
    motion_steps: int = 40,
    destination: str = "white_bin",
) -> dict[str, object]:
    """Run one inspectable stage and persist its authoritative status JSON."""

    if stage not in STAGES:
        raise ConfigurationError(f"stage must be one of {STAGES}")
    if mode not in {"mock", "real"}:
        raise ConfigurationError("mode must be 'mock' or 'real'")
    if target not in GRASPABLE_INSTANCE_IDS:
        raise ConfigurationError(f"unknown graspable target: {target}")
    if destination not in DESTINATION_INSTANCE_IDS:
        raise ConfigurationError(f"unknown destination bin: {destination}")
    root = Path(project_root).resolve()
    output_path = resolve_project_path(root, output)
    try:
        if stage == "scene":
            return _run_scene(output_path, root, mode, target, render_backend)
        predecessor = _require_predecessor(output_path, stage, mode, target)
        del predecessor
        if stage == "pointcloud":
            return _run_pointcloud(output_path, root, mode, target)
        if stage == "grasp":
            return _run_grasp(
                output_path, root, mode, target, checkpoint_path, device
            )
        grasp = read_json(_status_path(output_path, "grasp"))
        if stage == "pick":
            if mode == "mock":
                return _run_mock_pick(output_path, root, target, grasp)
            return _run_real_pick(
                output_path, root, target, render_backend, motion_steps
            )
        if mode == "mock":
            return _run_mock_place(
                output_path, root, target, destination, grasp
            )
        return _run_real_place(
            output_path,
            root,
            target,
            destination,
            render_backend,
            motion_steps,
        )
    except AgenticManipulationError as exc:
        status_path = _status_path(output_path, stage)
        if not status_path.is_file():
            atomic_write_json(
                status_path,
                {
                    "stage": stage,
                    "status": "error",
                    "mode": mode,
                    "is_mock": mode == "mock",
                    "target": target,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        raise
