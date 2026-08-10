#!/usr/bin/env python
"""EE Camera Demo Server — wrist-camera robot + file-based external control.

Supports ``xarm6_robotiq_wristcam`` and ``panda_wristcam``.

Images are saved **only** when a command is actively executing, at the
command rate (once per completed command). During idle the server sends
zero actions to hold position but writes nothing to disk.

Usage:
  python scripts/ee_camera_demo.py                                          # CPU headless, Panda
  python scripts/ee_camera_demo.py --robot panda_wristcam --render-mode human
  python scripts/ee_camera_demo.py --render-mode human --show-frustum
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure src/ is on PYTHONPATH so our custom env registration fires.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mani_skill.envs  # noqa: E402,F401 — registers ManiSkill built-in envs
import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402

# Import our custom env so it registers.
import agentic_manipulation.envs  # noqa: E402, F401
from agentic_manipulation.demo.ee_protocol import (  # noqa: E402
    EECommand,
    camera_response,
    parse_ee_command,
)
from agentic_manipulation.demo.agent_session import (  # noqa: E402
    AgentSessionPaths,
    mark_session_ready,
    mark_session_stopped,
    publish_agent_response,
    read_agent_request,
)
from agentic_manipulation.demo.debug_presenter import (  # noqa: E402
    Open3DGraspPresenter,
    OpenCVDebugPresenter,
)
from agentic_manipulation.demo.panda_agent_demo import (  # noqa: E402
    AgentDemoOptions,
    RealPandaAgentSession,
)
from agentic_manipulation.demo.protocol import (  # noqa: E402
    atomic_write_json,
    read_json,
    resolve_project_path,
)
from agentic_manipulation.errors import ConfigurationError  # noqa: E402
from agentic_manipulation.types import RuntimeEvent  # noqa: E402


# ---------------------------------------------------------------------------
#  IPC paths
# ---------------------------------------------------------------------------

_IPC_DIR = _PROJECT_ROOT / "temp" / "ee"
_COMMAND_FILE = _IPC_DIR / "command.json"
_RESPONSE_FILE = _IPC_DIR / "response.json"


def publish_runtime_progress(
    paths: AgentSessionPaths,
    request_id: int,
    event: RuntimeEvent,
    history: list[dict[str, object]],
) -> None:
    """Append one runtime event and publish the complete visible history."""

    record: dict[str, object] = {
        "event_id": len(history) + 1,
        "phase": event.type.value,
        "task_step": event.task_step,
        "message": event.message,
    }
    history.append(record)
    publish_agent_response(
        paths,
        request_id,
        {
            "status": "running",
            "progress_events": [dict(item) for item in history],
        },
    )

# ---------------------------------------------------------------------------
#  Per-robot action-space clipping  (pos_lo, pos_hi, rot_lo, rot_hi, grip_lo, grip_hi)
# ---------------------------------------------------------------------------

_ROBOT_ACTION_CLIP: dict[str, tuple[float, ...]] = {
    "xarm6_robotiq_wristcam": (-1.0, 1.0, -1.0, 1.0, 0.0, 0.81),
    "xarm6_robotiq":          (-1.0, 1.0, -1.0, 1.0, 0.0, 0.81),
    "panda_wristcam":         (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    "panda":                  (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
}

# ---------------------------------------------------------------------------
#  IPC helpers  (atomic-write: temp → rename)
# ---------------------------------------------------------------------------


def configure_ipc_dir(value: str | Path) -> None:
    global _IPC_DIR, _COMMAND_FILE, _RESPONSE_FILE
    _IPC_DIR = resolve_project_path(_PROJECT_ROOT, value)
    _COMMAND_FILE = _IPC_DIR / "command.json"
    _RESPONSE_FILE = _IPC_DIR / "response.json"


def ensure_ipc_dir() -> None:
    _IPC_DIR.mkdir(parents=True, exist_ok=True)


def read_command() -> Optional[EECommand]:
    if not _COMMAND_FILE.exists():
        return None
    try:
        data = read_json(_COMMAND_FILE)
        return parse_ee_command(data)
    finally:
        try:
            _COMMAND_FILE.unlink()
        except OSError:
            pass


def write_response(
    command_id: int,
    status: str,
    ee_pose: list,
    rgb_path: str,
    depth_path: str,
    intrinsic: object,
    world_from_camera: object,
    message: str = "",
) -> None:
    ensure_ipc_dir()
    payload = camera_response(
        command_id,
        ee_pose,
        rgb_path,
        depth_path,
        intrinsic,
        world_from_camera,
        is_mock=False,
        status=status,
        message=message,
    )
    atomic_write_json(_RESPONSE_FILE, payload)


def save_rgb(rgb: np.ndarray, command_id: int) -> str:
    """Save RGB to a **per-command** PNG.  Returns relative path."""
    ensure_ipc_dir()
    from PIL import Image
    path = _IPC_DIR / f"rgb_{command_id:06d}.png"
    Image.fromarray(rgb).save(str(path))
    return str(path.relative_to(_PROJECT_ROOT))


def save_depth(depth: np.ndarray, command_id: int) -> str:
    """Save depth to a **per-command** .npy.  Returns relative path."""
    ensure_ipc_dir()
    path = _IPC_DIR / f"depth_{command_id:06d}.npy"
    np.save(str(path), depth.astype(np.float32))
    return str(path.relative_to(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
#  Pose utilities
# ---------------------------------------------------------------------------


def _rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → XYZ intrinsic Euler angles."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def _pose_to_44(pose) -> np.ndarray:
    if hasattr(pose, "to_transformation_matrix"):
        mat = pose.to_transformation_matrix()
    elif hasattr(pose, "raw_pose"):
        mat = pose.raw_pose
    else:
        mat = pose
    mat = mat[0].detach().cpu().numpy() if hasattr(mat, "detach") else np.asarray(mat)
    return mat.reshape(4, 4).astype(np.float64)


def _pose_to_44_list(pose) -> list:
    return _pose_to_44(pose).tolist()


def compute_delta_action(
    current_tcp_44: np.ndarray,
    target_tcp_44: np.ndarray,
    gripper_target: float,
    clip: tuple[float, ...],
) -> np.ndarray:
    """7D ``pd_ee_delta_pose`` action."""
    delta_pos = target_tcp_44[:3, 3] - current_tcp_44[:3, 3]
    delta_R = target_tcp_44[:3, :3] @ current_tcp_44[:3, :3].T
    delta_euler = _rotmat_to_euler_xyz(delta_R)

    action = np.array([
        delta_pos[0], delta_pos[1], delta_pos[2],
        delta_euler[0], delta_euler[1], delta_euler[2],
        gripper_target,
    ], dtype=np.float32)

    pl, ph, rl, rh, gl, gh = clip
    action[0:3] = np.clip(action[0:3], pl, ph)
    action[3:6] = np.clip(action[3:6], rl, rh)
    action[6] = np.clip(action[6], gl, gh)
    return action


def interpolate_pose(start_44: np.ndarray, target_44: np.ndarray, frac: float) -> np.ndarray:
    """Linear position, slerp rotation.  *frac* ∈ [0, 1]."""
    from scipy.spatial.transform import Rotation, Slerp

    frac = float(np.clip(frac, 0.0, 1.0))
    interp_p = start_44[:3, 3] + frac * (target_44[:3, 3] - start_44[:3, 3])
    key_rots = Rotation.from_matrix([start_44[:3, :3], target_44[:3, :3]])
    interp_R = Slerp([0.0, 1.0], key_rots)([frac]).as_matrix()[0]

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = interp_R
    result[:3, 3] = interp_p
    return result


# ---------------------------------------------------------------------------
#  Camera frustum visualization  (human render mode only)
# ---------------------------------------------------------------------------

_FRUSTUM_COLOR = np.array([0.2, 1.0, 0.3, 0.9], dtype=np.float32)
_FRUSTUM_LINE_WIDTH = 2.0
_FRUSTUM_DEPTH = 0.20  # how far the frustum extends (metres)


def _build_frustum_vertices(
    width: int, height: int, fov: float, near: float, far_depth: float
) -> np.ndarray:
    """Return line-pair vertices (N×2, 3) for a camera frustum wireframe.

    Built in **camera-local** space where +Z is forward.
    """
    aspect = width / height if height > 0 else 1.0
    half_h = np.tan(fov / 2.0) * far_depth
    half_w = half_h * aspect

    n = near
    f = far_depth

    cn = np.array([
        [-half_w * n / f, -half_h * n / f, n],
        [ half_w * n / f, -half_h * n / f, n],
        [ half_w * n / f,  half_h * n / f, n],
        [-half_w * n / f,  half_h * n / f, n],
    ], dtype=np.float32)
    cf = np.array([
        [-half_w, -half_h, f],
        [ half_w, -half_h, f],
        [ half_w,  half_h, f],
        [-half_w,  half_h, f],
    ], dtype=np.float32)

    pairs = [
        (cn[0], cn[1]), (cn[1], cn[2]), (cn[2], cn[3]), (cn[3], cn[0]),
        (cf[0], cf[1]), (cf[1], cf[2]), (cf[2], cf[3]), (cf[3], cf[0]),
        (cn[0], cf[0]), (cn[1], cf[1]), (cn[2], cf[2]), (cn[3], cf[3]),
    ]
    verts = []
    for a, b in pairs:
        verts.append(a)
        verts.append(b)
    return np.array(verts, dtype=np.float32)


def create_camera_frustum(viewer, width: int, height: int, fov: float, near: float = 0.02):
    if viewer is None:
        return None
    try:
        verts = _build_frustum_vertices(width, height, fov, near, _FRUSTUM_DEPTH)
        colors = np.tile(_FRUSTUM_COLOR, (verts.shape[0], 1))
        ls = viewer.renderer_context.create_line_set(verts, colors)
        obj = viewer.render_scene.add_line_set(ls)
        obj.line_width = _FRUSTUM_LINE_WIDTH
        return obj
    except Exception:
        return None


def update_camera_frustum(frustum_obj, extrinsic_cv_44: np.ndarray) -> None:
    if frustum_obj is None:
        return
    world_from_camera = np.linalg.inv(extrinsic_cv_44)
    pos = world_from_camera[:3, 3]
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(world_from_camera[:3, :3])
    q_scipy = r.as_quat()  # [x, y, z, w]
    q_sapien = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]  # [w, x, y, z]
    frustum_obj.set_position(pos)
    frustum_obj.set_rotation(q_sapien)


def remove_camera_frustum(frustum_obj, viewer) -> None:
    if frustum_obj is not None and viewer is not None:
        try:
            viewer.render_scene.remove_node(frustum_obj)
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Camera data extraction
# ---------------------------------------------------------------------------


def extract_hand_camera(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract RGB (uint8 HWC) and depth (float32 HW, **metres**) from wrist cam."""
    sensor_data = obs.get("sensor_data", obs)
    if "hand_camera" not in sensor_data:
        raise KeyError("hand_camera not found — is the robot a wristcam variant?")

    cam = sensor_data["hand_camera"]
    rgb = cam["rgb"]
    depth = cam["depth"]

    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()

    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # ManiSkill minimal shader → int16 millimetres → float32 metres
    if depth.dtype == np.int16:
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)

    return rgb, depth


def _extrinsic_cv_44(obs: dict) -> np.ndarray:
    """Return 4×4 world→camera matrix for ``hand_camera``."""
    sp = obs.get("sensor_param", {})
    if "hand_camera" not in sp:
        raise KeyError("hand_camera params missing")
    ecv = sp["hand_camera"]["extrinsic_cv"]
    if hasattr(ecv, "detach"):
        ecv = ecv.detach().cpu().numpy()
    ecv = np.asarray(ecv, dtype=np.float64)
    if ecv.ndim == 3:
        ecv = ecv[0]
    out = np.eye(4, dtype=np.float64)
    out[:3] = ecv.reshape(3, 4)
    return out


def _intrinsic_33(obs: dict) -> np.ndarray:
    """Return the 3×3 OpenCV intrinsic matrix for ``hand_camera``."""
    sp = obs.get("sensor_param", {})
    if "hand_camera" not in sp:
        raise KeyError("hand_camera params missing")
    intrinsic = sp["hand_camera"]["intrinsic_cv"]
    if hasattr(intrinsic, "detach"):
        intrinsic = intrinsic.detach().cpu().numpy()
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.ndim == 3:
        intrinsic = intrinsic[0]
    return intrinsic.reshape(3, 3)


def _world_from_camera_44(obs: dict) -> np.ndarray:
    """Return camera pose in world coordinates."""
    return np.linalg.inv(_extrinsic_cv_44(obs))


def render_human_frame(
    env,
    frustum_obj,
    *,
    observation: dict | None = None,
    image_presenter: object | None = None,
    grasp_presenter: object | None = None,
) -> dict:
    """Refresh wrist-frustum pose, render, and keep both debug windows alive."""

    obs = env.unwrapped.get_obs() if observation is None else observation
    if frustum_obj is not None:
        try:
            update_camera_frustum(frustum_obj, _extrinsic_cv_44(obs))
        except (KeyError, ValueError, np.linalg.LinAlgError):
            pass
    env.render()
    for presenter in (image_presenter, grasp_presenter):
        if presenter is not None:
            presenter.pump()
    return obs


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    configure_ipc_dir(args.ipc_dir)
    ensure_ipc_dir()
    robot_uid: str = args.robot
    clip = _ROBOT_ACTION_CLIP.get(robot_uid, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))

    print(f"[ee_camera_demo] Robot: {robot_uid}")
    print(f"[ee_camera_demo] Render: mode={args.render_mode}, backend={args.render_backend}")
    print(f"[ee_camera_demo] Camera: {args.camera_width}x{args.camera_height}")
    print(f"[ee_camera_demo] Frustum: {args.show_frustum}")
    agent_enabled = robot_uid == "panda_wristcam" and not args.disable_agent
    agent_paths = AgentSessionPaths.from_value(_PROJECT_ROOT, args.agent_session_dir)

    env: gym.Env = gym.make(
        "EECameraScene-v1",
        robot_uids=robot_uid,
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb+depth+segmentation",

        reward_mode="none",

        render_mode=args.render_mode if args.render_mode != "none" else None,
        render_backend=args.render_backend,
        sim_backend="auto",
        num_envs=1,
        sensor_configs={
            "hand_camera": {"width": args.camera_width, "height": args.camera_height,"fov": np.deg2rad(90.0),},
            "scene_camera": {"width": args.camera_width, "height": args.camera_height},
        },
        human_render_camera_configs=dict(width=640, height=480),
        enable_shadow=True,
    )

    obs, _ = env.reset(seed=0, options=dict(reconfigure=True))
    print(f"[ee_camera_demo] Env ready.  Control mode: {env.unwrapped.control_mode}")
    print(f"[ee_camera_demo] Listening at: {_COMMAND_FILE}")
    if agent_enabled:
        mark_session_ready(agent_paths, pid=os.getpid())
        print(f"[ee_camera_demo] Agent session ready: {agent_paths.request}")
        print(f"[ee_camera_demo] Agent artifacts: {args.agent_output_root}")

    # --- Viewer & frustum (human mode) ---
    viewer = None
    frustum_obj = None
    if args.render_mode == "human":
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = False
        if args.show_frustum:
            frustum_obj = create_camera_frustum(
                viewer, args.camera_width, args.camera_height, np.pi / 2, near=0.02
            )
            if frustum_obj is not None:
                print("[ee_camera_demo] Camera frustum overlay active (green wireframe).")
    presenter = (
        OpenCVDebugPresenter()
        if agent_enabled and args.render_mode == "human" and not args.no_agent_popups
        else None
    )
    grasp_presenter = (
        Open3DGraspPresenter()
        if agent_enabled and args.render_mode == "human" and not args.no_agent_popups
        else None
    )
    agent_session: RealPandaAgentSession | None = None
    last_agent_request_id = 0
    if agent_enabled and agent_paths.response.is_file():
        try:
            previous_id = read_json(agent_paths.response).get("request_id", 0)
            if isinstance(previous_id, int) and not isinstance(previous_id, bool):
                last_agent_request_id = max(0, previous_id)
        except ConfigurationError:
            pass

    # --- Command state ---
    last_command_id: int = -1
    pending_target: Optional[np.ndarray] = None
    pending_steps: int = 0
    pending_gripper: float = 0.0
    current_step: int = 0
    start_pose: Optional[np.ndarray] = None
    command_active = False

    try:
        while True:
            # ---------------------------------------------------------------
            # 0. Run a high-level Agent request in this exact environment.
            # ---------------------------------------------------------------
            if agent_enabled and not command_active and agent_paths.request.is_file():
                agent_request = None
                try:
                    agent_request = read_agent_request(agent_paths)
                    if agent_request is None:
                        raise ConfigurationError("agent request disappeared while reading")
                    try:
                        agent_paths.request.unlink()
                    except OSError as exc:
                        raise ConfigurationError(
                            f"failed to consume agent request: {exc}"
                        ) from exc
                    if agent_request.request_id <= last_agent_request_id:
                        raise ConfigurationError(
                            "request_id must increase monotonically; "
                            f"last={last_agent_request_id}, got={agent_request.request_id}"
                        )
                    last_agent_request_id = agent_request.request_id
                    publish_agent_response(
                        agent_paths,
                        agent_request.request_id,
                        {"status": "running", "command": agent_request.command},
                    )
                    print(
                        f"[ee_camera_demo] Agent request #{agent_request.request_id}: "
                        f"{agent_request.command}"
                    )
                    if agent_session is None:
                        template = AgentDemoOptions(
                            mode="real",
                            command=agent_request.command,
                            seed=0,
                            max_retries=args.agent_max_retries,
                            output_root=args.agent_output_root,
                            render_backend=args.render_backend,
                            record=args.agent_record,
                            checkpoint=args.agent_checkpoint,
                            device=args.agent_device,
                        )
                        agent_session = RealPandaAgentSession(
                            template,
                            _PROJECT_ROOT,
                            env,
                            render_callback=(
                                (
                                    lambda: render_human_frame(
                                        env,
                                        frustum_obj,
                                        image_presenter=presenter,
                                        grasp_presenter=grasp_presenter,
                                    )
                                )
                                if args.render_mode == "human"
                                else None
                            ),
                            image_callback=presenter,
                            grasp_3d_callback=grasp_presenter,
                        )
                    progress_events: list[dict[str, object]] = []
                    status = agent_session.run(
                        agent_request.command,
                        event_callback=lambda event: publish_runtime_progress(
                            agent_paths,
                            agent_request.request_id,
                            event,
                            progress_events,
                        ),
                    )
                    status = dict(status)
                    status["progress_events"] = [
                        dict(item) for item in progress_events
                    ]
                    publish_agent_response(
                        agent_paths, agent_request.request_id, status
                    )
                    print(
                        f"[ee_camera_demo] Agent request #{agent_request.request_id} "
                        f"finished: {status['status']}; waiting for the next command."
                    )
                except ConfigurationError as exc:
                    print(f"[ee_camera_demo] Agent request rejected: {exc}")
                    if agent_paths.request.is_file():
                        try:
                            agent_paths.request.unlink()
                        except OSError:
                            pass
                    if agent_request is not None:
                        publish_agent_response(
                            agent_paths,
                            agent_request.request_id,
                            {"status": "error", "message": str(exc)},
                        )
                except Exception as exc:
                    print(f"[ee_camera_demo] Agent request failed: {exc}")
                    if agent_request is not None:
                        publish_agent_response(
                            agent_paths,
                            agent_request.request_id,
                            {"status": "error", "message": str(exc)},
                        )
                finally:
                    obs = env.unwrapped.get_obs()

            # ---------------------------------------------------------------
            # 1. Poll for new external command
            # ---------------------------------------------------------------
            try:
                cmd = read_command()
            except ConfigurationError as exc:
                atomic_write_json(
                    _RESPONSE_FILE,
                    {
                        "command_id": None,
                        "status": "error",
                        "message": str(exc),
                        "is_mock": False,
                    },
                )
                print(f"[ee_camera_demo] Rejected command: {exc}")
                cmd = None
            if cmd is not None and cmd.command_id != last_command_id:
                last_command_id = cmd.command_id
                start_pose = _pose_to_44(env.unwrapped.agent.tcp_pose)
                pending_target = cmd.target_pose
                pending_steps = max(cmd.steps, 1)
                pending_gripper = cmd.gripper
                current_step = 0
                command_active = True
                print(f"[ee_camera_demo] Cmd #{cmd.command_id}: "
                      f"steps={pending_steps}, gripper={pending_gripper:.3f}")

            # ---------------------------------------------------------------
            # 2. Compute action
            # ---------------------------------------------------------------
            if command_active and current_step < pending_steps:
                current_step += 1
                frac = current_step / pending_steps
                interp_target = interpolate_pose(start_pose, pending_target, frac)
                current_tcp = _pose_to_44(env.unwrapped.agent.tcp_pose)
                action = compute_delta_action(
                    current_tcp, interp_target, pending_gripper, clip
                )
            else:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  dtype=np.float32)
                command_active = False

            # ---------------------------------------------------------------
            # 3. Step environment
            # ---------------------------------------------------------------
            obs, _reward, _term, _trunc, _info = env.step(action)

            # ---------------------------------------------------------------
            # 4. Save camera images ONLY when a command just finished.
            #    During idle nothing is written to disk.
            # ---------------------------------------------------------------
            if command_active and current_step >= pending_steps:
                command_active = False
                try:
                    rgb, depth = extract_hand_camera(obs)
                    rgb_rel = save_rgb(rgb, last_command_id)
                    depth_rel = save_depth(depth, last_command_id)
                    ee = _pose_to_44_list(env.unwrapped.agent.tcp_pose)
                    write_response(
                        last_command_id,
                        "ok",
                        ee,
                        rgb_rel,
                        depth_rel,
                        _intrinsic_33(obs),
                        _world_from_camera_44(obs),
                    )
                    print(f"[ee_camera_demo] Cmd #{last_command_id} done, "
                          f"images saved.")
                except Exception as exc:
                    try:
                        intrinsic = _intrinsic_33(obs)
                        world_from_camera = _world_from_camera_44(obs)
                    except (KeyError, ValueError, np.linalg.LinAlgError):
                        intrinsic = np.eye(3, dtype=np.float64)
                        world_from_camera = np.eye(4, dtype=np.float64)
                    write_response(
                        last_command_id, "error",
                        _pose_to_44_list(env.unwrapped.agent.tcp_pose),
                        "", "", intrinsic, world_from_camera, str(exc),
                    )

            # ---------------------------------------------------------------
            # 5. Frustum update & human render
            # ---------------------------------------------------------------
            if args.render_mode == "human":
                render_human_frame(
                    env,
                    frustum_obj,
                    observation=obs,
                    image_presenter=presenter,
                    grasp_presenter=grasp_presenter,
                )
            elif presenter is not None:
                presenter.pump()

    except KeyboardInterrupt:
        print("\n[ee_camera_demo] Caught Ctrl+C, shutting down.")
    finally:
        if agent_enabled:
            mark_session_stopped(agent_paths, pid=os.getpid())
        if presenter is not None:
            presenter.close()
        if grasp_presenter is not None:
            grasp_presenter.close()
        remove_camera_frustum(frustum_obj, viewer)
        env.close()
        print("[ee_camera_demo] Env closed.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EE Camera Demo Server")
    parser.add_argument("--robot", default="panda_wristcam",
                        choices=["xarm6_robotiq_wristcam", "panda_wristcam",
                                 "xarm6_robotiq", "panda"],
                        help="Robot UID (default: panda_wristcam)")
    parser.add_argument("--render-mode", default="rgb_array",
                        choices=["rgb_array", "human", "none"])
    parser.add_argument("--render-backend", default="cpu",
                        choices=["cpu", "gpu"])
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--ipc-dir",
        default="temp/ee",
        help="Project-relative file IPC directory (default: temp/ee)",
    )
    parser.add_argument("--show-frustum", action="store_true",
                        help="Draw a green wireframe showing the wrist-camera "
                             "viewing frustum (human render mode only).")
    parser.add_argument(
        "--agent-session-dir",
        default="temp/panda_agent_session",
        help="Project-relative high-level Agent IPC directory.",
    )
    parser.add_argument(
        "--agent-output-root",
        default="temp/panda_agent",
        help="Project-relative directory for per-command Agent artifacts.",
    )
    parser.add_argument("--agent-max-retries", type=int, default=4)
    parser.add_argument(
        "--agent-checkpoint", default="graspnet-baseline/checkpoint-rs.tar"
    )
    parser.add_argument("--agent-device", default="cuda")
    parser.add_argument("--agent-record", action="store_true")
    parser.add_argument(
        "--disable-agent",
        action="store_true",
        help="Disable high-level Agent requests and retain only EE waypoint IPC.",
    )
    parser.add_argument(
        "--no-agent-popups",
        action="store_true",
        help="Save Agent overlays without opening OpenCV debug windows.",
    )
    main(parser.parse_args())
