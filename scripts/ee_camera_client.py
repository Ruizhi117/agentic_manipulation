#!/usr/bin/env python
"""Example external client for the EE Camera Demo.

Sends target end-effector poses via the file-based IPC protocol,
waits for the server to finish executing, then reads back RGBD images.

Usage:
  python scripts/ee_camera_client.py
  python scripts/ee_camera_client.py --pose "0.0,0.0,0.20" --steps 60
  python scripts/ee_camera_client.py --loop --interval 3.0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentic_manipulation.demo.ee_protocol import (  # noqa: E402
    EECommand,
    parse_ee_command,
)
from agentic_manipulation.demo.protocol import (  # noqa: E402
    atomic_write_json,
    read_json,
    resolve_project_path,
)
from agentic_manipulation.errors import ConfigurationError  # noqa: E402


_IPC_DIR = _PROJECT_ROOT / "temp" / "ee"
_COMMAND_FILE = _IPC_DIR / "command.json"
_RESPONSE_FILE = _IPC_DIR / "response.json"


# ---------------------------------------------------------------------------
#  IPC helpers
# ---------------------------------------------------------------------------


def configure_ipc_dir(value: str | Path) -> None:
    global _IPC_DIR, _COMMAND_FILE, _RESPONSE_FILE
    _IPC_DIR = resolve_project_path(_PROJECT_ROOT, value)
    _COMMAND_FILE = _IPC_DIR / "command.json"
    _RESPONSE_FILE = _IPC_DIR / "response.json"


def read_response() -> Optional[dict]:
    if not _RESPONSE_FILE.exists():
        return None
    try:
        return read_json(_RESPONSE_FILE)
    except ConfigurationError:
        return None


def send_command(
    target_pose: np.ndarray,
    command_id: int,
    gripper: float = 0.0,
    steps: int = 50,
) -> None:
    """Write a command that the demo server will pick up."""
    _IPC_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "command_id": command_id,
        "target_pose": target_pose.tolist(),
        "gripper": gripper,
        "steps": steps,
    }
    atomic_write_json(_COMMAND_FILE, payload)


def wait_for_response(expected_command_id: int, timeout: float = 60.0) -> Optional[dict]:
    """Poll until the server responds with our command_id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = read_response()
        if resp is not None and resp.get("command_id") == expected_command_id:
            return resp
        time.sleep(0.1)
    return None


def load_trajectory(path: Path) -> tuple[EECommand, ...]:
    """Load ordered EE waypoints and assign command IDs starting at one."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"failed to read trajectory {source}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ConfigurationError("trajectory must be a nonempty JSON array")
    commands: list[EECommand] = []
    for index, waypoint in enumerate(value, start=1):
        if not isinstance(waypoint, dict):
            raise ConfigurationError(f"trajectory waypoint {index} must be an object")
        commands.append(
            parse_ee_command(
                {
                    "command_id": index,
                    "target_pose": waypoint.get("target_pose"),
                    "gripper": waypoint.get("gripper", 0.0),
                    "steps": waypoint.get("steps", 50),
                }
            )
        )
    return tuple(commands)


# ---------------------------------------------------------------------------
#  Pose helpers
# ---------------------------------------------------------------------------


def _make_pose(x: float, y: float, z: float) -> np.ndarray:
    """Build a 4×4 world-from-ee pose with the gripper pointing down (-Z)."""
    pose = np.eye(4, dtype=np.float64)
    # Gripper approach = world -Z  → TCP Z points down
    pose[0, 0] = 1.0
    pose[1, 1] = -1.0
    pose[2, 2] = -1.0
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


# ---------------------------------------------------------------------------
#  Response display
# ---------------------------------------------------------------------------


def show_response(resp: dict, *, show_image: bool = False) -> None:
    """Print response summary and optionally open its RGB image."""
    status = resp.get("status", "???")
    msg = resp.get("message", "")
    print(f"  Status: {status}")
    if msg:
        print(f"  Message: {msg}")

    ee = resp.get("ee_pose", [])
    if ee and len(ee) >= 3:
        x, y, z = ee[0][3], ee[1][3], ee[2][3]
        print(f"  EE TCP: [{x:.4f}, {y:.4f}, {z:.4f}]")

    rgb_rel = resp.get("rgb_path", "")
    if rgb_rel:
        rgb_path = _PROJECT_ROOT / rgb_rel.replace("\\", "/")
        if rgb_path.exists() and rgb_path.stat().st_size > 0:
            try:
                from PIL import Image
                img = Image.open(str(rgb_path))
                print(f"  RGB: {img.size[0]}×{img.size[1]}")
                if show_image:
                    try:
                        img.show()
                    except Exception:
                        pass
            except Exception as exc:
                print(f"  RGB: read error — {exc}")
        else:
            print(f"  RGB: MISSING or empty  ({rgb_path})")

    depth_rel = resp.get("depth_path", "")
    if depth_rel:
        depth_path = _PROJECT_ROOT / depth_rel.replace("\\", "/")
        if depth_path.exists() and depth_path.stat().st_size > 0:
            try:
                depth = np.load(str(depth_path))
                print(f"  Depth: {depth.shape}, range [{depth.min():.4f}, {depth.max():.4f}] m")
            except Exception as exc:
                print(f"  Depth: read error — {exc}")
        else:
            print(f"  Depth: MISSING or empty")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="EE Camera Demo Client")
    parser.add_argument("--pose", type=str, default=None,
                        help="Target 'x,y,z' (metres).  Default: canned poses.")
    parser.add_argument("--gripper", type=float, default=0.0,
                        help="Gripper opening.  xarm6: 0=closed, 0.81=open.  "
                             "panda: -1=closed, 1=open.")
    parser.add_argument("--steps", type=int, default=60,
                        help="Interpolation steps (default 60).")
    parser.add_argument("--loop", action="store_true",
                        help="Send a sequence of canned poses in a loop.")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between loop poses (default 3.0).")
    parser.add_argument(
        "--trajectory",
        help="Project-relative JSON array of target_pose/gripper/steps waypoints.",
    )
    parser.add_argument(
        "--ipc-dir",
        default="temp/ee",
        help="Project-relative file IPC directory (default: temp/ee)",
    )
    parser.add_argument("--show-image", action="store_true")
    args = parser.parse_args(argv)
    configure_ipc_dir(args.ipc_dir)

    command_id = 0

    if args.trajectory:
        try:
            trajectory_path = resolve_project_path(_PROJECT_ROOT, args.trajectory)
            commands = load_trajectory(trajectory_path)
        except ConfigurationError as exc:
            print(f"Trajectory error: {exc}")
            return 2
        for command in commands:
            print(f"Sending trajectory waypoint #{command.command_id} ...")
            send_command(
                command.target_pose,
                command.command_id,
                gripper=command.gripper,
                steps=command.steps,
            )
            response = wait_for_response(
                command.command_id,
                timeout=float(command.steps) * 0.15 + 10.0,
            )
            if response is None:
                print(f"  No response for waypoint #{command.command_id}")
                return 2
            show_response(response, show_image=args.show_image)
            if response.get("status") != "ok":
                return 2
        return 0

    if args.pose:
        try:
            parts = [float(v.strip()) for v in args.pose.split(",")]
        except ValueError:
            print("Error: --pose requires three finite numbers 'x,y,z'")
            return 2
        if len(parts) != 3:
            print("Error: --pose requires 'x,y,z'")
            return 2
        if not np.isfinite(parts).all():
            print("Error: --pose requires finite values")
            return 2
        target = _make_pose(*parts)
        command_id += 1
        print(f"Sending pose → [{parts[0]}, {parts[1]}, {parts[2]}]  "
              f"(cmd #{command_id}) ...")
        send_command(target, command_id, gripper=args.gripper, steps=args.steps)
        resp = wait_for_response(command_id, timeout=float(args.steps) * 0.15 + 10.0)
        if resp:
            show_response(resp, show_image=args.show_image)
            return 0 if resp.get("status") == "ok" else 2
        else:
            print("  No response — is the demo server running?")
            return 2

    elif args.loop:
        canned = [
            ("Above plate centre", _make_pose(0.0, 0.0, 0.20)),
            ("Above red cube", _make_pose(0.08, 0.08, 0.18)),
            ("Above blue sphere", _make_pose(-0.05, 0.10, 0.18)),
            ("Above yellow cylinder", _make_pose(0.10, -0.05, 0.18)),
        ]
        try:
            while True:
                for label, pose in canned:
                    command_id += 1
                    print(f"\n--- Cmd #{command_id}: {label} ---")
                    send_command(pose, command_id, gripper=0.0, steps=60)
                    resp = wait_for_response(command_id, timeout=15.0)
                    if resp:
                        show_response(resp, show_image=args.show_image)
                    else:
                        print("  No response. Is the server running?")
                        print(f"    python scripts/ee_camera_demo.py --render-mode human")
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nClient stopped.")
        return 0

    else:
        target = _make_pose(0.0, 0.0, 0.20)
        command_id += 1
        print(f"Sending default pose (above plate at z=0.20, cmd #{command_id}) ...")
        send_command(target, command_id, gripper=0.0, steps=60)
        resp = wait_for_response(command_id, timeout=15.0)
        if resp:
            show_response(resp, show_image=args.show_image)
            return 0 if resp.get("status") == "ok" else 2
        else:
            print("  No response.  Start the server first:")
            print("    python scripts/ee_camera_demo.py --render-mode human")
            print("  Then re-run this client.")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
