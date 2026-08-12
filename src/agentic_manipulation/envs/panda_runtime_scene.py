"""Runtime state/camera adapter for the registered Panda sorting scene."""

from __future__ import annotations

from typing import Any

from agentic_manipulation.errors import ConfigurationError
from agentic_manipulation.perception.camera import CameraAdapter
from agentic_manipulation.types import CameraFrame


class PandaSortingScene:
    """Expose only calibrated wrist RGB-D to the V2 Panda agent."""

    def __init__(self, env: Any, camera_uid: str = "hand_camera") -> None:
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if self.base_env.robot_uids != "panda_wristcam":
            raise ConfigurationError("PandaSortingScene requires panda_wristcam")
        if camera_uid not in getattr(self.base_env, "_sensors", {}):
            raise ConfigurationError(f"Panda camera is unavailable: {camera_uid}")
        self.camera_uid = camera_uid
        self._camera = CameraAdapter()

    def capture(self) -> CameraFrame:
        observation = self.base_env.get_obs()
        return self._camera.capture(
            observation,
            observation["sensor_param"],
            self.camera_uid,
        )

    def close(self) -> None:
        self.env.close()
