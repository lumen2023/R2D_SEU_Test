import importlib
import os
import sys

import gymnasium as gym
import numpy as np


DEFAULT_METADRIVE_SOURCE = "/home/ac/@Lyz-Code/safeRL-metadrive/metadrive"


def make_metadrive_env(
    split="train",
    seed=0,
    action_repeat=1,
    metadrive_source=DEFAULT_METADRIVE_SOURCE,
    num_scenarios=None,
    start_seed=None,
    extra_config=None,
):
    """Create the state-only SafeMetaDrive wrapper used by r2Dreamer."""
    return MetaDrive(
        "safe",
        action_repeat=action_repeat,
        seed=seed,
        split=split,
        metadrive_source=metadrive_source,
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        extra_config=extra_config,
    )


def _maybe_add_metadrive_source(metadrive_source):
    source = os.path.expanduser(metadrive_source or DEFAULT_METADRIVE_SOURCE)
    if os.path.isdir(os.path.join(source, "metadrive")) and source not in sys.path:
        sys.path.insert(0, source)


def _load_safe_env(metadrive_source):
    try:
        importlib.import_module("metadrive.envs.safe_metadrive_env")
    except ImportError:
        _maybe_add_metadrive_source(metadrive_source)

    from envs.metadrive_safe_env import SafeMetaDriveEnv_mini, TRAINING_CONFIG, VALIDATION_CONFIG

    return SafeMetaDriveEnv_mini, TRAINING_CONFIG, VALIDATION_CONFIG


class MetaDrive(gym.Env):
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        seed=0,
        split="train",
        metadrive_source=None,
        num_scenarios=None,
        start_seed=None,
        extra_config=None,
    ):
        if name != "safe":
            raise NotImplementedError(name)

        safe_env_cls, train_config, eval_config = _load_safe_env(metadrive_source)
        base_config = dict(train_config if split == "train" else eval_config)
        if num_scenarios is not None:
            base_config["num_scenarios"] = int(num_scenarios)
        if start_seed is not None:
            base_config["start_seed"] = int(start_seed)
        if extra_config:
            base_config.update(dict(extra_config))

        self._env = safe_env_cls(base_config)
        self._action_repeat = int(action_repeat)
        self._rng = np.random.RandomState(int(seed))
        self._start_seed = int(base_config["start_seed"])
        self._num_scenarios = int(base_config["num_scenarios"])
        self._episode_has_crash = False
        self._episode_crash_vehicle = False
        self._episode_crash_object = False
        self._episode_out_of_road = False
        self.reward_range = [-np.inf, np.inf]

        if not isinstance(self._env.action_space, gym.spaces.Box):
            raise TypeError(f"MetaDrive requires a continuous action space, got {self._env.action_space!r}.")
        if tuple(self._env.action_space.shape) != (2,):
            raise ValueError(f"Expected MetaDrive action shape (2,), got {self._env.action_space.shape}.")

    @property
    def observation_space(self):
        spaces = {
            "state": self._env.observation_space,
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        for key in (
            "log_cost",
            "log_event_cost",
            "log_risk_field_cost",
            "log_risk_field_event_equivalent_cost",
            "log_success",
            "log_safe_success",
            "log_safe_route_completion",
            "log_route_completion",
            "log_crash_vehicle",
            "log_crash_object",
            "log_out_of_road",
            "log_max_step",
        ):
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Box(
            self._env.action_space.low,
            self._env.action_space.high,
            dtype=np.float32,
        )

    def step(self, action):
        assert np.isfinite(action).all(), action
        total_reward = 0.0
        obs = None
        info = {}
        terminated = False
        truncated = False
        accumulated_logs = {
            "log_cost": 0.0,
            "log_event_cost": 0.0,
            "log_risk_field_cost": 0.0,
            "log_risk_field_event_equivalent_cost": 0.0,
        }

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += float(reward)
            accumulated_logs["log_cost"] += float(info.get("cost", 0.0))
            accumulated_logs["log_event_cost"] += float(info.get("event_cost", 0.0))
            accumulated_logs["log_risk_field_cost"] += float(info.get("risk_field_cost", 0.0))
            accumulated_logs["log_risk_field_event_equivalent_cost"] += float(
                info.get("risk_field_event_equivalent_cost", 0.0)
            )
            self._update_episode_safety(info)
            if terminated or truncated:
                break

        done = bool(terminated or truncated)
        return (
            self._format_obs(
                obs,
                info,
                is_first=False,
                is_last=done,
                is_terminal=bool(terminated),
                accumulated_logs=accumulated_logs,
            ),
            np.float32(total_reward),
            done,
            info,
        )

    def reset(self, **kwargs):
        self._episode_has_crash = False
        self._episode_crash_vehicle = False
        self._episode_crash_object = False
        self._episode_out_of_road = False
        force_seed = kwargs.pop("seed", None)
        if force_seed is None:
            force_seed = self._start_seed + self._rng.randint(self._num_scenarios)
        obs, info = self._env.reset(seed=int(force_seed), **kwargs)
        return self._format_obs(
            obs,
            info,
            is_first=True,
            is_last=False,
            is_terminal=False,
            accumulated_logs=None,
        )

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

    def _format_obs(self, obs, info, *, is_first, is_last, is_terminal, accumulated_logs):
        if isinstance(obs, dict):
            if "state" not in obs:
                raise KeyError("MetaDrive dict observations must contain a 'state' key for state-only training.")
            state = obs["state"]
        else:
            state = obs

        logs = self._step_logs(accumulated_logs)
        if is_last:
            logs.update(self._episode_logs(info))

        return {
            "state": np.asarray(state, dtype=np.float32),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            **{key: np.array([value], dtype=np.float32) for key, value in logs.items()},
        }

    @staticmethod
    def _step_logs(accumulated_logs):
        logs = {
            "log_cost": 0.0,
            "log_event_cost": 0.0,
            "log_risk_field_cost": 0.0,
            "log_risk_field_event_equivalent_cost": 0.0,
            "log_success": 0.0,
            "log_safe_success": 0.0,
            "log_safe_route_completion": 0.0,
            "log_route_completion": 0.0,
            "log_crash_vehicle": 0.0,
            "log_crash_object": 0.0,
            "log_out_of_road": 0.0,
            "log_max_step": 0.0,
        }
        if accumulated_logs:
            logs.update(accumulated_logs)
        return logs

    def _episode_logs(self, info):
        arrived = bool(info.get("arrive_dest", False))
        crash_vehicle = bool(info.get("crash_vehicle", False))
        crash_object = bool(info.get("crash_object", False))
        crash_building = bool(info.get("crash_building", False))
        crash_human = bool(info.get("crash_human", False))
        out_of_road = bool(info.get("out_of_road", False))
        max_step = bool(info.get("max_step", False))
        route_completion = float(info.get("route_completion", 0.0))
        unsafe_episode = bool(
            self._episode_has_crash
            or self._episode_out_of_road
            or crash_vehicle
            or crash_object
            or crash_building
            or crash_human
            or out_of_road
        )
        safe_success = arrived and not unsafe_episode
        return {
            "log_success": float(arrived),
            "log_safe_success": float(safe_success),
            "log_safe_route_completion": route_completion if not unsafe_episode else 0.0,
            "log_route_completion": route_completion,
            "log_crash_vehicle": float(self._episode_crash_vehicle or crash_vehicle),
            "log_crash_object": float(self._episode_crash_object or crash_object),
            "log_out_of_road": float(self._episode_out_of_road or out_of_road),
            "log_max_step": float(max_step),
        }

    def _update_episode_safety(self, info):
        self._episode_crash_vehicle = bool(self._episode_crash_vehicle or info.get("crash_vehicle", False))
        self._episode_crash_object = bool(self._episode_crash_object or info.get("crash_object", False))
        self._episode_has_crash = bool(
            self._episode_has_crash
            or info.get("crash_vehicle", False)
            or info.get("crash_object", False)
            or info.get("crash_building", False)
            or info.get("crash_human", False)
        )
        self._episode_out_of_road = bool(self._episode_out_of_road or info.get("out_of_road", False))
