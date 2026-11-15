"""Explicit model switches for the paper's Full R17-C backend."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    temporal_camera: bool = True
    gyro_response: bool = True
    accelerometer: bool = True
    clock: bool = True
    thermal: bool = True
    final_refinement: bool = True

    def __post_init__(self):
        for field in fields(self):
            if type(getattr(self, field.name)) is not bool:
                raise ValueError(f"{field.name} must be a JSON boolean")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("Config must be a JSON object")
        expected = {field.name for field in fields(cls)}
        if values.keys() != expected:
            raise ValueError(
                f"Config keys: missing {sorted(expected-values.keys())}, unknown {sorted(values.keys()-expected)}"
            )
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    def native_environment(self):
        """Native feature flags test presence: OFF must mean absent, never '0'."""
        env = json.loads(Path(__file__).with_name("native_defaults.json").read_text())
        env["SAVE_STATE"] = "1"
        if self.gyro_response:
            env["GYRO_REFINE"] = "1"
        if not self.temporal_camera:
            env["FACTORIAL_STATIC_CAMERA"] = "1"
        groups = {
            "accelerometer": ("ACC_ALIGN", "ACC_SCALE", "ACC_CROSS"),
            "clock": ("TIME_REFINE",),
            "thermal": ("THERMAL", "BIAS_THERMAL"),
            "final_refinement": ("FINAL_REFINE",),
        }
        for name, flags in groups.items():
            if not getattr(self, name):
                for flag in flags:
                    env.pop(flag, None)
        return env
