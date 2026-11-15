import json
from pathlib import Path
import tempfile
import unittest

from slam_dunk import Config
from slam_dunk.runtime import clean_environment
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_model_switches_map_to_native_flags(self):
        base = Config.load(ROOT / 'configs/full.json').native_environment()
        cases = {
            'baseline': ({'GYRO_REFINE'}, {'FACTORIAL_STATIC_CAMERA': '1'}),
            'camera_only': ({'GYRO_REFINE'}, {}),
            'gyro_only': (set(), {'FACTORIAL_STATIC_CAMERA': '1'}),
            'no_accelerometer': ({'ACC_ALIGN', 'ACC_SCALE', 'ACC_CROSS'}, {}),
            'no_clock': ({'TIME_REFINE'}, {}),
            'no_thermal': ({'THERMAL', 'BIAS_THERMAL'}, {}),
            'no_auxiliary': ({'ACC_ALIGN', 'ACC_SCALE', 'ACC_CROSS', 'TIME_REFINE', 'THERMAL', 'BIAS_THERMAL'}, {}),
            'ladder_0': ({'GYRO_REFINE', 'ACC_ALIGN', 'ACC_SCALE', 'ACC_CROSS', 'TIME_REFINE', 'THERMAL', 'BIAS_THERMAL'}, {'FACTORIAL_STATIC_CAMERA': '1'}),
            'ladder_1': ({'GYRO_REFINE', 'TIME_REFINE', 'THERMAL', 'BIAS_THERMAL'}, {'FACTORIAL_STATIC_CAMERA': '1'}),
            'ladder_2': ({'GYRO_REFINE', 'THERMAL', 'BIAS_THERMAL'}, {'FACTORIAL_STATIC_CAMERA': '1'}),
        }
        for name, (removed, added) in cases.items():
            with self.subTest(config=name):
                env = Config.load(ROOT / 'configs' / f'{name}.json').native_environment()
                self.assertEqual(env, dict({k: v for k, v in base.items() if k not in removed}, **added))

    def test_all_configs_load(self):
        for path in (ROOT / 'configs').glob('*.json'):
            Config.load(path)

    def test_strict_config_rejects_typos_and_non_booleans(self):
        for values in ({}, {'clock': True}, dict(Config().to_dict(), clok=False),
                       dict(Config().to_dict(), clock='false'), dict(Config().to_dict(), clock=0)):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(values))
                with self.assertRaises(ValueError):
                    Config.load(path)

    def test_shell_flags_cannot_enable_disabled_models(self):
        with patch.dict('os.environ', {'GYRO_REFINE': '1', 'FACTORIAL_STATIC_CAMERA': '0',
                                      'ARIA_FINAL_REFINE': '0', 'PYTHONPATH': '/stale'}):
            env = clean_environment()
        self.assertNotIn('GYRO_REFINE', env)
        self.assertNotIn('FACTORIAL_STATIC_CAMERA', env)
        self.assertNotIn('ARIA_FINAL_REFINE', env)
        self.assertNotIn('PYTHONPATH', env)
