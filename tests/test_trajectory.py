from pathlib import Path
import tempfile
import unittest

from slam_dunk.trajectory import validate_trajectory


class TrajectoryTests(unittest.TestCase):
    def check_rows(self, rows, expected_error=False):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            # Adjacent ns at epoch scale must not collapse through float64.
            (directory / 'frames.csv').write_text('1700000000000000000,a.jpg\n1700000000000000001,b.jpg\n')
            (directory / 'poses.txt').write_text(rows)
            if expected_error:
                with self.assertRaises((ValueError, OverflowError)):
                    validate_trajectory(directory / 'frames.csv', directory / 'poses.txt')
            else:
                self.assertTrue(validate_trajectory(directory / 'frames.csv', directory / 'poses.txt')['exact_grid'])

    def test_exact_integer_nanoseconds(self):
        self.check_rows('1700000000000000000 0 0 0 0 0 0 1\n1700000000000000001 1 0 0 0 0 0 1\n')

    def test_wrong_grid_is_rejected(self):
        self.check_rows('1700000000000000000 0 0 0 0 0 0 1\n1700000000000000000 1 0 0 0 0 0 1\n', True)

    def test_nonfinite_and_invalid_quaternions(self):
        for value in ('nan', '0'):
            self.check_rows(f'1700000000000000000 0 0 0 0 0 0 {value}\n1700000000000000001 1 0 0 0 0 0 1\n', True)

    def test_empty_or_malformed(self):
        for rows in ('', '1700000000000000000 0 0 0\n'):
            self.check_rows(rows, True)
