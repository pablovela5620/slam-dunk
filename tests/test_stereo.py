import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from slam_dunk.backend import packet_files
from slam_dunk.stereo import export_packet, require_stereo_packet, _swap_packet


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'slam_dunk/engine' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class StereoTests(unittest.TestCase):
    def test_packet_permutation_preserves_measurement_rows_and_is_invertible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frontend, output = (root / x for x in ('recording', 'frontend', 'packet'))
            recording.mkdir(); frontend.mkdir(); (recording / 'cam0').mkdir()
            calibration = {'cam0': {'sensor': 'moving'}, 'cam1': {'sensor': 'reference'}}
            (recording / 'calibration.json').write_text(json.dumps(calibration))
            (recording / 'cam0/data.csv').write_text('100,a.jpg\n200,b.jpg\n')
            (frontend / 'calib.json').write_text(json.dumps([{'sensor': 'moving'}, {'sensor': 'reference'}]))
            for name in ('imu.bin', 'vio.txt', 'export.npz'):
                (frontend / name).write_bytes(b'unchanged')
            n, p, o, i = 3, 2, 6, 4
            K = np.arange(32, dtype=float)
            ext = np.arange(14, dtype=float)
            poses_and_points = np.arange(11*n+3*p, dtype=float)
            observations = np.arange(5*o, dtype=float).reshape(o,5)
            observations[:,1] = [0,1,1,0,1,0]
            imu = np.arange(7*i, dtype=float)
            with (frontend / 'input.bin').open('wb') as f:
                np.array([n,p,o,i],np.int64).tofile(f)
                for values in (K,ext,poses_and_points,observations,imu):
                    values.tofile(f)
            meta = np.arange(2+7*n,dtype=float)
            meta.tofile(frontend / 'input.bin.meta')
            clock = np.arange(2*n,dtype=float).reshape(n,2)
            clock.tofile(frontend / 'input.bin.camtime')
            candidates = np.arange(32,dtype=float).reshape(4,8)
            candidates[:,2] = [0,1,0,1]
            with (frontend / 'input.bin.candidates').open('wb') as f:
                np.array([p,4],np.int64).tofile(f)
                np.array([-1,1],np.int64).tofile(f)
                candidates.tofile(f)
            for name in ('weights','obsweights','candidateweights'):
                (frontend / ('input.bin.'+name)).write_bytes(b'unchanged')
            original = {x.name:x.read_bytes() for x in frontend.iterdir()}
            with self.assertRaisesRegex(ValueError, 'rerun prepare'):
                require_stereo_packet(frontend)
            export_packet(recording,frontend,output)
            files = packet_files(output)
            self.assertIn(output/'input.bin.stereo',files)
            self.assertIn(output/'frame_times.csv',files)
            body = np.fromfile(output/'input.bin',dtype=float,offset=32)
            assert_array_equal(body[:32].reshape(2,16),K.reshape(2,16)[::-1])
            assert_array_equal(body[32:46].reshape(2,7),ext.reshape(2,7)[::-1])
            start=46+11*n+3*p
            expected=observations.copy();expected[:,1]=1-expected[:,1]
            assert_array_equal(body[start:start+5*o].reshape(o,5),expected)
            actual_meta=np.fromfile(output/'input.bin.meta',dtype=float)
            assert_array_equal(actual_meta[:2],meta[:2][::-1])
            assert_array_equal(actual_meta[2:].reshape(n,7),meta[2:].reshape(n,7)[:,[1,0,2,3,4,6,5]])
            assert_array_equal(np.fromfile(output/'input.bin.camtime',dtype=float).reshape(n,2),clock[:,::-1])
            actual_candidates=np.fromfile(output/'input.bin.candidates',dtype=float,offset=16+8*p).reshape(4,8)
            expected_candidates=candidates.copy();expected_candidates[:,2]=1-expected_candidates[:,2]
            assert_array_equal(actual_candidates,expected_candidates)
            self.assertEqual(json.loads((output/'calibration.json').read_text())['cam0']['sensor'],'reference')
            self.assertEqual(json.loads((output/'calib.json').read_text())[1]['sensor'],'moving')
            self.assertEqual((output/'frame_times.csv').read_bytes(),(recording/'cam0/data.csv').read_bytes())
            # Applying the permutation twice recovers every original binary byte,
            # including poses/points, row order and all precision weights.
            _swap_packet(output/'input.bin')
            for name,content in original.items():
                if name.startswith('input.bin'):
                    self.assertEqual((output/name).read_bytes(),content,name)
            self.assertEqual({x.name:x.read_bytes() for x in frontend.iterdir()},original)

    def test_dense_orientation_uses_composed_camera1_transform(self):
        nominal_rig=module('nominal_rig')
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            reference=Rotation.from_rotvec([.4,-.2,.3])
            nominal=Rotation.from_rotvec([-.3,.1,.15])
            relative=reference.inv()*nominal*Rotation.from_rotvec([.001,-.002,.003])
            calibration={'cam0':{'T_b_s':{'qvec':reference.as_quat().tolist()}},
                         'cam1':{'T_b_s':{'qvec':nominal.as_quat().tolist()}}}
            (work/'calibration.json').write_text(json.dumps(calibration))
            (work/'ba.txt').write_text('1000000000 0 0 0 0 0 0 1\n')
            diag=np.zeros((2,14));diag[:,0]=[0,1]
            diag[:,7:11]=relative.as_quat()[[3,0,1,2]]
            np.savetxt(work/'ba.txt.diag',diag)
            frames=np.array([1,2,6,7])*10**9
            p=np.arange(12,dtype=float).reshape(4,3)
            q=Rotation.from_rotvec(np.arange(12).reshape(4,3)*.01).as_quat()
            actual_p,actual_q,_=nominal_rig.fit(work,work,frames,p,q)
            # Independent expression of the previous composed-body export.
            delta=(reference*relative)*nominal.inv()
            expected=(Rotation.from_quat(q)*Rotation.from_rotvec(.5*delta.as_rotvec())).as_quat()
            assert_array_equal(actual_p,p)
            assert_allclose(actual_q,expected,atol=1e-14)


if __name__ == '__main__':
    unittest.main()
