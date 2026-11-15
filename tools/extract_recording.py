"""Extract native Aria stereo JPEGs and IMU streams without reading reference labels."""

import argparse
import json
from pathlib import Path
import shutil


IMU_HEADER = "#timestamp [ns],gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n"


def extract(vrs, calibration, output, *, primary_only=False, frame_limit=None):
    import pyvrs
    from projectaria_tools.core import data_provider

    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(calibration, output / "calibration.json")
    reader = pyvrs.SyncVRSReader(str(vrs))
    (output / "factory_calibration.json").write_text(reader.file_tags["calib_json"])
    counts = {}
    end_ns = None
    for camera, stream in [("cam0", "1201-1"), ("cam1", "1201-2")]:
        folder = output / camera
        (folder / "data").mkdir(parents=True)
        last = -1
        count = 0
        with (folder / "data.csv").open("w") as frames, (folder / "meta.csv").open("w") as meta:
            frames.write("#timestamp [ns],filename\n")
            meta.write("#timestamp [ns],exposure [s],gain,temperature [degC]\n")
            for record in reader.filtered_by_fields(stream_ids={stream}, record_types={"data"}):
                fields = record.metadata_blocks[0]
                timestamp = int(fields["capture_timestamp_ns"])
                if timestamp <= last:
                    raise ValueError(f"{camera} timestamps are not strictly increasing")
                image = bytes(record.image_blocks[0])
                if not image.startswith(b"\xff\xd8"):
                    raise ValueError(f"{camera} frame {timestamp} is not a native JPEG")
                (folder / "data" / f"{timestamp}.jpg").write_bytes(image)
                frames.write(f"{timestamp},{timestamp}.jpg\n")
                meta.write(
                    f"{timestamp},{fields['exposure_duration_s']:.9g},"
                    f"{fields['gain']:.6g},{fields['temperature_deg_c']:.4g}\n"
                )
                last = timestamp
                count += 1
                if frame_limit is not None and count >= frame_limit:
                    break
        if count < 2:
            raise ValueError(f"{camera} has fewer than two images")
        counts[camera] = count
        if frame_limit is not None:
            end_ns = max(end_ns or 0, last + 100_000_000)

    provider = data_provider.create_vrs_data_provider(str(vrs))
    for folder, label in [("imu0", "imu-right"), ("imu1", "imu-left")]:
        if primary_only and folder == "imu1":
            continue
        stream = provider.get_stream_id_from_label(label)
        if stream is None or provider.get_num_data(stream) == 0:
            if folder == "imu0":
                raise ValueError("Recording has no native imu-right stream")
            continue
        (output / folder).mkdir()
        last = -1
        count = 0
        with (output / folder / "data.csv").open("w") as csv:
            csv.write(IMU_HEADER)
            for index in range(provider.get_num_data(stream)):
                sample = provider.get_imu_data_by_index(stream, index)
                if not (sample.accel_valid and sample.gyro_valid):
                    continue
                timestamp = int(sample.capture_timestamp_ns)
                if end_ns is not None and timestamp > end_ns:
                    break
                if timestamp <= last:
                    continue
                gyro, accel = sample.gyro_radsec, sample.accel_msec2
                csv.write(
                    f"{timestamp},{gyro[0]:.9g},{gyro[1]:.9g},{gyro[2]:.9g},"
                    f"{accel[0]:.9g},{accel[1]:.9g},{accel[2]:.9g}\n"
                )
                last = timestamp
                count += 1
        if count < 2:
            raise ValueError(f"{label} has fewer than two valid samples")
        counts[folder] = count
    report = dict(
        stream_counts=counts,
        primary_only=primary_only,
        frame_limit=frame_limit,
        reference_labels_read=False,
    )
    (output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vrs", type=Path)
    parser.add_argument("output", type=Path, help="New prepared recording directory")
    parser.add_argument(
        "--calibration", type=Path, required=True, help="Matching benchmark camera calibration JSON"
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Extract only imu-right for a primary-only input experiment",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        help="Extract a small camera prefix for debugging, not full-record evaluation",
    )
    args = parser.parse_args()
    if args.frame_limit is not None and args.frame_limit < 2:
        parser.error("--frame-limit must be at least 2")
    print(
        json.dumps(
            extract(
                args.vrs.resolve(),
                args.calibration.resolve(),
                args.output.resolve(),
                primary_only=args.primary_only,
                frame_limit=args.frame_limit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
