"""
Download and prepare GWOSC strain data into the O1-style prepared directory layout.

This script:
- downloads a curated subset of 4 kHz HDF5 files directly from GWOSC archive URLs
- extracts clean 4-second strain segments for downstream edge-free central-window scoring
- splits clean off-event segments into train/validation/test noise sets
- optionally extracts sliding 4-second windows around supplied events for evaluation
- estimates a reference PSD from the prepared training noise segments

Prepared outputs are written under `data/o1_prepared_4s_crop2s/` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass

import h5py
import numpy as np


FS = 4096
DEFAULT_SEG_SECONDS = 4.0
ARCHIVE_BLOCK = 1048576
DEFAULT_RUN_NAME = "O1"

DEFAULT_O1_EVENTS = [
    {"name": "GW150914", "gps": 1126259462.4},
    {"name": "LVT151012", "gps": 1128678900.4},
    {"name": "GW151226", "gps": 1135136350.6},
]

DETECTORS = ("H1", "L1")
DEFAULT_NUM_TRAIN_BLOCKS = 32
GWOSC_RUN_STRAIN_API = "https://gwosc.org/api/v2/runs/{run_name}/strain-files"

DQ_BITS = {
    "DATA": 0,
    "CBC_CAT1": 1,
    "CBC_CAT2": 2,
}


@dataclass(frozen=True)
class FileSpec:
    detector: str
    gps_start: int
    run_name: str = DEFAULT_RUN_NAME

    @property
    def archive_dir(self) -> int:
        return (self.gps_start // ARCHIVE_BLOCK) * ARCHIVE_BLOCK

    @property
    def filename(self) -> str:
        prefix = "H" if self.detector == "H1" else "L"
        return f"{prefix}-{self.detector}_LOSC_4_V1-{self.gps_start}-4096.hdf5"

    @property
    def url(self) -> str:
        return f"https://gwosc.org/archive/data/{self.run_name}/{self.archive_dir}/{self.filename}"

    @staticmethod
    def from_filename(filename: str, run_name: str = DEFAULT_RUN_NAME) -> "FileSpec":
        match = re.match(r"^[HL]-(H1|L1)_LOSC_4_V1-(\d+)-4096\.hdf5$", filename)
        if not match:
            raise ValueError(f"Unrecognized raw LOSC filename: {filename}")
        return FileSpec(match.group(1), int(match.group(2)), run_name=run_name)


def event_file_start(gps_time: float) -> int:
    return int(math.floor(gps_time / 4096.0) * 4096)


def _fetch_json_with_retries(url: str) -> dict:
    payload = None
    last_error = None
    for delay in (0.0, 1.0, 3.0):
        if delay > 0.0:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_error = exc
    if payload is None:
        raise RuntimeError(f"Failed to query GWOSC API at {url}: {last_error}")
    return payload


def _row_download_url(row: dict) -> str:
    for key in ("download_url", "hdf5_url", "url"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"No downloadable HDF5 URL found in GWOSC row keys: {sorted(row.keys())}")


def fetch_available_run_files(run_name: str, detector: str) -> dict[int, str]:
    files_by_start: dict[int, str] = {}
    page = 1
    while True:
        url = (
            f"{GWOSC_RUN_STRAIN_API.format(run_name=run_name)}"
            f"?detector={detector}&sample-rate=4&duration=4096&file-format=hdf5&format=json&page={page}&page_size=100"
        )
        payload = _fetch_json_with_retries(url)
        results = payload.get("results", [])
        for row in results:
            gps_start = int(row["gps_start"])
            files_by_start[gps_start] = _row_download_url(row)
        if not payload.get("next"):
            break
        page += 1
    return files_by_start


def fetch_available_run_file_starts(run_name: str, detector: str) -> list[int]:
    return sorted(fetch_available_run_files(run_name, detector).keys())


def candidate_train_file_starts(run_name: str, events: list[dict[str, float]]) -> list[int]:
    event_starts = {event_file_start(event["gps"]) for event in events}
    detector_starts = [set(fetch_available_run_file_starts(run_name, det)) for det in DETECTORS]
    common_starts = sorted(set.intersection(*detector_starts))
    starts = [gps_start for gps_start in common_starts if gps_start not in event_starts]
    return starts


def choose_train_file_starts(run_name: str, events: list[dict[str, float]], num_blocks: int) -> list[int]:
    candidates = candidate_train_file_starts(run_name, events)
    if num_blocks <= 0:
        raise ValueError("--num-train-blocks must be positive")
    if num_blocks >= len(candidates):
        return candidates
    idx = np.linspace(0, len(candidates) - 1, num=num_blocks, dtype=int)
    starts = [candidates[i] for i in idx]
    return sorted(set(starts))


def required_noise_files(run_name: str, events: list[dict[str, float]], num_train_blocks: int) -> list[FileSpec]:
    train_starts = choose_train_file_starts(run_name, events, num_train_blocks)
    return [FileSpec(det, start, run_name=run_name) for start in train_starts for det in DETECTORS]


def required_event_files(run_name: str, events: list[dict[str, float]]) -> list[FileSpec]:
    starts = sorted({event_file_start(event["gps"]) for event in events})
    return [FileSpec(det, start, run_name=run_name) for start in starts for det in DETECTORS]


def ensure_download(spec: FileSpec, raw_dir: str, download_url: str | None = None) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    target = os.path.join(raw_dir, spec.filename)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"[skip] {spec.filename} already exists")
        return target

    source_url = download_url or spec.url
    print(f"[download] {source_url}")
    with urllib.request.urlopen(source_url) as response, open(target, "wb") as fout:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    return target


def discover_existing_raw_specs(raw_dir: str, run_name: str) -> list[FileSpec]:
    specs = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".hdf5"):
            continue
        try:
            spec = FileSpec.from_filename(name, run_name=run_name)
        except ValueError:
            continue
        path = os.path.join(raw_dir, name)
        try:
            with h5py.File(path, "r"):
                pass
        except Exception:
            print(f"[skip] ignoring incomplete or unreadable raw file: {name}")
            continue
        specs.append(spec)
    return specs


def _resolve_dataset(h5obj: h5py.File, candidates: list[str]) -> np.ndarray:
    for path in candidates:
        if path in h5obj:
            return h5obj[path][:]
    raise KeyError(f"None of the dataset paths exist: {candidates}")


def load_frame(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5f:
        strain = _resolve_dataset(h5f, ["strain/Strain", "/strain/Strain"]).astype(np.float64)
        dqmask = _resolve_dataset(h5f, ["quality/simple/DQmask", "/quality/simple/DQmask"]).astype(np.int64)
        injmask = _resolve_dataset(
            h5f,
            ["quality/injections/Injmask", "/quality/injections/Injmask"],
        ).astype(np.int64)
    return strain, dqmask, injmask


def has_required_dq(dqmask_1hz: np.ndarray) -> np.ndarray:
    good = np.ones_like(dqmask_1hz, dtype=bool)
    for bit in (DQ_BITS["DATA"], DQ_BITS["CBC_CAT1"], DQ_BITS["CBC_CAT2"]):
        good &= (dqmask_1hz & (1 << bit)) != 0
    return good


def has_no_injection(injmask_1hz: np.ndarray) -> np.ndarray:
    # LOSC stores "no hardware injection" as positive bits. In the O3a public
    # 4 kHz release, the NO_CW_HW_INJ bit is commonly unset even for otherwise
    # clean open data, while the other NO_* bits remain populated. We therefore
    # require the non-CW NO_* flags and ignore the CW bit here.
    required_no_inj_mask = 0b10111
    return (injmask_1hz & required_no_inj_mask) == required_no_inj_mask


def extract_clean_segments(
    strain: np.ndarray,
    dqmask: np.ndarray,
    injmask: np.ndarray,
    segment_seconds: float = DEFAULT_SEG_SECONDS,
    step_seconds: float | None = None,
    start_offset_seconds: float = 0.0,
) -> np.ndarray:
    if step_seconds is None:
        step_seconds = segment_seconds
    seg_seconds_int = int(round(segment_seconds))
    seg_samples = int(round(segment_seconds * FS))
    good_dq = has_required_dq(dqmask)
    no_inj = has_no_injection(injmask)
    keep = []
    step_samples = max(1, int(round(step_seconds * FS)))
    start_offset_samples = max(0, int(round(start_offset_seconds * FS)))
    max_start = len(strain) - seg_samples
    for start in range(start_offset_samples, max_start + 1, step_samples):
        sec = start // FS
        end_sec = int(np.ceil((start + seg_samples) / FS))
        if not np.all(good_dq[sec:sec + seg_seconds_int]):
            continue
        if not np.all(good_dq[sec:end_sec]):
            continue
        if not np.all(no_inj[sec:end_sec]):
            continue
        stop = start + seg_samples
        segment = strain[start:stop]
        if len(segment) != seg_samples:
            continue
        if not np.all(np.isfinite(segment)):
            continue
        keep.append(segment.astype(np.float32))
    if not keep:
        return np.empty((0, seg_samples), dtype=np.float32)
    return np.stack(keep, axis=0)


def extract_event_windows(
    strain: np.ndarray,
    gps_start: int,
    event_gps: float,
    half_window_seconds: int,
    step_seconds: float,
    segment_seconds: float = DEFAULT_SEG_SECONDS,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    seg_samples = int(round(segment_seconds * FS))
    offsets = np.arange(-half_window_seconds, half_window_seconds + 1e-9, step_seconds, dtype=np.float64)
    windows = []
    meta = []
    for offset in offsets:
        center = event_gps + float(offset)
        start_gps = center - segment_seconds / 2.0
        rel_start = int(round((start_gps - gps_start) * FS))
        rel_stop = rel_start + seg_samples
        if rel_start < 0 or rel_stop > len(strain):
            continue
        seg = strain[rel_start:rel_stop]
        if len(seg) != seg_samples or not np.all(np.isfinite(seg)):
            continue
        windows.append(seg.astype(np.float32))
        meta.append((center, float(offset)))
    if not windows:
        return np.empty((0, seg_samples), dtype=np.float32), []
    return np.stack(windows, axis=0), meta


def estimate_reference_psd(noise_segments: np.ndarray, fs: int, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    psds = []
    for seg in noise_segments:
        freq, psd = welch(np.asarray(seg, dtype=np.float64), fs=fs, nperseg=nperseg)
        psds.append(psd)
    return freq.astype(np.float64), np.mean(psds, axis=0).astype(np.float64)


def write_event_metadata(rows: list[dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "event_name",
                "detector",
                "event_gps",
                "window_center_gps",
                "offset_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_noise_metadata(rows: list[dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "detector",
                "file_gps_start",
                "segment_gps_start",
                "segment_index_in_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_event_specs(event_specs: list[str] | None) -> list[dict[str, float]]:
    events = []
    for spec in event_specs or []:
        try:
            name, gps_str = spec.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --event value '{spec}'. Expected NAME:GPS.") from exc
        events.append({"name": name, "gps": float(gps_str)})
    return events


def compatibility_event_rows(
    noise_test: np.ndarray,
    noise_test_rows: list[dict[str, object]],
    max_events: int,
    segment_seconds: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if len(noise_test) == 0:
        raise RuntimeError("Cannot synthesize compatibility event windows from an empty noise test split.")
    count = min(len(noise_test), max(1, int(max_events)))
    event_windows = noise_test[:count].astype(np.float32, copy=True)
    event_rows = []
    for idx, row in enumerate(noise_test_rows[:count]):
        center_gps = float(row["segment_gps_start"]) + segment_seconds / 2.0
        event_rows.append(
            {
                "event_name": f"NOISE_COMPAT_{idx:04d}",
                "detector": row["detector"],
                "event_gps": center_gps,
                "window_center_gps": center_gps,
                "offset_seconds": 0.0,
            }
        )
    return event_windows, event_rows


def prepare_dataset(args: argparse.Namespace) -> None:
    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(args.output_dir, exist_ok=True)
    seg_seconds = float(args.segment_seconds)
    seg_samples = int(round(seg_seconds * FS))
    start_offset_seconds = float(args.train_start_offset_seconds)
    run_name = args.run_name
    events = parse_event_specs(args.event)
    if not events and run_name == "O1":
        events = list(DEFAULT_O1_EVENTS)

    if args.use_existing_raw:
        if not os.path.isdir(raw_dir):
            raise RuntimeError(f"--use-existing-raw requested but raw dir does not exist: {raw_dir}")
        existing_specs = discover_existing_raw_specs(raw_dir, run_name=run_name)
        event_starts = {event_file_start(event["gps"]) for event in events}
        noise_specs = [spec for spec in existing_specs if spec.gps_start not in event_starts]
        download_urls_by_detector: dict[str, dict[int, str]] = {}
    else:
        download_urls_by_detector = {
            det: fetch_available_run_files(run_name, det)
            for det in DETECTORS
        }
        noise_specs = required_noise_files(run_name, events, args.num_train_blocks)
    event_specs = required_event_files(run_name, events)

    if args.use_existing_raw:
        print(f"Using {len(noise_specs)} existing training files and {len(event_specs)} required event files...")
        missing_event = [spec.filename for spec in event_specs if not os.path.exists(os.path.join(raw_dir, spec.filename))]
        if missing_event:
            raise RuntimeError(
                "Missing required event files in existing raw dir:\n" + "\n".join(missing_event)
            )
    else:
        print(f"Downloading {len(noise_specs)} training files and {len(event_specs)} event files for run {run_name}...")
        for spec in noise_specs + event_specs:
            det_urls = download_urls_by_detector.get(spec.detector, {})
            download_url = det_urls.get(spec.gps_start)
            if download_url is None:
                raise RuntimeError(
                    f"No GWOSC HDF5 file URL found for run {run_name}, detector {spec.detector}, "
                    f"gps_start {spec.gps_start}"
                )
            ensure_download(spec, raw_dir, download_url=download_url)

    noise_segments = []
    noise_rows = []
    for spec in noise_specs:
        path = os.path.join(raw_dir, spec.filename)
        strain, dqmask, injmask = load_frame(path)
        segments = extract_clean_segments(
            strain,
            dqmask,
            injmask,
            segment_seconds=seg_seconds,
            step_seconds=args.train_stride_seconds,
            start_offset_seconds=start_offset_seconds,
        )
        if len(segments) == 0:
            print(f"[warn] no usable clean segments in {spec.filename}")
            continue
        print(f"[noise] {spec.filename}: kept {len(segments)} clean segments")
        noise_segments.append(segments)
        keep_idx = 0
        good_dq = has_required_dq(dqmask)
        no_inj = has_no_injection(injmask)
        step_samples = max(1, int(round(args.train_stride_seconds * FS)))
        start_offset_samples = max(0, int(round(start_offset_seconds * FS)))
        for start in range(start_offset_samples, len(strain) - seg_samples + 1, step_samples):
            sec = start // FS
            end_sec = int(np.ceil((start + seg_samples) / FS))
            if not np.all(good_dq[sec:end_sec]):
                continue
            if not np.all(no_inj[sec:end_sec]):
                continue
            stop = start + seg_samples
            segment = strain[start:stop]
            if len(segment) != seg_samples:
                continue
            if not np.all(np.isfinite(segment)):
                continue
            noise_rows.append(
                {
                    "detector": spec.detector,
                    "file_gps_start": spec.gps_start,
                    "segment_gps_start": spec.gps_start + start / FS,
                    "segment_index_in_file": keep_idx,
                }
            )
            keep_idx += 1

    if not noise_segments:
        raise RuntimeError(f"No clean {run_name} noise segments were extracted.")

    noise_all = np.concatenate(noise_segments, axis=0)
    if len(noise_rows) != len(noise_all):
        raise RuntimeError("Noise metadata length does not match extracted segments.")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(noise_all))
    noise_all = noise_all[perm]
    noise_rows = [noise_rows[i] for i in perm]

    n_test = max(1, int(len(noise_all) * args.test_fraction))
    n_val = max(1, int(len(noise_all) * args.val_fraction))
    if n_test + n_val >= len(noise_all):
        raise RuntimeError("Validation/test split consumed all noise segments; reduce split fractions.")

    noise_test = noise_all[:n_test]
    noise_val = noise_all[n_test:n_test + n_val]
    noise_train = noise_all[n_test + n_val:]
    noise_test_rows = noise_rows[:n_test]
    noise_val_rows = noise_rows[n_test:n_test + n_val]
    noise_train_rows = noise_rows[n_test + n_val:]

    freq, psd = estimate_reference_psd(noise_train[: min(len(noise_train), args.psd_segments)], fs=FS)
    detector_psds = {}
    for detector in DETECTORS:
        det_indices = [idx for idx, row in enumerate(noise_train_rows) if row["detector"] == detector]
        if not det_indices:
            continue
        det_segments = noise_train[np.array(det_indices[: min(len(det_indices), args.psd_segments)], dtype=np.int64)]
        det_freq, det_psd = estimate_reference_psd(det_segments, fs=FS)
        detector_psds[detector] = (det_freq, det_psd)

    np.save(os.path.join(args.output_dir, "noise_train.npy"), noise_train.astype(np.float32))
    np.save(os.path.join(args.output_dir, "noise_val.npy"), noise_val.astype(np.float32))
    np.save(os.path.join(args.output_dir, "noise_test.npy"), noise_test.astype(np.float32))
    write_noise_metadata(noise_train_rows, os.path.join(args.output_dir, "noise_train_metadata.csv"))
    write_noise_metadata(noise_val_rows, os.path.join(args.output_dir, "noise_val_metadata.csv"))
    write_noise_metadata(noise_test_rows, os.path.join(args.output_dir, "noise_test_metadata.csv"))
    np.savez(os.path.join(args.output_dir, "reference_psd.npz"), freq=freq, psd=psd)
    np.save(os.path.join(args.output_dir, "psd_freqs.npy"), freq.astype(np.float64))
    np.save(os.path.join(args.output_dir, "sample_rate.npy"), np.array(FS, dtype=np.int64))
    for detector, (det_freq, det_psd) in detector_psds.items():
        np.savez(os.path.join(args.output_dir, f"reference_psd_{detector}.npz"), freq=det_freq, psd=det_psd)
        np.save(os.path.join(args.output_dir, f"psd_{detector}.npy"), det_psd.astype(np.float64))
        print(
            f"[psd] {detector}: saved {len(det_psd)} bins from "
            f"{min(len([row for row in noise_train_rows if row['detector'] == detector]), args.psd_segments)} "
            f"training segments",
            flush=True,
        )

    event_arrays = []
    event_rows = []
    if events:
        for event in events:
            frame_start = event_file_start(event["gps"])
            for detector in DETECTORS:
                spec = FileSpec(detector, frame_start, run_name=run_name)
                path = os.path.join(raw_dir, spec.filename)
                strain, _, _ = load_frame(path)
                windows, meta = extract_event_windows(
                    strain,
                    gps_start=frame_start,
                    event_gps=event["gps"],
                    half_window_seconds=args.event_half_window,
                    step_seconds=args.event_step,
                    segment_seconds=seg_seconds,
                )
                if len(windows) == 0:
                    print(f"[warn] no event windows extracted for {event['name']} {detector}")
                    continue
                event_arrays.append(windows)
                for center_gps, offset_seconds in meta:
                    event_rows.append(
                        {
                            "event_name": event["name"],
                            "detector": detector,
                            "event_gps": event["gps"],
                            "window_center_gps": center_gps,
                            "offset_seconds": offset_seconds,
                        }
                    )
                print(f"[event] {event['name']} {detector}: extracted {len(windows)} windows")
        if not event_arrays:
            raise RuntimeError("No event windows were extracted from the supplied events.")
        event_windows = np.concatenate(event_arrays, axis=0).astype(np.float32)
    else:
        if not args.allow_noise_only_compat:
            raise RuntimeError(
                f"No events supplied for run {run_name}. Pass one or more --event NAME:GPS values "
                "or use --allow-noise-only-compat for pure-noise cross-run evaluation."
            )
        print("[event] using noise-test compatibility windows because no events were supplied", flush=True)
        event_windows, event_rows = compatibility_event_rows(
            noise_test,
            noise_test_rows,
            max_events=args.compat_event_count,
            segment_seconds=seg_seconds,
        )

    np.save(os.path.join(args.output_dir, "event_windows.npy"), event_windows.astype(np.float32))
    write_event_metadata(event_rows, os.path.join(args.output_dir, "event_metadata.csv"))

    summary = {
        "run_name": run_name,
        "noise_file_blocks": int(len({row["file_gps_start"] for row in noise_rows})),
        "noise_train_segments": int(len(noise_train)),
        "noise_val_segments": int(len(noise_val)),
        "noise_test_segments": int(len(noise_test)),
        "event_windows": int(len(event_windows)),
        "sample_rate_hz": FS,
        "segment_seconds": seg_seconds,
        "train_stride_seconds": float(args.train_stride_seconds),
        "train_start_offset_seconds": start_offset_seconds,
        "central_crop_seconds": 2.0,
    }
    np.savez(os.path.join(args.output_dir, "summary.npz"), **summary)
    print("Prepared GWOSC dataset:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare direct GWOSC archive data.")
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "data", "o1_prepared_4s_crop2s"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="Event specification as NAME:GPS. May be repeated. Defaults to standard O1 events for --run-name O1.",
    )
    parser.add_argument(
        "--allow-noise-only-compat",
        action="store_true",
        help="If no events are supplied, synthesize compatibility event windows from noise-test segments.",
    )
    parser.add_argument(
        "--compat-event-count",
        type=int,
        default=32,
        help="Number of compatibility event windows to synthesize when --allow-noise-only-compat is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-blocks", type=int, default=DEFAULT_NUM_TRAIN_BLOCKS)
    parser.add_argument("--use-existing-raw", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--train-stride-seconds", type=float, default=2.0)
    parser.add_argument("--train-start-offset-seconds", type=float, default=0.0)
    parser.add_argument("--segment-seconds", type=float, default=DEFAULT_SEG_SECONDS)
    parser.add_argument("--psd-segments", type=int, default=2048)
    parser.add_argument("--event-half-window", type=int, default=16)
    parser.add_argument("--event-step", type=float, default=0.25)
    args = parser.parse_args()

    prepare_dataset(args)


if __name__ == "__main__":
    main()
