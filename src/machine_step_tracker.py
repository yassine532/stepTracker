import cv2
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from scipy.signal import correlate, find_peaks, savgol_filter

ROI = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), normalized
ProgressCB = Optional[Callable[[int, Optional[int]], None]]   # (frames_done, frames_total_or_None)
StageCB = Optional[Callable[[str], None]]

# Fallbacks if auto-selection finds nothing usable.
FALLBACK_ROI_REAR: ROI = (0.34, 0.7900000000000003, 0.37, 0.8700000000000002)
FALLBACK_ROI_SIDE: ROI = (0.34, 0.7900000000000003, 0.37, 0.8700000000000002)

PROBE_SEC = 45          # short window used for ROI selection + period estimate
UNLIMITED = 10 ** 12    # "no limit" sentinel (we never trust CAP_PROP_FRAME_COUNT)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 else n + 1


def _prep(sig: np.ndarray, fps: float) -> np.ndarray:
    """Detrend (kills lighting/auto-exposure drift), lightly smooth, z-score."""
    sig = np.asarray(sig, dtype=np.float32)
    n = len(sig)
    win = min(_odd(fps * 2), n if n % 2 else n - 1)
    if win >= 5:
        sig = sig - savgol_filter(sig, win, 1)
    if n >= 5:
        sig = savgol_filter(sig, 5, 2)
    return (sig - sig.mean()) / (sig.std() + 1e-6)


def _dominant_period(sig: np.ndarray, fps: float, lo_s=0.25, hi_s=2.0):
    """Autocorrelation peak in a plausible step-period range -> (lag_frames, strength).
    FFT-based: identical values to np.correlate(mode="full"), but O(n log n)."""
    ac = correlate(sig, sig, mode="full", method="fft")[len(sig) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(lo_s * fps), min(int(hi_s * fps), len(ac) - 1)
    if hi <= lo:
        return None, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    return k, float(ac[k])


class MachineStepTracker:
    """
    Counts physical stair treads via a brightness signal from a small strip ROI.
    Memory-safe for very long videos: only a short probe window is held as frames;
    the full video is streamed and reduced to one float per frame.
    """

    def __init__(self, min_peak_distance_frames: int = 9, prominence_z: float = 0.5,
                 strip_w: float = 0.03, strip_h: float = 0.08):
        self.min_peak_distance = min_peak_distance_frames
        self.prominence_z = prominence_z
        self.strip_w = strip_w
        self.strip_h = strip_h
        self.last_kymograph: Optional[np.ndarray] = None   # probe window only

    # ---------- helpers ----------
    @staticmethod
    def _read_stack(cap, limit_frames: int, width: int = 320) -> np.ndarray:
        """Read up to limit_frames as downscaled grayscale into a preallocated array."""
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        h = max(1, int(h0 * width / max(w0, 1)))
        limit_frames = int(min(limit_frames, 20_000))      # hard safety cap
        stack = np.empty((limit_frames, h, width), dtype=np.uint8)
        n = 0
        while n < limit_frames:
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stack[n] = cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA)
            n += 1
        return stack[:n]                                   # (T, H, W)

    @staticmethod
    def _stream_signal(cap, roi: ROI, limit_frames: int, total: Optional[int] = None,
                       on_progress: ProgressCB = None, chunk: int = 100_000,
                       report_every: int = 500, width: int = 320) -> np.ndarray:
        """One float per frame from the ROI. Reads from the cap's current position and
        never stores frames, so memory is ~4 bytes/frame."""
        # Same geometry as the original stack: 320 px wide, INTER_AREA, then ROI crop
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = width
        H = max(1, int(h0 * width / max(w0, 1)))
        x1, x2 = int(W * roi[0]), max(int(W * roi[2]), int(W * roi[0]) + 2)
        y1, y2 = int(H * roi[1]), max(int(H * roi[3]), int(H * roi[1]) + 3)
        mid = (y2 - y1) // 2
        chunks, buf, k, n = [], np.empty(chunk, np.float32), 0, 0
        while n < limit_frames:
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)[y1:y2, x1:x2]
            buf[k] = g.mean(axis=1)[max(0, mid - 1):mid + 2].mean()
            k += 1
            n += 1
            if k == chunk:
                chunks.append(buf)
                buf, k = np.empty(chunk, np.float32), 0
            if on_progress and n % report_every == 0:
                on_progress(n, total)
        chunks.append(buf[:k])
        if on_progress:
            on_progress(n, n)                              # snap to 100%
        return np.concatenate(chunks)

    @staticmethod
    def _roi_signal(stack: np.ndarray, roi: ROI):
        T, H, W = stack.shape
        x1, x2 = int(W * roi[0]), max(int(W * roi[2]), int(W * roi[0]) + 2)
        y1, y2 = int(H * roi[1]), max(int(H * roi[3]), int(H * roi[1]) + 3)
        strip = stack[:, y1:y2, x1:x2].astype(np.float32)
        kymo = strip.mean(axis=2)                          # (T, y)
        mid = kymo.shape[1] // 2
        sig = kymo[:, max(0, mid - 1):mid + 2].mean(axis=1)
        return sig, kymo.T, (y1 + mid) / H

    def auto_select_roi(self, stack: np.ndarray, fps: float,
                        search: ROI = (0.30, 0.45, 0.75, 0.95), stride: float = 0.02):
        """Slide a small strip over the search area; keep the most regular, contrasty one."""
        best, best_score = None, 0.0
        sw, sh = self.strip_w, self.strip_h
        for x in np.arange(search[0], search[2] - sw, stride):
            for y in np.arange(search[1], search[3] - sh, stride):
                roi = (float(x), float(y), float(x + sw), float(y + sh))
                sig, _, _ = self._roi_signal(stack, roi)
                contrast = float(sig.std())
                if contrast < 2.0:
                    continue
                _, strength = _dominant_period(_prep(sig, fps), fps)
                score = strength * min(contrast / 10.0, 1.0)
                if score > best_score:
                    best, best_score = roi, score
        best = (0.34, 0.7900000000000003, 0.37, 0.8700000000000002)        
        return best, best_score

    @staticmethod
    def _empty_result(video_path: Path) -> Dict[str, Any]:
        return {
            "video_name": video_path.name, "total_machine_steps": 0, "duration_sec": 0.0,
            "cadence_spm": 0.0, "mean_step_interval_sec": 0.0, "periodicity_confidence": 0.0,
            "roi_norm": [], "roi_source": "none", "trigger_line_y_norm": 0.0, "steps": [],
        }

    # ---------- main ----------
    def count_machine_steps(self, video_path: Path, stair_roi_norm: Optional[ROI] = None,
                            max_duration_sec: Optional[float] = None,
                            viewpoint: Optional[str] = None,
                            on_progress: ProgressCB = None,
                            on_stage: StageCB = None) -> Dict[str, Any]:
        stage = on_stage or (lambda msg: None)
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = int(max_duration_sec * fps) if max_duration_sec else UNLIMITED
        if reported > 0:
            total = min(limit, reported)
        else:
            total = limit if limit < UNLIMITED else None   # None -> indeterminate bar

        # 1) short probe window -> ROI + step period
        stage("Reading probe window")
        probe = self._read_stack(cap, min(int(PROBE_SEC * fps), limit))
        if len(probe) < 10:
            cap.release()
            return self._empty_result(video_path)

        side = viewpoint == "side" or "sample1" in video_path.name.lower()
        roi_source = "manual"
        if stair_roi_norm is None:
            stage("Auto-selecting ROI")
            stair_roi_norm, score = self.auto_select_roi(probe, fps)
            roi_source = f"auto (score={score:.2f})"
            if stair_roi_norm is None:
                stair_roi_norm = FALLBACK_ROI_SIDE if side else FALLBACK_ROI_REAR
                roi_source = "fallback"

        _, kymo, trigger_y_norm = self._roi_signal(probe, stair_roi_norm)
        self.last_kymograph = kymo
        del probe

        # 2) stream the whole video, one float per frame
        stage("Scanning video")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sig = self._stream_signal(cap, stair_roi_norm, limit, total, on_progress)
        cap.release()
        if len(sig) < 10:
            return self._empty_result(video_path)

        # 3) detect steps on the 1-D signal
        stage("Detecting steps")
        z = _prep(sig, fps)
        lag, strength = _dominant_period(z, fps)
        dist = max(3, int(0.6 * lag)) if lag else self.min_peak_distance
        peaks, _ = find_peaks(z, distance=dist, prominence=self.prominence_z)

        duration = len(sig) / fps
        n = len(peaks)
        if n >= 2:
            median_int = float(np.median(np.diff(peaks)) / fps)
            cadence = 60.0 / median_int
        else:
            median_int = duration / n if n else 0.0
            cadence = (n / max(duration, 0.1)) * 60.0

        steps = [{
            "machine_step_id": i + 1,
            "frame": int(p),
            "timestamp_sec": round(p / fps, 3),
            "time_str": f"{int(p / fps // 60):02d}:{(p / fps) % 60:05.2f}",
        } for i, p in enumerate(peaks)]

        return {
            "video_name": video_path.name,
            "total_machine_steps": n,
            "duration_sec": round(duration, 2),
            "cadence_spm": round(cadence, 2),
            "mean_step_interval_sec": round(median_int, 3),
            "periodicity_confidence": round(strength, 2),  # <0.3 = don't trust the count
            "roi_norm": [round(v, 3) for v in stair_roi_norm],
            "roi_source": roi_source,
            "trigger_line_y_norm": round(trigger_y_norm, 3),
            "steps": steps,
        }