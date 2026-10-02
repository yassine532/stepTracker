import cv2
import json
import subprocess
from pathlib import Path
from typing import Optional

from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn)

from src.machine_step_tracker import (
    MachineStepTracker, _prep, FALLBACK_ROI_REAR, FALLBACK_ROI_SIDE,
)

DEFAULT_CLIP_SEC = 60.0   # never render an unbounded clip


def render_solution3_annotated_video(
    video_path: Path,
    json_path: Path,
    output_video_path: Path,
    start_sec: float = 0.0,
    max_duration_sec: Optional[float] = None,
):
    """Render a HUD-annotated slice [start_sec, start_sec + max_duration_sec]."""
    video_path, json_path = Path(video_path), Path(json_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    cadence_spm = data.get("cadence_spm", 0.0)
    avg_interval = data.get("mean_step_interval_sec", 0.0)
    confidence = data.get("periodicity_confidence", None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(start_sec * fps))
    n_frames = int((max_duration_sec or DEFAULT_CLIP_SEC) * fps)

    # Use the ROI the tracker actually measured (single source of truth).
    is_side = "sample1" in video_path.name.lower()
    roi = data.get("roi_norm") or (FALLBACK_ROI_SIDE if is_side else FALLBACK_ROI_REAR)
    x1, y1 = int(width * roi[0]), int(height * roi[1])
    x2, y2 = int(width * roi[2]), int(height * roi[3])
    trigger_y_norm = data.get("trigger_line_y_norm", (roi[1] + roi[3]) / 2)
    trigger_line_y = int(height * trigger_y_norm)

    temp_raw_path = output_video_path.parent / f"temp_{output_video_path.stem}.mp4"

    with Progress(
        SpinnerColumn(), TextColumn("[bold green]{task.description}"),
        BarColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
    ) as prog:
        task = prog.add_task("Pass 1: extracting signal", total=n_frames)

        # Pass 1: 1-D signal for the slice only (streamed, no frame storage)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        raw_sig = MachineStepTracker._stream_signal(
            cap, tuple(roi), n_frames, n_frames,
            on_progress=lambda d, t: prog.update(task, completed=d),
        )
        if len(raw_sig) == 0:
            cap.release()
            return
        n_frames = len(raw_sig)
        signal = _prep(raw_sig, fps)
        sig_min, sig_max = float(signal.min()), float(signal.max())
        sig_range = max(1e-6, sig_max - sig_min)

        # Step frames inside this slice, converted to slice-local indices
        step_frames = {s["frame"] - start_frame for s in steps
                       if start_frame <= s["frame"] < start_frame + n_frames}
        running_treads = sum(1 for s in steps if s["frame"] < start_frame)  # steps before the slice

        out_writer = cv2.VideoWriter(str(temp_raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        last_trigger_frame, flash_frames, hist_len = -99, 8, 60

        # Pass 2: render
        prog.update(task, description="Pass 2: rendering video", completed=0, total=n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in step_frames:
                running_treads += 1
                last_trigger_frame = i
            is_triggering = 0 <= (i - last_trigger_frame) <= flash_frames

            overlay = frame.copy()
            hud_x1, hud_y1, hud_x2, hud_y2 = 20, 25, 480, 222
            cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (12, 18, 16), -1)
            cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (16, 185, 129), 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 60, 40), -1)

            spark_w, spark_h = 320, 130
            sx1, sy1 = width - spark_w - 20, height - spark_h - 30
            sx2, sy2 = sx1 + spark_w, sy1 + spark_h
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (12, 18, 16), -1)
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (50, 160, 100), 1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

            # HUD text
            cv2.putText(frame, "SOLUTION 3: MACHINE TREAD TRACKER", (hud_x1 + 15, hud_y1 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.60, (16, 220, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, "METHOD: SPATIO-TEMPORAL KYMOGRAPH", (hud_x1 + 15, hud_y1 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 190), 1, cv2.LINE_AA)
            count_color = (0, 255, 255) if is_triggering else (0, 255, 128)
            cv2.putText(frame, f"TREADS: {running_treads}", (hud_x1 + 15, hud_y1 + 105),
                        cv2.FONT_HERSHEY_DUPLEX, 1.35, count_color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"CADENCE: {cadence_spm:.1f} SPM  |  AVG: {avg_interval:.2f}s/STEP",
                        (hud_x1 + 15, hud_y1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            status = "STATUS: >> TREAD PASSAGE TRIGGERED <<" if is_triggering else "STATUS: TRACKING REVOLVING STAIRS"
            cv2.putText(frame, status, (hud_x1 + 15, hud_y1 + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 255, 255) if is_triggering else (120, 230, 160), 1, cv2.LINE_AA)
            conf_txt = (f"SIGNAL CONFIDENCE: {confidence:.2f}" if confidence is not None
                        else "ROI: " + data.get("roi_source", ""))
            conf_color = (130, 160, 150) if (confidence or 1) >= 0.3 else (60, 60, 255)
            cv2.putText(frame, conf_txt, (hud_x1 + 15, hud_y1 + 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1, cv2.LINE_AA)

            # ROI brackets, label, trigger line
            roi_color = (0, 255, 255) if is_triggering else (0, 255, 128)
            th, cl = (3 if is_triggering else 2), 10
            for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (px, py), (px + dx * cl, py), roi_color, th)
                cv2.line(frame, (px, py), (px, py + dy * cl), roi_color, th)
            cv2.putText(frame, "STAIR TREAD ROI", (x1 - 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 1, cv2.LINE_AA)
            ext = 20 if is_triggering else 8
            cv2.line(frame, (x1 - ext, trigger_line_y), (x2 + ext, trigger_line_y),
                     (0, 255, 255) if is_triggering else (0, 220, 255), 3 if is_triggering else 2, cv2.LINE_AA)
            if is_triggering:
                cv2.putText(frame, f"TREAD #{running_treads}", (x2 + ext + 8, trigger_line_y + 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

            # Sparkline of the detection signal with detected peaks marked
            cv2.putText(frame, "OPTICAL KYMOGRAPH SIGNAL", (sx1 + 10, sy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (16, 220, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, "Detrended, z-scored tread brightness", (sx1 + 10, sy1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 170, 155), 1, cv2.LINE_AA)
            start = max(0, i - hist_len)
            seg = signal[start:i + 1]
            if len(seg) > 1:
                plot_w, plot_h, base_y = spark_w - 24, spark_h - 55, sy2 - 12
                dxp = plot_w / float(hist_len)
                pts, marks = [], []
                for k, val in enumerate(seg):
                    px = int(sx1 + 12 + (hist_len - len(seg) + k) * dxp)
                    py = int(base_y - (val - sig_min) / sig_range * plot_h)
                    pts.append((px, py))
                    if (start + k) in step_frames:
                        marks.append((px, py))
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, a, b, (0, 220, 130), 2, cv2.LINE_AA)
                for m in marks:
                    cv2.drawMarker(frame, m, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
                cv2.circle(frame, pts[-1], 5, (0, 255, 255) if is_triggering else (0, 255, 120), -1)

            out_writer.write(frame)
            if i % 25 == 0:
                prog.update(task, completed=i)

        cap.release()
        out_writer.release()

        prog.update(task, description="Encoding (ffmpeg)", completed=0, total=None)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(temp_raw_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(output_video_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            temp_raw_path.unlink(missing_ok=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # ffmpeg missing or failed: keep the raw mp4v file so the render isn't lost
            temp_raw_path.replace(output_video_path)

    print(f"✔ Successfully generated Solution 3 annotated video: {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render Solution 3 annotated HUD video")
    parser.add_argument("--sample", type=int, default=2)
    parser.add_argument("--video", type=str, default=None, help="Custom video path (overrides --sample)")
    parser.add_argument("--start", type=float, default=0.0, help="Clip start in seconds")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_CLIP_SEC,
                        help=f"Clip length in seconds (default: {DEFAULT_CLIP_SEC:.0f})")
    args = parser.parse_args()

    BASE = Path(__file__).resolve().parent.parent.parent
    render_solution3_annotated_video(
        video_path=Path(args.video) if args.video else BASE / "samples" / f"sample{args.sample}.mp4",
        json_path=BASE / "output" / "solution3" / f"sample{args.sample}_machine_steps.json",
        output_video_path=BASE / "output" / "solution3" / f"sample{args.sample}_annotated_hud.mp4",
        start_sec=args.start,
        max_duration_sec=args.max_duration,
    )