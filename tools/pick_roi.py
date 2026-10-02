import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2

ROI = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), normalized


def pick_roi(video_path: Path, at_sec: float = 0.0, max_view_h: int = 900) -> Optional[ROI]:
    """Show one frame, let the user drag a box, return it as a normalized ROI.
    Returns None if the user cancels (C / ESC)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame at {at_sec}s")

    H, W = frame.shape[:2]
    scale = max_view_h / H if H > max_view_h else 1.0
    view = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame

    print("Drag a box over the stair treads, then press ENTER or SPACE (C to cancel).")
    x, y, w, h = cv2.selectROI("Drag ROI - ENTER to confirm, C to cancel", view, showCrosshair=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if w == 0 or h == 0:
        return None

    roi = (round(x / scale / W, 3), round(y / scale / H, 3),
           round((x + w) / scale / W, 3), round((y + h) / scale / H, 3))
    print(f"ROI selected: {roi}")
    print(f"Reuse with:   --roi {' '.join(map(str, roi))}")
    return roi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drag-select a normalized ROI on a video frame")
    parser.add_argument("video", type=str)
    parser.add_argument("--at", type=float, default=0.0, help="Second of the video to show")
    args = parser.parse_args()
    pick_roi(Path(args.video), args.at)