import json
import time
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn)
from src.machine_step_tracker import MachineStepTracker
from src.utils import resolve_video_path
from tools.pick_roi import pick_roi

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "output" / "solution3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_solution3(video_path: Path, output_json: Path, max_duration_sec: float = None, roi=None):
    tracker = MachineStepTracker()
    start_time = time.time()

    with Progress(
        SpinnerColumn(), TextColumn("[bold green]{task.description}"),
        BarColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Starting", total=None)

        def on_stage(msg):
            prog.update(task, description=msg, completed=0, total=None)

        def on_progress(done, total):
            prog.update(task, completed=done, total=total)

        res = tracker.count_machine_steps(
            video_path=video_path,
            stair_roi_norm=roi,
            max_duration_sec=max_duration_sec,
            on_progress=on_progress,
            on_stage=on_stage,
        )

    runtime = round(time.time() - start_time, 3)
    res["solution"] = "Solution 3: Physical Machine Step Tracker (Kymograph)"
    res["runtime_sec"] = runtime

    table = Table(title=f"⚙️ Solution 3 (Physical Machine Steps): {video_path.name}", border_style="green")
    table.add_column("Step #", style="bold white", width=8)
    table.add_column("Frame", style="yellow", width=14)
    table.add_column("Timestamp", style="bold green", width=14)
    table.add_column("Time Str", style="cyan", width=14)

    for s in res["steps"][:15]:
        table.add_row(f"{s['machine_step_id']:02d}", f"Frame #{s['frame']:03d}",
                      f"{s['timestamp_sec']}s", s["time_str"])
    if len(res["steps"]) > 15:
        table.add_row("...", "...", "...", f"+ {len(res['steps']) - 15} more logged")
    console.print(table)

    console.print(Panel(
        f"🏁 [bold green]SOLUTION 3 SUMMARY: {video_path.name}[/bold green]\n"
        f"• Machine Physical Steps Counted: [bold yellow]{res['total_machine_steps']}[/bold yellow] steps\n"
        f"• Machine Cadence: [bold cyan]{res['cadence_spm']} Steps/Min[/bold cyan]\n"
        f"• Duration: {res['duration_sec']} s\n"
        f"• Periodicity confidence: {res['periodicity_confidence']}\n"
        f"• Processing Time: [bold magenta]{runtime:.3f} s[/bold magenta]",
        border_style="green",
    ))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    console.print(f"[bold green]✔ Saved structured JSON to:[/bold green] {output_json}")
    return res


def main():
    parser = argparse.ArgumentParser(description="Solution 3: Physical Machine Step Tracker")
    parser.add_argument("--video", type=str, default=None, help="Custom path to input video file")
    parser.add_argument("--sample", choices=["1", "2"], default="2", help="Standard sample index (default: 2)")
    parser.add_argument("--max-duration", type=float, default=5.0, help="Max duration in seconds (default: 5.0)")
    parser.add_argument("--save-video", action="store_true", help="Generate annotated video with telemetry HUD")
    parser.add_argument("--annot-start", type=float, default=0.0,
                        help="Annotated clip start time in seconds (default: 0)")
    parser.add_argument("--annot-duration", type=float, default=60.0,
                        help="Annotated clip length in seconds (default: 60)")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Manual ROI, normalized (skips auto-selection)")
    parser.add_argument("--pick-roi", action="store_true",
                        help="Open a frame and drag the ROI box with the mouse")
    parser.add_argument("--pick-at", type=float, default=0.0,
                        help="Second of the video to show when picking the ROI (default: 0)")

    args = parser.parse_args()

    video_file = resolve_video_path(sample_num=int(args.sample), custom_path=args.video)
    json_out = OUTPUT_DIR / f"sample{args.sample}_machine_steps.json"
    video_out = OUTPUT_DIR / f"sample{args.sample}_annotated_hud.mp4" if args.save_video else None

    roi = tuple(args.roi) if args.roi else None
    if args.pick_roi:
        roi = pick_roi(video_file, args.pick_at)

    run_solution3(video_file, json_out, args.max_duration, roi=roi)
    if video_out:
        from .video_annotator import render_solution3_annotated_video
        render_solution3_annotated_video(
            video_path=video_file,
            json_path=json_out,
            output_video_path=video_out,
            start_sec=args.annot_start,
            max_duration_sec=args.annot_duration,
        )


if __name__ == "__main__":
    main()