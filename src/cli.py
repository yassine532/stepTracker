import sys
import argparse
import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.utils import resolve_video_path

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent

def run_benchmark(video_path: Path, max_duration: float = 5.0):
    console.print(Panel.fit(
        "[bold cyan]🧗 STAIRMASTER AI STEP COUNTER — BENCHMARK SUITE[/bold cyan]\n"
        f"Evaluating Computer Vision & Signal Pipelines on [yellow]{video_path.name}[/yellow] ({max_duration}s sample)",
        border_style="cyan"
    ))

    results = []

    # Solution 2: YOLOv8-Pose
    try:
        from src.solution2.pipeline import YOLOPoseStepCounter
        counter2 = YOLOPoseStepCounter()
        t0 = time.time()
        res2 = counter2.process_video(
            video_path=video_path,
            output_json_path=BASE_DIR / "output" / "solution2" / "sample2_yolo_steps.json",
            max_duration_sec=max_duration
        )
        rt2 = time.time() - t0
        results.append({
            "solution": "Solution 2: YOLOv8-Pose Kinematics",
            "target": "Human Ankle/Knee Elevation",
            "steps": res2.get("total_steps", 0),
            "cadence": f"{res2.get('cadence_spm', 0):.1f} SPM",
            "runtime": f"{rt2:.3f} s",
            "fps": f"{res2.get('fps_processing', 0)} FPS",
            "cost": "$0.00 (Local)",
            "privacy": "Anonymized"
        })
    except Exception as e:
        console.print(f"[red]Error running Solution 2: {e}[/red]")

    # Solution 3: Machine Step Tracker
    try:
        from src.machine_step_tracker import MachineStepTracker
        tracker = MachineStepTracker()
        t0 = time.time()
        res3 = tracker.count_machine_steps(video_path=video_path, max_duration_sec=max_duration)
        rt3 = time.time() - t0
        fps3 = round(res3.get("analyzed_frames", 150) / max(0.001, rt3), 1)
        results.append({
            "solution": "Solution 3: Machine Tread Tracker (Kymograph)",
            "target": "Revolving Stair Treads",
            "steps": res3.get("total_machine_steps", 0),
            "cadence": f"{res3.get('cadence_spm', 0):.1f} SPM",
            "runtime": f"{rt3:.3f} s",
            "fps": f"{fps3} FPS",
            "cost": "$0.00 (Local)",
            "privacy": "Zero Body Dependency"
        })
    except Exception as e:
        console.print(f"[red]Error running Solution 3: {e}[/red]")

    # Print Comparison Table
    table = Table(title="📊 Stairmaster Computer Vision Telemetry Benchmark", border_style="cyan")
    table.add_column("Solution", style="bold white", width=36)
    table.add_column("Target Metric", style="dim", width=24)
    table.add_column("Steps", style="bold green", justify="center", width=8)
    table.add_column("Cadence", style="bold cyan", justify="center", width=14)
    table.add_column("Runtime", style="yellow", justify="center", width=12)
    table.add_column("Speed", style="magenta", justify="center", width=12)
    table.add_column("Cost", style="green", justify="center", width=14)
    table.add_column("Privacy", style="blue", justify="center", width=16)

    for r in results:
        table.add_row(
            r["solution"],
            r["target"],
            str(r["steps"]),
            r["cadence"],
            r["runtime"],
            r["fps"],
            r["cost"],
            r["privacy"]
        )

    console.print(table)

    if len(results) >= 2:
        steps_athlete = results[0]["steps"]
        steps_machine = results[1]["steps"]
        drift = steps_athlete - steps_machine
        drift_color = "green" if drift == 0 else ("yellow" if abs(drift) <= 1 else "red")
        console.print(Panel(
            f"🧠 [bold cyan]BIOMECHANICAL DUAL-METRIC INSIGHT[/bold cyan]\n"
            f"• Athlete Kinematic Steps: [bold green]{steps_athlete}[/bold green]\n"
            f"• Physical Machine Treads: [bold green]{steps_machine}[/bold green]\n"
            f"• Kinematic Drift (Δ = Steps_athlete - Steps_machine): [bold {drift_color}]{drift:+d} steps[/bold {drift_color}]\n"
            f"  [dim]Δ = 0 indicates the athlete is maintaining perfect equilibrium with machine revolving speed.[/dim]",
            border_style="cyan"
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Stairmaster AI Step Counter — Multi-Modal Computer Vision Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run stairmaster --benchmark
  uv run stairmaster --solution 2 --duration 5.0 --save-video
  uv run stairmaster --solution 3 --video local/samples/sample2.mp4
        """
    )
    parser.add_argument("--solution", choices=["1", "2", "3", "4", "5", "all"], default="2",
                        help="Select algorithm solution to execute (default: 2)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to custom video file (defaults to sample2)")
    parser.add_argument("--sample", choices=["1", "2"], default="2",
                        help="Standard benchmark sample index (default: 2)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Max video duration in seconds (default: 5.0)")
    parser.add_argument("--save-video", action="store_true",
                        help="Render annotated video with real-time HUD telemetry")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run multi-solution performance and latency benchmark")
    parser.add_argument("--pick-roi", action="store_true",
                        help="Drag the tread ROI box on a video frame before running (solution 3)")
    parser.add_argument("--pick-at", type=float, default=0.0,
                        help="Second of the video to show when picking the ROI (default: 0)")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Manual normalized ROI for solution 3 (skips auto-selection)")
    

    args = parser.parse_args()

    video_path = resolve_video_path(sample_num=int(args.sample), custom_path=args.video)

    if args.benchmark:
        run_benchmark(video_path=video_path, max_duration=args.duration)
        return

    if args.solution == "2":
        from src.solution2.pipeline import YOLOPoseStepCounter
        counter = YOLOPoseStepCounter()
        out_vid = BASE_DIR / "output" / "solution2" / f"sample{args.sample}_annotated_hud.mp4" if args.save_video else None
        counter.process_video(
            video_path=video_path,
            output_json_path=BASE_DIR / "output" / "solution2" / f"sample{args.sample}_yolo_steps.json",
            output_video_path=out_vid,
            max_duration_sec=args.duration
        )
    elif args.solution == "3":
        from src.solution3.pipeline import run_solution3
        roi = tuple(args.roi) if args.roi else None
        if args.pick_roi:
            from tools.pick_roi import pick_roi
            roi = pick_roi(video_path, args.pick_at)
        out_json = BASE_DIR / "output" / "solution3" / f"sample{args.sample}_machine_steps.json"
        run_solution3(video_path=video_path, output_json=out_json,
                      max_duration_sec=args.duration, roi=roi)
        if args.save_video:
            from src.solution3.video_annotator import render_solution3_annotated_video
            render_solution3_annotated_video(
                video_path=video_path,
                json_path=out_json,
                output_video_path=BASE_DIR / "output" / "solution3" / f"sample{args.sample}_annotated_hud.mp4",
                max_duration_sec=args.duration
            )
    elif args.solution == "all":
        run_benchmark(video_path=video_path, max_duration=args.duration)
    else:
        console.print(f"[yellow]Solution {args.solution} requested. Running via dedicated module...[/yellow]")
        if args.solution == "1":
            from src.solution1.pipeline import main as sol1_main
            sys.argv = [sys.argv[0], "--sample", args.sample, "--max-duration", str(args.duration)]
            if args.save_video:
                sys.argv.append("--save-video")
            sol1_main()
        elif args.solution == "4":
            from src.solution4.pipeline import main as sol4_main
            sys.argv = [sys.argv[0], "--sample", args.sample, "--max-duration", str(args.duration)]
            if args.save_video:
                sys.argv.append("--save-video")
            sol4_main()
        elif args.solution == "5":
            from src.solution5.pipeline import main as sol5_main
            sys.argv = [sys.argv[0], "--sample", args.sample, "--max-duration", str(args.duration)]
            if args.save_video:
                sys.argv.append("--save-video")
            sol5_main()

if __name__ == "__main__":
    main()
