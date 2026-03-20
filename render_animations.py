"""
Render all Manim animations for the ML course.
Run from the ml-course directory: python render_animations.py

Renders each scene to a GIF in animations/rendered/
Uses medium quality (-qm) for good balance of quality and speed.
"""

import subprocess
import sys
import os
from pathlib import Path

# All scenes to render: (source_file, scene_class_name)
SCENES = [
    # Calculus (Module 0B)
    ("calculus.py", "GradientVector2D"),
    ("calculus.py", "ChainRuleComputationGraph"),

    # Linear Algebra (Module 0C)
    ("linear_algebra.py", "LinearTransformation2D"),
    ("linear_algebra.py", "SVDDecomposition"),

    # Probability (Module 0D)
    ("probability.py", "BayesTheoremUpdate"),
    ("probability.py", "CentralLimitTheorem"),

    # Optimization (Module 0F)
    ("optimization.py", "GradientDescentContour"),
    ("optimization.py", "MomentumVsSGD"),

    # Regression (Module 1B)
    ("regression.py", "RegressionProjection"),
    ("regression.py", "RegularizationPath"),

    # Classification (Module 1C)
    ("classification.py", "DecisionBoundaries"),

    # Trees (Module 1E)
    ("trees.py", "DecisionTreeGrowth"),

    # Unsupervised (Module 1F)
    ("unsupervised.py", "PCAVarianceDirections"),

    # Neural Networks (Module 2A)
    ("neural_nets.py", "BackpropFlow"),

    # CNNs (Module 2D)
    ("cnn.py", "ConvolutionSliding"),

    # RNNs (Module 2E)
    ("rnn.py", "RNNUnrolling"),

    # Attention (Module 3A)
    ("attention.py", "AttentionWeights"),

    # Generative (Module 3B)
    ("generative.py", "DiffusionProcess"),

    # RL (Module 3D)
    ("rl.py", "TDLearningUpdate"),
]

def render_scene(src_file: str, scene_name: str, quality: str = "m") -> bool:
    """Render a single manim scene to GIF."""
    src_path = Path("animations/src") / src_file
    out_dir = Path("animations/rendered")

    if not src_path.exists():
        print(f"  SKIP: {src_path} not found")
        return False

    # Check if already rendered
    gif_path = out_dir / f"{scene_name}.gif"
    if gif_path.exists():
        print(f"  EXISTS: {gif_path}")
        return True

    cmd = [
        sys.executable, "-m", "manim", "render",
        f"-q{quality}",
        "--format", "gif",
        "--media_dir", str(Path("animations/rendered/_media")),
        str(src_path),
        scene_name,
    ]

    print(f"  Rendering {scene_name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Find the output GIF and move it to rendered/
            # Manim outputs to media_dir/videos/...
            import glob
            gifs = glob.glob(f"animations/rendered/_media/videos/**/{scene_name}.gif", recursive=True)
            if gifs:
                os.rename(gifs[0], str(gif_path))
                print(f"  OK: {gif_path}")
                return True
            else:
                print(f"  WARNING: Render succeeded but GIF not found for {scene_name}")
                print(f"  stdout: {result.stdout[-200:]}")
                return False
        else:
            print(f"  FAILED: {scene_name}")
            print(f"  stderr: {result.stderr[-300:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {scene_name}")
        return False
    except Exception as e:
        print(f"  ERROR: {scene_name}: {e}")
        return False


def main():
    os.chdir(Path(__file__).parent)

    out_dir = Path("animations/rendered")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(SCENES)} animations...\n")

    succeeded = 0
    failed = 0
    skipped = 0

    for src_file, scene_name in SCENES:
        src_path = Path("animations/src") / src_file
        if not src_path.exists():
            print(f"[SKIP] {src_file}::{scene_name} — source file not found")
            skipped += 1
            continue

        print(f"[{succeeded + failed + 1}/{len(SCENES)}] {src_file}::{scene_name}")
        if render_scene(src_file, scene_name):
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    print(f"GIFs saved to: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
