from manim import *
import numpy as np


class DiffusionProcess(Scene):
    def construct(self):
        # Title
        title = Text("Diffusion Process", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.5)

        GRID_SIZE = 6
        CELL_SIZE = 0.25
        grid_side = GRID_SIZE * CELL_SIZE  # 1.5 units

        # Define arrow/cross pattern: a plus/cross shape
        # 0=grey background, 1=blue pattern
        pattern = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # Horizontal bar (row 2 and 3, cols 0-5)
        for c in range(GRID_SIZE):
            pattern[2, c] = 1
            pattern[3, c] = 1
        # Vertical bar (col 2 and 3, rows 0-5)
        for r in range(GRID_SIZE):
            pattern[r, 2] = 1
            pattern[r, 3] = 1

        pattern_colors = {0: "#555555", 1: BLUE}
        noise_palette = [RED, ORANGE, YELLOW, GREEN, TEAL, PURPLE, PINK, "#555555", BLUE, WHITE]

        def make_grid(data, center):
            """Build a VGroup of colored squares from 2D int or color array."""
            squares = VGroup()
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    val = data[r, c]
                    if isinstance(val, (int, np.integer)):
                        color = pattern_colors.get(int(val), "#555555")
                    else:
                        color = val
                    sq = Square(
                        side_length=CELL_SIZE,
                        fill_color=color, fill_opacity=0.9,
                        stroke_width=0.5, stroke_color=GREY_D,
                    )
                    sq.move_to(
                        center
                        + RIGHT * (c - GRID_SIZE / 2 + 0.5) * CELL_SIZE
                        + DOWN * (r - GRID_SIZE / 2 + 0.5) * CELL_SIZE
                    )
                    squares.add(sq)
            return squares

        def add_noise_to_pattern(base, fraction, rng):
            """Return color array with fraction of cells randomized."""
            result = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    result[r, c] = pattern_colors.get(int(base[r, c]), "#555555")
            n_total = GRID_SIZE * GRID_SIZE
            n_noisy = int(n_total * fraction)
            indices = rng.choice(n_total, size=n_noisy, replace=False)
            for idx in indices:
                r, c = divmod(idx, GRID_SIZE)
                result[r, c] = rng.choice(noise_palette)
            return result

        def make_pure_noise(rng):
            result = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    result[r, c] = rng.choice(noise_palette)
            return result

        rng = np.random.default_rng(42)

        # Forward stages: clean → slight noise → more noise → pure noise
        forward_data = [
            pattern,
            add_noise_to_pattern(pattern, 0.25, rng),
            add_noise_to_pattern(pattern, 0.6, rng),
            make_pure_noise(rng),
        ]
        # Reverse stages: pure noise → less noisy → clearer → reconstructed
        rng2 = np.random.default_rng(99)
        reverse_data = [
            make_pure_noise(rng2),
            add_noise_to_pattern(pattern, 0.6, rng2),
            add_noise_to_pattern(pattern, 0.2, rng2),
            pattern,
        ]

        n_stages = 4
        x_spacing = 2.8
        x_start = -(n_stages - 1) * x_spacing / 2

        # Forward row
        forward_y = 1.5
        forward_label = Text("Forward Process", font_size=20, color=RED)
        forward_label.move_to([0, forward_y + grid_side / 2 + 0.4, 0])
        self.play(FadeIn(forward_label), run_time=0.4)

        forward_grids = []
        forward_arrows = []
        for i in range(n_stages):
            cx = x_start + i * x_spacing
            center = np.array([cx, forward_y, 0])
            grid = make_grid(forward_data[i], center)
            forward_grids.append(grid)

            self.play(FadeIn(grid), run_time=0.4)
            if i > 0:
                prev_right = np.array([x_start + (i - 1) * x_spacing + grid_side / 2 + 0.1, forward_y, 0])
                curr_left = np.array([cx - grid_side / 2 - 0.1, forward_y, 0])
                arr = Arrow(prev_right, curr_left, color=RED, stroke_width=2, buff=0.05,
                            max_tip_length_to_length_ratio=0.15)
                arr_label = Text("Add noise", font_size=12, color=RED)
                arr_label.next_to(arr, UP, buff=0.08)
                forward_arrows.append(VGroup(arr, arr_label))
                self.play(GrowArrow(arr), FadeIn(arr_label), run_time=0.3)

        self.wait(0.3)

        # Reverse row
        reverse_y = -1.5
        reverse_label = Text("Reverse Process", font_size=20, color=GREEN)
        reverse_label.move_to([0, reverse_y + grid_side / 2 + 0.4, 0])
        self.play(FadeIn(reverse_label), run_time=0.4)

        reverse_grids = []
        reverse_arrows = []
        for i in range(n_stages):
            # Right to left: stage 0 at right, stage 3 at left
            cx = x_start + (n_stages - 1 - i) * x_spacing
            center = np.array([cx, reverse_y, 0])
            grid = make_grid(reverse_data[i], center)
            reverse_grids.append(grid)

            self.play(FadeIn(grid), run_time=0.4)
            if i > 0:
                prev_cx = x_start + (n_stages - i) * x_spacing
                prev_left = np.array([prev_cx - grid_side / 2 - 0.1, reverse_y, 0])
                curr_right = np.array([cx + grid_side / 2 + 0.1, reverse_y, 0])
                arr = Arrow(prev_left, curr_right, color=GREEN, stroke_width=2, buff=0.05,
                            max_tip_length_to_length_ratio=0.15)
                arr_label = Text("Denoise (learned)", font_size=12, color=GREEN)
                arr_label.next_to(arr, DOWN, buff=0.08)
                reverse_arrows.append(VGroup(arr, arr_label))
                self.play(GrowArrow(arr), FadeIn(arr_label), run_time=0.3)

        self.wait(1.5)
