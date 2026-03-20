from manim import *
import numpy as np


class GradientDescentContour(Scene):
    """Gradient descent on f(x,y) = x^2 + 4y^2 with three learning rates."""

    def construct(self):
        title = Text("Gradient Descent: Learning Rate Effects", font_size=28)
        title.to_edge(UP, buff=0.3)
        func_label = MathTex(r"f(x,y) = x^2 + 4y^2", font_size=20)
        func_label.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), run_time=0.7)
        self.play(Write(func_label), run_time=0.4)

        # Axes (no tick numbers)
        axes = Axes(
            x_range=[-4.5, 4.5, 1],
            y_range=[-3, 3, 1],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        )
        axes.shift(DOWN * 0.5)
        self.play(Create(axes), run_time=0.6)

        # Contour ellipses for f(x,y) = x^2 + 4y^2 = c
        levels = [1, 4, 8, 14, 22, 32]
        contour_colors = [BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E, DARK_BLUE]
        contours = VGroup()
        for c_val, color in zip(levels, contour_colors):
            a = np.sqrt(c_val)
            b = np.sqrt(c_val / 4)
            if a > 4.2 or b > 2.8:
                continue
            ellipse = axes.plot_parametric_curve(
                lambda t, a=a, b=b: np.array([a * np.cos(t), b * np.sin(t), 0]),
                t_range=[0, 2 * np.pi, 0.02],
                color=color,
                stroke_width=1.5,
            )
            contours.add(ellipse)
        self.play(Create(contours, lag_ratio=0.05), run_time=0.8)

        # Legend box in upper-right corner
        legend_items = VGroup()
        configs = [
            (RED, r"\alpha = 0.55 \text{ (too high)}"),
            (YELLOW, r"\alpha = 0.02 \text{ (too low)}"),
            (GREEN, r"\alpha = 0.12 \text{ (just right)}"),
        ]
        for color, tex in configs:
            dot = Dot(radius=0.06, color=color)
            lab = MathTex(tex, font_size=14, color=color)
            row = VGroup(dot, lab).arrange(RIGHT, buff=0.15)
            legend_items.add(row)
        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_items.to_corner(UR, buff=0.5)
        legend_items.shift(DOWN * 0.8)
        legend_box = SurroundingRectangle(
            legend_items, color=WHITE, buff=0.15,
            stroke_width=1, fill_color=BLACK, fill_opacity=0.75,
        )
        self.play(FadeIn(legend_box), Write(legend_items), run_time=0.8)

        # Gradient: grad f = (2x, 8y)
        start = np.array([3.0, 1.5])

        def compute_gd_path(lr, steps):
            pos = start.copy()
            points = [axes.c2p(*pos)]
            for _ in range(steps):
                grad = np.array([2 * pos[0], 8 * pos[1]])
                pos = pos - lr * grad
                pos[0] = np.clip(pos[0], -4.3, 4.3)
                pos[1] = np.clip(pos[1], -2.8, 2.8)
                points.append(axes.c2p(*pos))
            return points

        all_paths = VGroup()
        all_dots = VGroup()

        # Run 1: lr=0.55 (too high - oscillates)
        pts1 = compute_gd_path(0.55, 15)
        path1 = VMobject(color=RED, stroke_width=2.5, stroke_opacity=0.8)
        path1.set_points_as_corners(pts1)
        dot1 = Dot(pts1[0], color=RED, radius=0.07)
        self.play(FadeIn(dot1), run_time=0.2)
        self.play(MoveAlongPath(dot1, path1), Create(path1), run_time=2)
        all_paths.add(path1)
        all_dots.add(dot1)
        self.wait(0.3)

        # Fade path 1 to low opacity, keep for final comparison
        self.play(path1.animate.set_stroke(opacity=0.3), FadeOut(dot1), run_time=0.3)

        # Run 2: lr=0.02 (too low)
        pts2 = compute_gd_path(0.02, 15)
        path2 = VMobject(color=YELLOW, stroke_width=2.5, stroke_opacity=0.8)
        path2.set_points_as_corners(pts2)
        dot2 = Dot(pts2[0], color=YELLOW, radius=0.07)
        self.play(FadeIn(dot2), run_time=0.2)
        self.play(MoveAlongPath(dot2, path2), Create(path2), run_time=2)
        all_paths.add(path2)
        all_dots.add(dot2)
        self.wait(0.3)

        # Fade path 2
        self.play(path2.animate.set_stroke(opacity=0.3), FadeOut(dot2), run_time=0.3)

        # Run 3: lr=0.12 (just right)
        pts3 = compute_gd_path(0.12, 20)
        path3 = VMobject(color=GREEN, stroke_width=2.5, stroke_opacity=0.8)
        path3.set_points_as_corners(pts3)
        dot3 = Dot(pts3[0], color=GREEN, radius=0.07)
        self.play(FadeIn(dot3), run_time=0.2)
        self.play(MoveAlongPath(dot3, path3), Create(path3), run_time=2)
        all_paths.add(path3)
        all_dots.add(dot3)
        self.wait(0.3)

        # Show all three paths at full opacity for comparison
        self.play(
            path1.animate.set_stroke(opacity=0.8),
            path2.animate.set_stroke(opacity=0.8),
            run_time=0.5,
        )

        # Minimum star
        star = Star(n=5, outer_radius=0.12, color=GREEN, fill_opacity=1)
        star.move_to(axes.c2p(0, 0))
        self.play(FadeIn(star), run_time=0.3)
        self.wait(1.5)


class MomentumVsSGD(Scene):
    """Compare vanilla SGD vs SGD with momentum on an elongated loss surface."""

    def construct(self):
        title = Text("SGD vs. SGD + Momentum", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.7)

        # Two panels side by side - smaller to fit
        ax_config = dict(
            x_range=[-4.5, 4.5, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=3.5,
            axis_config={"include_numbers": False, "stroke_width": 1},
        )
        axes_left = Axes(**ax_config)
        axes_left.shift(LEFT * 3.5 + DOWN * 0.3)
        axes_right = Axes(**ax_config)
        axes_right.shift(RIGHT * 3.5 + DOWN * 0.3)

        # Panel labels above
        label_left = Text("Vanilla SGD", font_size=20, color=RED)
        label_left.next_to(axes_left, UP, buff=0.2)
        label_right = Text("SGD + Momentum", font_size=20, color=GREEN)
        label_right.next_to(axes_right, UP, buff=0.2)

        self.play(
            Create(axes_left), Create(axes_right),
            Write(label_left), Write(label_right),
            run_time=0.8,
        )

        # Draw contours on both: f(x,y) = x^2 + 16y^2
        levels = [1, 4, 10, 20, 35]
        for ax in [axes_left, axes_right]:
            contours = VGroup()
            for c_val in levels:
                a = np.sqrt(c_val)
                b = np.sqrt(c_val / 16)
                if a > 4.2:
                    continue
                ellipse = ax.plot_parametric_curve(
                    lambda t, a=a, b=b: np.array([a * np.cos(t), b * np.sin(t), 0]),
                    t_range=[0, 2 * np.pi, 0.02],
                    color=BLUE_D,
                    stroke_width=1.2,
                )
                contours.add(ellipse)
            self.play(Create(contours, lag_ratio=0.02), run_time=0.4)

        # Gradient of f: (2x, 32y)
        start = np.array([3.5, 1.2])
        lr = 0.03
        n_steps = 60

        # Vanilla SGD with noise
        np.random.seed(123)
        pos_sgd = start.copy()
        sgd_points = [axes_left.c2p(*pos_sgd)]
        for _ in range(n_steps):
            grad = np.array([2 * pos_sgd[0], 32 * pos_sgd[1]])
            noise = np.random.randn(2) * 0.5
            pos_sgd = pos_sgd - lr * (grad + noise)
            pos_sgd[0] = np.clip(pos_sgd[0], -4.2, 4.2)
            pos_sgd[1] = np.clip(pos_sgd[1], -1.8, 1.8)
            sgd_points.append(axes_left.c2p(*pos_sgd))

        # SGD with momentum (same noise seed)
        np.random.seed(123)
        pos_mom = start.copy()
        velocity = np.zeros(2)
        beta = 0.9
        mom_points = [axes_right.c2p(*pos_mom)]
        for _ in range(n_steps):
            grad = np.array([2 * pos_mom[0], 32 * pos_mom[1]])
            noise = np.random.randn(2) * 0.5
            velocity = beta * velocity + lr * (grad + noise)
            pos_mom = pos_mom - velocity
            pos_mom[0] = np.clip(pos_mom[0], -4.2, 4.2)
            pos_mom[1] = np.clip(pos_mom[1], -1.8, 1.8)
            mom_points.append(axes_right.c2p(*pos_mom))

        # Create paths and dots
        dot_sgd = Dot(sgd_points[0], color=RED, radius=0.05)
        dot_mom = Dot(mom_points[0], color=GREEN, radius=0.05)
        self.play(FadeIn(dot_sgd), FadeIn(dot_mom), run_time=0.3)

        path_sgd = VMobject(color=RED, stroke_width=1.8, stroke_opacity=0.7)
        path_sgd.set_points_as_corners(sgd_points)
        path_mom = VMobject(color=GREEN, stroke_width=1.8, stroke_opacity=0.7)
        path_mom.set_points_as_corners(mom_points)

        self.play(
            MoveAlongPath(dot_sgd, path_sgd),
            Create(path_sgd),
            MoveAlongPath(dot_mom, path_mom),
            Create(path_mom),
            run_time=4,
        )
        self.wait(0.3)

        # Annotations below each panel
        zigzag_note = Text("Zigzags", font_size=16, color=RED)
        zigzag_note.next_to(axes_left, DOWN, buff=0.2)
        smooth_note = Text("Smooth convergence", font_size=16, color=GREEN)
        smooth_note.next_to(axes_right, DOWN, buff=0.2)
        self.play(Write(zigzag_note), Write(smooth_note), run_time=0.6)

        # Momentum formula at bottom
        mom_formula = MathTex(
            r"v_t = \beta \, v_{t-1} + \eta \, \nabla f, \quad"
            r"\theta_t = \theta_{t-1} - v_t",
            font_size=16,
        )
        mom_formula.to_edge(DOWN, buff=0.35)
        mom_bg = BackgroundRectangle(mom_formula, fill_opacity=0.85, buff=0.1)
        self.play(FadeIn(mom_bg), Write(mom_formula), run_time=0.8)
        self.wait(1.5)
