from manim import *
import numpy as np


class DecisionBoundaries(Scene):
    def construct(self):
        # Title
        title = Text("Decision Boundaries", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)

        # Axes — no number labels, clean
        axes = Axes(
            x_range=[-3, 4, 1],
            y_range=[-2, 3, 1],
            x_length=10,
            y_length=6,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.2, "color": GREY_C},
        ).shift(DOWN * 0.3)
        self.play(Create(axes), run_time=0.8)

        # Generate two-moon data parametrically
        np.random.seed(42)
        n = 40

        # Moon 1 (Class 0, BLUE): upper arc
        t1 = np.linspace(0, np.pi, n)
        moon1_x = np.cos(t1) + np.random.randn(n) * 0.12
        moon1_y = np.sin(t1) + np.random.randn(n) * 0.12

        # Moon 2 (Class 1, ORANGE): lower arc, shifted
        t2 = np.linspace(0, np.pi, n)
        moon2_x = 1.0 - np.cos(t2) + np.random.randn(n) * 0.12
        moon2_y = -np.sin(t2) + 0.5 + np.random.randn(n) * 0.12

        dots_0 = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.045, fill_opacity=0.8)
            for x, y in zip(moon1_x, moon1_y)
        ])
        dots_1 = VGroup(*[
            Dot(axes.c2p(x, y), color=ORANGE, radius=0.045, fill_opacity=0.8)
            for x, y in zip(moon2_x, moon2_y)
        ])

        self.play(FadeIn(dots_0), FadeIn(dots_1), run_time=1)

        # Legend in top-right
        legend = VGroup(
            VGroup(
                Dot(color=BLUE, radius=0.05),
                Text("Class 0", font_size=16, color=BLUE),
            ).arrange(RIGHT, buff=0.12),
            VGroup(
                Dot(color=ORANGE, radius=0.05),
                Text("Class 1", font_size=16, color=ORANGE),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.6)
        bg_rect = SurroundingRectangle(legend, color=GREY_D, fill_color=BLACK, fill_opacity=0.6, buff=0.12, corner_radius=0.05)
        legend_group = VGroup(bg_rect, legend)
        self.play(FadeIn(legend_group), run_time=0.5)

        # Classifier label placeholder
        clf_label = Text("", font_size=20)
        clf_label.next_to(axes, DOWN, buff=0.3)

        # --- 1) Logistic Regression: straight line ---
        lr_line = axes.plot(
            lambda x: -0.6 * x + 0.5,
            x_range=[-2.5, 3.5],
            color=GREEN,
            stroke_width=3,
        )
        lr_label = Text("Logistic Regression", font_size=20, color=GREEN)
        lr_label.next_to(axes, DOWN, buff=0.3)

        self.play(Create(lr_line), FadeIn(lr_label), run_time=1.2)
        self.wait(1)

        # --- 2) SVM (RBF kernel): curved boundary ---
        svm_curve = axes.plot(
            lambda x: 0.4 * np.sin(1.8 * x) + 0.25,
            x_range=[-2.5, 3.5],
            color=YELLOW,
            stroke_width=3,
        )
        svm_label = Text("SVM (RBF kernel)", font_size=20, color=YELLOW)
        svm_label.next_to(axes, DOWN, buff=0.3)

        self.play(
            FadeOut(lr_line), ReplacementTransform(lr_label, svm_label),
            run_time=0.5,
        )
        self.play(Create(svm_curve), run_time=1.2)
        self.wait(1)

        # --- 3) Decision Tree: axis-aligned steps ---
        dt_points = [
            [-2.5, 0.7], [-0.3, 0.7], [-0.3, -0.2], [0.8, -0.2],
            [0.8, 0.9], [1.5, 0.9], [1.5, -0.5], [3.5, -0.5],
        ]
        dt_path = VMobject(color=RED, stroke_width=3)
        dt_path.set_points_as_corners([axes.c2p(p[0], p[1]) for p in dt_points])

        dt_label = Text("Decision Tree", font_size=20, color=RED)
        dt_label.next_to(axes, DOWN, buff=0.3)

        self.play(
            FadeOut(svm_curve), ReplacementTransform(svm_label, dt_label),
            run_time=0.5,
        )
        self.play(Create(dt_path), run_time=1.2)
        self.wait(1)

        # --- Show all three together ---
        lr_line2 = lr_line.copy().set_stroke(opacity=0.7)
        svm_curve2 = svm_curve.copy().set_stroke(opacity=0.7)

        all_label = Text("All Boundaries", font_size=20, color=WHITE)
        all_label.next_to(axes, DOWN, buff=0.3)

        # Small legend for boundary types
        b_legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=GREEN, stroke_width=3),
                   Text("Logistic", font_size=14, color=GREEN)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=YELLOW, stroke_width=3),
                   Text("SVM", font_size=14, color=YELLOW)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=RED, stroke_width=3),
                   Text("Tree", font_size=14, color=RED)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.5)
        b_legend.next_to(all_label, DOWN, buff=0.2)

        self.play(
            FadeIn(lr_line2), FadeIn(svm_curve2),
            ReplacementTransform(dt_label, all_label),
            FadeIn(b_legend),
            run_time=1,
        )
        self.wait(2)
