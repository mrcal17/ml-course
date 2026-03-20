from manim import *
import numpy as np


class PCAVarianceDirections(Scene):
    def construct(self):
        # Title
        title = Text("PCA: Principal Components", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)

        # Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=9,
            y_length=6,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.2, "color": GREY_C},
        ).shift(DOWN * 0.2)
        x_lab = MathTex(r"x_1", font_size=20).next_to(axes.x_axis, DR, buff=0.1)
        y_lab = MathTex(r"x_2", font_size=20).next_to(axes.y_axis, UL, buff=0.1)
        self.play(Create(axes), FadeIn(x_lab), FadeIn(y_lab), run_time=0.8)

        # Generate correlated 2D data
        np.random.seed(7)
        n = 50
        angle = 30 * np.pi / 180
        raw = np.column_stack([
            np.random.randn(n) * 2.0,   # high variance along PC1
            np.random.randn(n) * 0.5,   # low variance along PC2
        ])
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        data = raw @ rot.T

        mean = data.mean(axis=0)
        cov = np.cov(data.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        pc1_dir = eigvecs[:, 0]
        pc2_dir = eigvecs[:, 1]

        # Explained variance
        var_explained = eigvals[0] / eigvals.sum() * 100
        var_pct = int(round(var_explained))

        # Data dots
        dots = VGroup(*[
            Dot(axes.c2p(d[0], d[1]), color=BLUE, radius=0.04, fill_opacity=0.7)
            for d in data
        ])
        self.play(FadeIn(dots, lag_ratio=0.02), run_time=1.2)

        # Mean point
        mean_dot = Dot(axes.c2p(mean[0], mean[1]), color=YELLOW, radius=0.07)
        mean_label = MathTex(r"\bar{x}", font_size=16, color=YELLOW)
        mean_label.next_to(mean_dot, UR, buff=0.12)
        self.play(FadeIn(mean_dot), FadeIn(mean_label), run_time=0.8)

        # PC1 arrow (RED, long)
        pc1_len = 2.8
        pc1_arrow = Arrow(
            start=axes.c2p(mean[0], mean[1]),
            end=axes.c2p(
                mean[0] + pc1_dir[0] * pc1_len,
                mean[1] + pc1_dir[1] * pc1_len,
            ),
            color=RED, stroke_width=4, buff=0,
        )
        pc1_label = Text("PC1", font_size=16, color=RED)
        pc1_label.next_to(pc1_arrow.get_end(), UR, buff=0.12)

        self.play(GrowArrow(pc1_arrow), FadeIn(pc1_label), run_time=1.2)

        # PC2 arrow (BLUE, short)
        pc2_len = 1.0
        pc2_arrow = Arrow(
            start=axes.c2p(mean[0], mean[1]),
            end=axes.c2p(
                mean[0] + pc2_dir[0] * pc2_len,
                mean[1] + pc2_dir[1] * pc2_len,
            ),
            color=BLUE_B, stroke_width=4, buff=0,
        )
        pc2_label = Text("PC2", font_size=16, color=BLUE_B)
        pc2_label.next_to(pc2_arrow.get_end(), UL, buff=0.12)

        self.play(GrowArrow(pc2_arrow), FadeIn(pc2_label), run_time=1.2)
        self.wait(0.8)

        # Project all points onto PC1
        centered = data - mean
        proj_coords = centered @ pc1_dir
        proj_points = mean + np.outer(proj_coords, pc1_dir)

        # Dashed lines from original to projected
        proj_lines = VGroup(*[
            DashedLine(
                axes.c2p(data[i][0], data[i][1]),
                axes.c2p(proj_points[i][0], proj_points[i][1]),
                color=WHITE, stroke_opacity=0.25, dash_length=0.04, stroke_width=1,
            )
            for i in range(n)
        ])

        # Animate dots sliding to PC1 axis
        anims = []
        for i in range(n):
            anims.append(
                dots[i].animate.move_to(axes.c2p(proj_points[i][0], proj_points[i][1]))
            )

        variance_label = Text(
            f"PC1 explains {var_pct}% variance", font_size=16, color=RED,
        )
        variance_label.next_to(axes, DOWN, buff=0.4)

        self.play(
            FadeIn(proj_lines, lag_ratio=0.02),
            *anims,
            FadeIn(variance_label),
            run_time=2.5,
        )

        self.wait(2)
