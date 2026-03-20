from manim import *
import numpy as np


class RegressionProjection(ThreeDScene):
    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        # Title fixed in frame
        title = Text("OLS = Projection", font_size=28, color=WHITE)
        title.to_corner(UL, buff=0.6)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title), run_time=0.5)

        # Minimal 3D axes — no number labels
        axes = ThreeDAxes(
            x_range=[-0.5, 4, 1],
            y_range=[-0.5, 4, 1],
            z_range=[-0.5, 4, 1],
            x_length=5, y_length=5, z_length=5,
            axis_config={"include_numbers": False, "stroke_width": 1.5, "color": GREY_C},
        )
        self.add(axes)

        # Vectors
        y_vec = np.array([1.5, 2.5, 3.0])
        y_hat = np.array([1.5, 2.5, 0.0])  # projection onto z=0 plane

        # Column space plane (z=0, translucent blue)
        plane = Surface(
            lambda u, v: axes.c2p(u, v, 0),
            u_range=[-0.3, 3.8],
            v_range=[-0.3, 3.8],
            resolution=(6, 6),
            fill_opacity=0.18,
            stroke_width=0.3,
            stroke_color=BLUE_D,
        )
        plane.set_color(BLUE)
        plane_label = Text("Col(X)", font_size=16, color=BLUE_B)
        plane_label.move_to(axes.c2p(3.5, 3.5, 0))
        self.add_fixed_orientation_mobjects(plane_label)

        self.play(Create(plane), FadeIn(plane_label), run_time=1.5)

        # y arrow (YELLOW)
        y_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*y_vec),
            color=YELLOW,
            resolution=8,
        )
        y_label = MathTex(r"\mathbf{y}", font_size=20, color=YELLOW)
        y_label.move_to(axes.c2p(*(y_vec + np.array([0.25, 0.25, 0.3]))))
        self.add_fixed_orientation_mobjects(y_label)

        self.play(Create(y_arrow), FadeIn(y_label), run_time=1.5)

        # y_hat arrow (GREEN, in the plane)
        yhat_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*y_hat),
            color=GREEN,
            resolution=8,
        )
        yhat_label = MathTex(r"\hat{\mathbf{y}}", font_size=20, color=GREEN)
        yhat_label.move_to(axes.c2p(*(y_hat + np.array([0.3, -0.4, 0]))))
        self.add_fixed_orientation_mobjects(yhat_label)

        # Dashed drop line from y to y_hat
        drop_line = DashedLine(
            axes.c2p(*y_vec), axes.c2p(*y_hat),
            color=GREY_B, dash_length=0.08, stroke_width=1.5,
        )

        self.play(
            Create(yhat_arrow), FadeIn(yhat_label), Create(drop_line),
            run_time=1.5,
        )

        # Residual arrow (RED, perpendicular)
        res_arrow = Arrow3D(
            start=axes.c2p(*y_hat),
            end=axes.c2p(*y_vec),
            color=RED,
            resolution=8,
        )
        res_label = MathTex(r"\mathbf{y} - \hat{\mathbf{y}}", font_size=16, color=RED)
        mid_res = (y_hat + y_vec) / 2 + np.array([0.6, 0.0, 0.0])
        res_label.move_to(axes.c2p(*mid_res))
        self.add_fixed_orientation_mobjects(res_label)

        # Right-angle marker at projection point
        # Small L-shape in the plane at y_hat
        marker_size = 0.18
        p0 = axes.c2p(*y_hat)
        # Directions: along z (residual) and along -x (arbitrary in-plane)
        right_angle = VGroup(
            Line(
                axes.c2p(y_hat[0], y_hat[1], marker_size),
                axes.c2p(y_hat[0] - marker_size, y_hat[1], marker_size),
                color=WHITE, stroke_width=1.5,
            ),
            Line(
                axes.c2p(y_hat[0] - marker_size, y_hat[1], marker_size),
                axes.c2p(y_hat[0] - marker_size, y_hat[1], 0),
                color=WHITE, stroke_width=1.5,
            ),
        )

        self.play(
            Create(res_arrow), FadeIn(res_label), Create(right_angle),
            run_time=2,
        )

        self.wait(2)


class RegularizationPath(Scene):
    def construct(self):
        # Title
        title = Text("Ridge vs Lasso Regularization", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)

        # Weight space axes
        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=6.5,
            y_length=6.5,
            tips=True,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        w1_label = MathTex(r"w_1", font_size=20).next_to(axes.x_axis, RIGHT, buff=0.15)
        w2_label = MathTex(r"w_2", font_size=20).next_to(axes.y_axis, UP, buff=0.15)
        self.play(Create(axes), FadeIn(w1_label), FadeIn(w2_label), run_time=1)

        # OLS solution center
        ols_center = np.array([2.0, 1.5])
        ols_screen = axes.c2p(*ols_center)

        # Loss contour ellipses (tilted, centered at OLS solution)
        ellipses = VGroup()
        for r in [0.6, 1.1, 1.7, 2.3, 2.9]:
            e = Ellipse(width=r * 1.5, height=r * 0.85, color=YELLOW, stroke_opacity=0.45, stroke_width=1.5)
            e.rotate(25 * DEGREES)
            e.move_to(ols_screen)
            ellipses.add(e)

        ols_dot = Dot(ols_screen, color=YELLOW, radius=0.06)
        ols_label = MathTex(r"\hat{w}_{OLS}", font_size=16, color=YELLOW)
        ols_label.next_to(ols_dot, UR, buff=0.15)

        self.play(
            *[Create(e) for e in ellipses],
            FadeIn(ols_dot), FadeIn(ols_label),
            run_time=1.5,
        )

        # --- L2 Constraint (Ridge) ---
        # Size the circle so it's tangent to one of the middle ellipses
        # The tangent point should be clearly off-axis (both w1, w2 nonzero)
        l2_radius_data = 1.8  # in data units
        l2_radius_screen = np.linalg.norm(
            np.array(axes.c2p(l2_radius_data, 0)) - np.array(axes.c2p(0, 0))
        )
        l2_circle = Circle(
            radius=l2_radius_screen, color=BLUE, stroke_width=2.5,
        ).move_to(axes.c2p(0, 0))
        l2_fill = l2_circle.copy().set_fill(BLUE, opacity=0.1).set_stroke(width=0)
        l2_label = MathTex(r"\|w\|_2^2 \leq t", font_size=16, color=BLUE)
        l2_label.next_to(l2_circle, DL, buff=0.2)

        self.play(Create(l2_circle), FadeIn(l2_fill), FadeIn(l2_label), run_time=1.2)

        # Ridge tangent point: on circle in direction of OLS solution
        ridge_dir = ols_center / np.linalg.norm(ols_center)
        ridge_pt_data = ridge_dir * l2_radius_data
        ridge_dot = Dot(axes.c2p(*ridge_pt_data), color=WHITE, radius=0.07)
        ridge_label = Text("Ridge: both w₁, w₂ ≠ 0", font_size=16, color=BLUE)
        ridge_label.next_to(ridge_dot, UP, buff=0.25)

        # Line from OLS to ridge point showing shrinkage
        shrink_line = DashedLine(
            ols_screen, axes.c2p(*ridge_pt_data),
            color=BLUE, dash_length=0.06, stroke_width=1.5,
        )

        self.play(
            FadeIn(ridge_dot), FadeIn(ridge_label), Create(shrink_line),
            run_time=1,
        )
        self.wait(1.5)

        # Fade out L2 elements
        self.play(
            FadeOut(l2_circle), FadeOut(l2_fill), FadeOut(l2_label),
            FadeOut(ridge_dot), FadeOut(ridge_label), FadeOut(shrink_line),
            run_time=0.8,
        )

        # --- L1 Constraint (Lasso) ---
        l1_size_data = 1.8
        diamond = Polygon(
            axes.c2p(l1_size_data, 0),
            axes.c2p(0, l1_size_data),
            axes.c2p(-l1_size_data, 0),
            axes.c2p(0, -l1_size_data),
            color=RED, stroke_width=2.5,
        )
        l1_fill = diamond.copy().set_fill(RED, opacity=0.1).set_stroke(width=0)
        l1_label = MathTex(r"\|w\|_1 \leq t", font_size=16, color=RED)
        l1_label.next_to(diamond, DL, buff=0.2)

        self.play(Create(diamond), FadeIn(l1_fill), FadeIn(l1_label), run_time=1.2)

        # Lasso tangent point: on axis (sparse solution) — w2=0
        lasso_dot = Dot(axes.c2p(l1_size_data, 0), color=WHITE, radius=0.07)
        lasso_label = Text("Lasso: w₂ = 0 (sparse!)", font_size=16, color=RED)
        lasso_label.next_to(lasso_dot, DR, buff=0.25)

        shrink_line2 = DashedLine(
            ols_screen, axes.c2p(l1_size_data, 0),
            color=RED, dash_length=0.06, stroke_width=1.5,
        )

        self.play(
            FadeIn(lasso_dot), FadeIn(lasso_label), Create(shrink_line2),
            run_time=1,
        )
        self.wait(2)
