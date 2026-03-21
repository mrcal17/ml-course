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


class LpNormsAndLossEllipses(Scene):
    """Unit balls for L1, L2, L-infinity norms overlaid with quadratic loss contours."""

    def construct(self):
        title = Text("Lp Unit Balls & Quadratic Loss Contours", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.7)

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=7,
            y_length=7,
            tips=True,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        w1_label = MathTex(r"w_1", font_size=20).next_to(axes.x_axis, RIGHT, buff=0.15)
        w2_label = MathTex(r"w_2", font_size=20).next_to(axes.y_axis, UP, buff=0.15)
        self.play(Create(axes), FadeIn(w1_label), FadeIn(w2_label), run_time=0.8)

        # --- L1 unit ball (diamond) ---
        l1_diamond = Polygon(
            axes.c2p(1, 0), axes.c2p(0, 1),
            axes.c2p(-1, 0), axes.c2p(0, -1),
            color=RED, stroke_width=2.5,
        )
        l1_fill = l1_diamond.copy().set_fill(RED, opacity=0.06).set_stroke(width=0)
        l1_label = MathTex(r"\|w\|_1 = 1", font_size=16, color=RED)
        l1_label.next_to(axes.c2p(0.6, 0.6), UR, buff=0.1)

        # --- L2 unit ball (circle) ---
        l2_radius = np.linalg.norm(
            np.array(axes.c2p(1, 0)) - np.array(axes.c2p(0, 0))
        )
        l2_circle = Circle(
            radius=l2_radius, color=BLUE, stroke_width=2.5,
        ).move_to(axes.c2p(0, 0))
        l2_fill = l2_circle.copy().set_fill(BLUE, opacity=0.06).set_stroke(width=0)
        l2_label = MathTex(r"\|w\|_2 = 1", font_size=16, color=BLUE)
        l2_label.next_to(axes.c2p(0.75, 0.75), UR, buff=0.15)

        # --- L-inf unit ball (square) ---
        linf_square = Polygon(
            axes.c2p(1, 1), axes.c2p(-1, 1),
            axes.c2p(-1, -1), axes.c2p(1, -1),
            color=GREEN, stroke_width=2.5,
        )
        linf_fill = linf_square.copy().set_fill(GREEN, opacity=0.06).set_stroke(width=0)
        linf_label = MathTex(r"\|w\|_\infty = 1", font_size=16, color=GREEN)
        linf_label.next_to(axes.c2p(1, 1), UR, buff=0.1)

        # Animate unit balls sequentially
        self.play(Create(l1_diamond), FadeIn(l1_fill), FadeIn(l1_label), run_time=1)
        self.play(Create(l2_circle), FadeIn(l2_fill), FadeIn(l2_label), run_time=1)
        self.play(Create(linf_square), FadeIn(linf_fill), FadeIn(linf_label), run_time=1)
        self.wait(0.5)

        # Legend
        legend_items = VGroup()
        for color, tex in [
            (RED, r"L^1 \text{ — diamond}"),
            (BLUE, r"L^2 \text{ — circle}"),
            (GREEN, r"L^\infty \text{ — square}"),
        ]:
            dot = Dot(radius=0.06, color=color)
            lab = MathTex(tex, font_size=14, color=color)
            row = VGroup(dot, lab).arrange(RIGHT, buff=0.15)
            legend_items.add(row)
        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_items.to_corner(UL, buff=0.5).shift(DOWN * 0.8)
        legend_box = SurroundingRectangle(
            legend_items, color=WHITE, buff=0.15,
            stroke_width=1, fill_color=BLACK, fill_opacity=0.75,
        )
        self.play(FadeIn(legend_box), Write(legend_items), run_time=0.8)
        self.wait(1)

        # --- Quadratic loss ellipses: L(w) = (w - w*)^T H (w - w*) ---
        # Center at w* = (1.8, 1.2), H has eigenvalues 4 and 1 (ellipse stretched along w2)
        w_star = np.array([1.8, 1.2])
        w_star_screen = axes.c2p(*w_star)

        # Hessian eigenvalues: lambda1=4 (w1 dir), lambda2=1 (w2 dir)
        # Level set: 4*(w1 - 1.8)^2 + (w2 - 1.2)^2 = c
        # semi-axes: a = sqrt(c/4) along w1, b = sqrt(c) along w2
        ellipses = VGroup()
        levels = [0.5, 1.5, 3.0, 5.0, 8.0]
        for c_val in levels:
            a = np.sqrt(c_val / 4)  # narrow (high curvature)
            b = np.sqrt(c_val)      # wide (low curvature)
            ellipse = axes.plot_parametric_curve(
                lambda t, a=a, b=b: np.array([
                    w_star[0] + a * np.cos(t),
                    w_star[1] + b * np.sin(t),
                    0,
                ]),
                t_range=[0, 2 * np.pi, 0.02],
                color=YELLOW,
                stroke_width=1.5,
                stroke_opacity=0.5,
            )
            ellipses.add(ellipse)

        w_star_dot = Dot(w_star_screen, color=YELLOW, radius=0.06)
        w_star_label = MathTex(r"w^*", font_size=18, color=YELLOW)
        w_star_label.next_to(w_star_dot, UR, buff=0.12)

        loss_label = MathTex(
            r"L(w) = 4(w_1 - w^*_1)^2 + (w_2 - w^*_2)^2",
            font_size=16, color=YELLOW,
        )
        loss_label.to_corner(UR, buff=0.5).shift(DOWN * 0.8)
        loss_bg = BackgroundRectangle(loss_label, fill_opacity=0.8, buff=0.1)

        self.play(
            Create(ellipses, lag_ratio=0.08),
            FadeIn(w_star_dot), FadeIn(w_star_label),
            FadeIn(loss_bg), Write(loss_label),
            run_time=2,
        )
        self.wait(0.5)

        # === Tangent point calculations ===
        # Loss: L(w) = 4(w1-1.8)^2 + (w2-1.2)^2
        # We find the point on each unit ball that minimizes L (constrained optimum).

        # --- L1 tangent: minimize L on |w1|+|w2|=1 ---
        # By Lagrange multipliers or direct check: since w* is in the first quadrant
        # and the loss has much higher curvature in w1 (lambda=4), the ellipse
        # hits the w2-axis corner first. The tangent lands on (0, 1) — w1=0 is sparse.
        l1_tangent = np.array([0.0, 1.0])
        # Draw the tangent ellipse through this point
        l1_c = 4 * (l1_tangent[0] - w_star[0])**2 + (l1_tangent[1] - w_star[1])**2
        l1_a = np.sqrt(l1_c / 4)
        l1_b = np.sqrt(l1_c)
        l1_ellipse = axes.plot_parametric_curve(
            lambda t, a=l1_a, b=l1_b: np.array([
                w_star[0] + a * np.cos(t),
                w_star[1] + b * np.sin(t), 0,
            ]),
            t_range=[0, 2 * np.pi, 0.02],
            color=RED, stroke_width=2.5, stroke_opacity=0.8,
        )
        l1_dot = Dot(axes.c2p(*l1_tangent), color=WHITE, radius=0.09)
        l1_ring = Circle(radius=0.18, color=RED, stroke_width=2.5).move_to(axes.c2p(*l1_tangent))
        l1_note = Text("w1 = 0 (sparse!)", font_size=14, color=RED)
        l1_note.next_to(axes.c2p(*l1_tangent), LEFT, buff=0.25)
        l1_note_bg = BackgroundRectangle(l1_note, fill_opacity=0.85, buff=0.08)
        l1_bottom = Text(
            "L1: Ellipse hits diamond corner on axis. One weight is exactly 0 (Lasso sparsity).",
            font_size=15, color=RED,
        )
        l1_bottom.to_edge(DOWN, buff=0.4)
        l1_bottom_bg = BackgroundRectangle(l1_bottom, fill_opacity=0.85, buff=0.1)

        self.play(
            Create(l1_ellipse),
            FadeIn(l1_dot), Create(l1_ring),
            FadeIn(l1_note_bg), Write(l1_note),
            FadeIn(l1_bottom_bg), Write(l1_bottom),
            run_time=1.5,
        )
        self.wait(2.5)

        # --- L2 tangent: minimize L on w1^2+w2^2=1 ---
        # Lagrange: gradient of L = lambda * gradient of g
        # (8(w1-1.8), 2(w2-1.2)) = lambda * (2w1, 2w2)
        # => 4(w1-1.8)/w1 = (w2-1.2)/w2
        # => 4 - 7.2/w1 = 1 - 1.2/w2
        # Numerically: project w* onto unit circle scaled by Hessian curvature
        # Using Lagrange solution numerically:
        from scipy.optimize import minimize as sp_minimize
        def loss_fn(w):
            return 4*(w[0]-1.8)**2 + (w[1]-1.2)**2
        from scipy.optimize import minimize as _min
        res = _min(loss_fn, [0.5, 0.5], method='SLSQP',
                   constraints={'type': 'eq', 'fun': lambda w: w[0]**2 + w[1]**2 - 1})
        l2_tangent = res.x

        l2_c = loss_fn(l2_tangent)
        l2_a = np.sqrt(l2_c / 4)
        l2_b_val = np.sqrt(l2_c)
        l2_ellipse = axes.plot_parametric_curve(
            lambda t, a=l2_a, b=l2_b_val: np.array([
                w_star[0] + a * np.cos(t),
                w_star[1] + b * np.sin(t), 0,
            ]),
            t_range=[0, 2 * np.pi, 0.02],
            color=BLUE, stroke_width=2.5, stroke_opacity=0.8,
        )
        l2_dot = Dot(axes.c2p(*l2_tangent), color=WHITE, radius=0.09)
        l2_ring = Circle(radius=0.18, color=BLUE, stroke_width=2.5).move_to(axes.c2p(*l2_tangent))
        l2_coord = Text(
            f"({l2_tangent[0]:.2f}, {l2_tangent[1]:.2f})",
            font_size=14, color=BLUE,
        )
        l2_coord.next_to(axes.c2p(*l2_tangent), DL, buff=0.2)
        l2_coord_bg = BackgroundRectangle(l2_coord, fill_opacity=0.85, buff=0.08)
        l2_bottom = Text(
            "L2: Ellipse is tangent to circle. Both weights nonzero — Ridge shrinks but keeps all features.",
            font_size=15, color=BLUE,
        )
        l2_bottom.to_edge(DOWN, buff=0.4)
        l2_bottom_bg = BackgroundRectangle(l2_bottom, fill_opacity=0.85, buff=0.1)

        # Fade out L1 annotations, show L2
        self.play(
            FadeOut(l1_ellipse), FadeOut(l1_dot), FadeOut(l1_ring),
            FadeOut(l1_note_bg), FadeOut(l1_note),
            FadeOut(l1_bottom_bg), FadeOut(l1_bottom),
            run_time=0.6,
        )
        self.play(
            Create(l2_ellipse),
            FadeIn(l2_dot), Create(l2_ring),
            FadeIn(l2_coord_bg), Write(l2_coord),
            FadeIn(l2_bottom_bg), Write(l2_bottom),
            run_time=1.5,
        )
        self.wait(2.5)

        # --- L-inf tangent: minimize L on max(|w1|,|w2|)=1 ---
        # The square boundary. Since w*=(1.8,1.2), the closest face is w1=1.
        # On that face, minimize over w2: L = 4(1-1.8)^2 + (w2-1.2)^2
        # Min at w2=1.2 (unconstrained in w2 since |1.2|>1 → clamp to w2=1)
        # Actually max(|w1|,|w2|)=1 means w1 in [-1,1], w2 in [-1,1], at least one is +/-1
        # Closest to w*: w1=1, w2=1 (corner) or w1=1, w2=1.2 but |w2|<=1 so w2=1
        linf_tangent = np.array([1.0, 1.0])

        linf_c = loss_fn(linf_tangent)
        linf_a = np.sqrt(linf_c / 4)
        linf_b = np.sqrt(linf_c)
        linf_ellipse = axes.plot_parametric_curve(
            lambda t, a=linf_a, b=linf_b: np.array([
                w_star[0] + a * np.cos(t),
                w_star[1] + b * np.sin(t), 0,
            ]),
            t_range=[0, 2 * np.pi, 0.02],
            color=GREEN, stroke_width=2.5, stroke_opacity=0.8,
        )
        linf_dot = Dot(axes.c2p(*linf_tangent), color=WHITE, radius=0.09)
        linf_ring = Circle(radius=0.18, color=GREEN, stroke_width=2.5).move_to(axes.c2p(*linf_tangent))
        linf_coord = Text("(1.00, 1.00)", font_size=14, color=GREEN)
        linf_coord.next_to(axes.c2p(*linf_tangent), UR, buff=0.2)
        linf_coord_bg = BackgroundRectangle(linf_coord, fill_opacity=0.85, buff=0.08)
        linf_bottom = Text(
            "L-inf: Ellipse hits square corner. Both weights maxed at boundary — no sparsity.",
            font_size=15, color=GREEN,
        )
        linf_bottom.to_edge(DOWN, buff=0.4)
        linf_bottom_bg = BackgroundRectangle(linf_bottom, fill_opacity=0.85, buff=0.1)

        # Fade out L2 annotations, show L-inf
        self.play(
            FadeOut(l2_ellipse), FadeOut(l2_dot), FadeOut(l2_ring),
            FadeOut(l2_coord_bg), FadeOut(l2_coord),
            FadeOut(l2_bottom_bg), FadeOut(l2_bottom),
            run_time=0.6,
        )
        self.play(
            Create(linf_ellipse),
            FadeIn(linf_dot), Create(linf_ring),
            FadeIn(linf_coord_bg), Write(linf_coord),
            FadeIn(linf_bottom_bg), Write(linf_bottom),
            run_time=1.5,
        )
        self.wait(2.5)

        # === Final: show all three tangent points together ===
        self.play(
            FadeOut(linf_ellipse),
            FadeOut(linf_bottom_bg), FadeOut(linf_bottom),
            FadeOut(linf_coord_bg), FadeOut(linf_coord),
            run_time=0.5,
        )

        # Recreate all three dots and rings
        all_dots = VGroup(
            Dot(axes.c2p(*l1_tangent), color=RED, radius=0.09),
            Dot(axes.c2p(*l2_tangent), color=BLUE, radius=0.09),
            Dot(axes.c2p(*linf_tangent), color=GREEN, radius=0.09),
        )
        all_rings = VGroup(
            Circle(radius=0.18, color=RED, stroke_width=2.5).move_to(axes.c2p(*l1_tangent)),
            Circle(radius=0.18, color=BLUE, stroke_width=2.5).move_to(axes.c2p(*l2_tangent)),
            Circle(radius=0.18, color=GREEN, stroke_width=2.5).move_to(axes.c2p(*linf_tangent)),
        )
        # Keep linf dot/ring, add the other two back
        final_note = Text(
            "Each norm's geometry determines WHERE on the boundary the constrained optimum lands",
            font_size=15, color=WHITE,
        )
        final_note.to_edge(DOWN, buff=0.4)
        final_bg = BackgroundRectangle(final_note, fill_opacity=0.85, buff=0.1)

        self.play(
            FadeIn(all_dots), Create(all_rings),
            FadeIn(final_bg), Write(final_note),
            run_time=1.2,
        )
        self.wait(3)
