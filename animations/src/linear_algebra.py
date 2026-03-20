from manim import *
import numpy as np


class LinearTransformation2D(Scene):
    """Visualize a matrix transformation on a 2D grid with eigenvectors."""

    def construct(self):
        title = Text("Linear Transformation", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        # Matrix label in upper-right corner
        matrix_label = MathTex(
            r"A = \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}",
            font_size=20,
        )
        matrix_label.to_corner(UR, buff=0.5)
        matrix_bg = BackgroundRectangle(matrix_label, fill_opacity=0.85, buff=0.1)

        # Number plane (grid)
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=9,
            y_length=5.5,
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        )
        plane.shift(DOWN * 0.3)
        self.play(Create(plane), run_time=1)

        # Basis vectors
        origin = plane.get_origin()
        i_vec = Arrow(
            origin, origin + plane.get_x_unit_size() * RIGHT,
            buff=0, color=RED, stroke_width=4,
        )
        j_vec = Arrow(
            origin, origin + plane.get_y_unit_size() * UP,
            buff=0, color=GREEN, stroke_width=4,
        )
        i_label = MathTex(r"\hat{\imath}", font_size=20, color=RED)
        i_label.next_to(i_vec.get_end(), DR, buff=0.15)
        j_label = MathTex(r"\hat{\jmath}", font_size=20, color=GREEN)
        j_label.next_to(j_vec.get_end(), UL, buff=0.15)

        self.play(
            GrowArrow(i_vec), GrowArrow(j_vec),
            Write(i_label), Write(j_label),
            run_time=0.8,
        )
        self.play(FadeIn(matrix_bg), Write(matrix_label), run_time=0.5)
        self.wait(0.5)

        # Apply matrix transformation
        matrix = [[2, 1], [0, 1]]

        # Compute new label positions after transform
        # i_hat -> (2, 0), j_hat -> (1, 1)
        new_i_end = origin + 2 * plane.get_x_unit_size() * RIGHT
        new_j_end = origin + plane.get_x_unit_size() * RIGHT + plane.get_y_unit_size() * UP

        self.play(
            ApplyMatrix(matrix, plane),
            ApplyMatrix(matrix, i_vec),
            ApplyMatrix(matrix, j_vec),
            i_label.animate.next_to(new_i_end, DR, buff=0.15),
            j_label.animate.next_to(new_j_end, UL, buff=0.15),
            run_time=3,
        )
        self.wait(0.5)

        # Eigenvectors: eigenvalues are 2 and 1
        # eigvec for lambda=2: (1,0)
        # eigvec for lambda=1: (-1,1) normalized
        eigen_title = Text("Eigenvectors", font_size=20, color=YELLOW)
        eigen_title.to_corner(DL, buff=0.4)
        eigen_bg = BackgroundRectangle(eigen_title, fill_opacity=0.85, buff=0.1)

        origin_now = plane.get_origin()

        # Eigenvector 1: along (1,0) direction
        ev1_dir = RIGHT
        ev1 = DashedLine(
            origin_now - ev1_dir * 3, origin_now + ev1_dir * 3,
            color=YELLOW, stroke_width=2, dash_length=0.15,
        )
        ev1_lab = MathTex(r"\lambda=2", font_size=16, color=YELLOW)
        ev1_lab.next_to(origin_now + ev1_dir * 3, UP, buff=0.15)
        ev1_lab_bg = BackgroundRectangle(ev1_lab, fill_opacity=0.8, buff=0.05)

        # Eigenvector 2: along (-1,1) direction
        ev2_unit = np.array([-1, 1, 0]) / np.sqrt(2)
        # Scale to screen space
        ev2_screen = (
            -1 * plane.get_x_unit_size() * RIGHT
            + 1 * plane.get_y_unit_size() * UP
        )
        ev2_screen_norm = ev2_screen / np.linalg.norm(ev2_screen)
        ev2 = DashedLine(
            origin_now - ev2_screen_norm * 2.5,
            origin_now + ev2_screen_norm * 2.5,
            color=YELLOW, stroke_width=2, dash_length=0.15,
        )
        ev2_lab = MathTex(r"\lambda=1", font_size=16, color=YELLOW)
        ev2_lab.next_to(origin_now + ev2_screen_norm * 2.5, LEFT, buff=0.15)
        ev2_lab_bg = BackgroundRectangle(ev2_lab, fill_opacity=0.8, buff=0.05)

        self.play(
            FadeIn(eigen_bg), Write(eigen_title),
            Create(ev1), Create(ev2),
            FadeIn(ev1_lab_bg), Write(ev1_lab),
            FadeIn(ev2_lab_bg), Write(ev2_lab),
            run_time=1.5,
        )
        self.wait(1.5)


class SVDDecomposition(Scene):
    """Visualize SVD: A = U Sigma V^T as three geometric steps on a unit circle."""

    def construct(self):
        title = Text("SVD Decomposition", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        formula = MathTex(r"A = U \Sigma V^T", font_size=20)
        formula.next_to(title, DOWN, buff=0.2)
        self.play(Write(formula), run_time=0.5)

        # Matrix A = [[2,1],[1,2]]
        A = np.array([[2, 1], [1, 2]], dtype=float)
        U, s, Vt = np.linalg.svd(A)
        S = np.diag(s)

        # Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6.5,
            y_length=5,
            axis_config={"include_numbers": False, "stroke_width": 1},
        )
        axes.shift(DOWN * 0.5)
        self.play(Create(axes), run_time=0.6)

        # Unit circle as dots
        n_pts = 40
        t_vals = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        circle_pts = np.array([[np.cos(t), np.sin(t)] for t in t_vals])

        def make_shape(pts, color, axes_ref):
            """Create a closed curve from 2D points."""
            curve = VMobject(color=color, stroke_width=3)
            screen_pts = [axes_ref.c2p(p[0], p[1]) for p in pts]
            screen_pts.append(screen_pts[0])  # close the shape
            curve.set_points_as_corners(screen_pts)
            return curve

        shape = make_shape(circle_pts, BLUE, axes)
        step_label = Text("Unit Circle", font_size=20, color=BLUE)
        step_label.to_edge(DOWN, buff=0.5)
        step_bg = BackgroundRectangle(step_label, fill_opacity=0.85, buff=0.1)
        self.play(Create(shape), FadeIn(step_bg), Write(step_label), run_time=1)
        self.wait(0.5)

        # Step 1: Apply V^T (rotation)
        pts_1 = circle_pts @ Vt.T
        new_shape_1 = make_shape(pts_1, YELLOW, axes)
        new_label_1 = Text("Step 1: Rotate (V^T)", font_size=20, color=YELLOW)
        new_label_1.to_edge(DOWN, buff=0.5)
        new_bg_1 = BackgroundRectangle(new_label_1, fill_opacity=0.85, buff=0.1)
        self.play(
            Transform(shape, new_shape_1),
            FadeOut(step_bg), FadeIn(new_bg_1),
            Transform(step_label, new_label_1),
            run_time=2,
        )
        self.wait(0.3)

        # Step 2: Apply Sigma (scaling)
        pts_2 = pts_1 @ S
        new_shape_2 = make_shape(pts_2, GREEN, axes)
        new_label_2 = Text("Step 2: Scale (\u03A3)", font_size=20, color=GREEN)
        new_label_2.to_edge(DOWN, buff=0.5)
        new_bg_2 = BackgroundRectangle(new_label_2, fill_opacity=0.85, buff=0.1)
        self.play(
            Transform(shape, new_shape_2),
            FadeOut(new_bg_1), FadeIn(new_bg_2),
            Transform(step_label, new_label_2),
            run_time=2,
        )
        self.wait(0.3)

        # Step 3: Apply U (rotation)
        pts_3 = pts_2 @ U.T
        new_shape_3 = make_shape(pts_3, RED, axes)
        new_label_3 = Text("Step 3: Rotate (U)", font_size=20, color=RED)
        new_label_3.to_edge(DOWN, buff=0.5)
        new_bg_3 = BackgroundRectangle(new_label_3, fill_opacity=0.85, buff=0.1)
        self.play(
            Transform(shape, new_shape_3),
            FadeOut(new_bg_2), FadeIn(new_bg_3),
            Transform(step_label, new_label_3),
            run_time=2,
        )
        self.wait(0.3)

        # Final: show A matrix
        final_label = MathTex(
            r"A = U\Sigma V^T = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}",
            font_size=20, color=WHITE,
        )
        final_label.to_edge(DOWN, buff=0.5)
        final_bg = BackgroundRectangle(final_label, fill_opacity=0.85, buff=0.1)
        self.play(
            FadeOut(new_bg_3), FadeIn(final_bg),
            Transform(step_label, final_label),
            run_time=1,
        )
        self.wait(1.5)
