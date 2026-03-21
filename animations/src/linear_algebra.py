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


class VectorProjection(ThreeDScene):
    """R3 projection onto a 2D orthonormal subspace using standard notation:
    v1, v2 = raw basis; u1_hat, u2_hat = unit basis; y = target; y_hat = projection;
    w1, w2 = weights; e = error."""

    def construct(self):
        # --- Raw orthogonal basis vectors in R3 ---
        v1 = np.array([3.0, 1.0, 0.0])
        v2 = np.array([-1.0, 3.0, 0.0])

        # --- Normalize: u_hat = v / ||v|| ---
        u1_hat = v1 / np.linalg.norm(v1)
        u2_hat = v2 / np.linalg.norm(v2)

        # --- Target vector (off the plane) ---
        y = np.array([2.0, 1.0, 3.0])

        # --- Weights: w_i = y . u_hat_i (since ||u_hat||=1) ---
        w1 = np.dot(y, u1_hat)
        w2 = np.dot(y, u2_hat)

        # --- Projection and error ---
        y_hat = w1 * u1_hat + w2 * u2_hat   # projection onto span{u1_hat, u2_hat}
        e = y - y_hat                         # error / residual
        e_mag = np.linalg.norm(e)

        # --- Camera ---
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        # --- Title ---
        title = Text("Projection onto Orthonormal Basis (R3 \u2192 2D plane)", font_size=24)
        title.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title), run_time=0.6)

        # --- 3D Axes ---
        axes = ThreeDAxes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[-1, 4, 1],
            x_length=6, y_length=6, z_length=5,
            axis_config={"include_numbers": False, "stroke_width": 1.5, "color": GREY_C},
        )
        self.add(axes)

        # Helpers
        def make_panel(text, color=GREY_B, font_size=15):
            t = Text(text, font_size=font_size, color=color)
            t.to_edge(DOWN, buff=0.35)
            bg = BackgroundRectangle(t, fill_opacity=0.85, buff=0.1)
            return bg, t

        def make_info_box(lines_data, position=UR):
            lines = VGroup(*[
                Text(txt, font_size=13, color=col) for txt, col in lines_data
            ])
            lines.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            lines.to_corner(position, buff=0.4).shift(DOWN * 0.5)
            box = SurroundingRectangle(
                lines, color=WHITE, buff=0.12,
                stroke_width=1, fill_color=BLACK, fill_opacity=0.85,
            )
            return box, lines

        origin = axes.c2p(0, 0, 0)

        # ============================================================
        # STEP 1: Show raw orthogonal basis v1, v2 and the subspace
        # ============================================================
        s1_bg, s1 = make_panel(
            "Step 1: v\u2081, v\u2082 — two orthogonal vectors spanning the subspace (v\u2081 \u00b7 v\u2082 = 0)"
        )
        self.add_fixed_in_frame_mobjects(s1_bg, s1)

        # Translucent plane
        plane_surf = Surface(
            lambda u, v: axes.c2p(u, v, 0),
            u_range=[-0.8, 3.5],
            v_range=[-0.8, 3.5],
            resolution=(6, 6),
            fill_opacity=0.12,
            stroke_width=0.3,
            stroke_color=BLUE_D,
        )
        plane_surf.set_color(BLUE)
        plane_label = Text("span{\u00fb\u2081, \u00fb\u2082}", font_size=14, color=BLUE_B)
        plane_label.move_to(axes.c2p(3.2, 3.2, 0))
        self.add_fixed_orientation_mobjects(plane_label)

        # v1 arrow
        v1_arrow = Arrow3D(start=origin, end=axes.c2p(*v1), color=BLUE_C, resolution=8)
        v1_label = Text("v\u2081 = (3, 1, 0)", font_size=13, color=BLUE_C)
        v1_label.move_to(axes.c2p(*(v1 + np.array([0.2, -0.3, 0.2]))))
        self.add_fixed_orientation_mobjects(v1_label)

        # v2 arrow
        v2_arrow = Arrow3D(start=origin, end=axes.c2p(*v2), color=PURPLE_B, resolution=8)
        v2_label = Text("v\u2082 = (-1, 3, 0)", font_size=13, color=PURPLE_B)
        v2_label.move_to(axes.c2p(*(v2 + np.array([-0.4, 0.3, 0.2]))))
        self.add_fixed_orientation_mobjects(v2_label)

        self.play(
            Create(plane_surf), FadeIn(plane_label),
            Create(v1_arrow), FadeIn(v1_label),
            Create(v2_arrow), FadeIn(v2_label),
            FadeIn(s1_bg), Write(s1),
            run_time=2,
        )
        self.wait(1.5)

        # ============================================================
        # STEP 2: Normalize v -> u_hat (unit vectors)
        # ============================================================
        s2_bg, s2 = make_panel(
            "Step 2: Normalize — \u00fb\u2081 = v\u2081/||v\u2081||,  \u00fb\u2082 = v\u2082/||v\u2082||  (unit vectors, ||u\u0302|| = 1)"
        )
        self.add_fixed_in_frame_mobjects(s2_bg, s2)

        norm_box, norm_lines = make_info_box([
            (f"||v\u2081|| = {np.linalg.norm(v1):.4f}", BLUE_C),
            (f"\u00fb\u2081 = v\u2081/||v\u2081|| = ({u1_hat[0]:.4f}, {u1_hat[1]:.4f}, {u1_hat[2]:.1f})", BLUE),
            (f"||v\u2082|| = {np.linalg.norm(v2):.4f}", PURPLE_B),
            (f"\u00fb\u2082 = v\u2082/||v\u2082|| = ({u2_hat[0]:.4f}, {u2_hat[1]:.4f}, {u2_hat[2]:.1f})", PURPLE),
            (f"\u00fb\u2081 \u00b7 \u00fb\u2082 = {np.dot(u1_hat, u2_hat):.1f},  ||\u00fb\u2081|| = ||\u00fb\u2082|| = 1", TEAL),
        ])
        self.add_fixed_in_frame_mobjects(norm_box, norm_lines)

        # Unit arrows
        u1_arrow = Arrow3D(start=origin, end=axes.c2p(*u1_hat), color=BLUE, resolution=8)
        u1_label_new = Text("\u00fb\u2081", font_size=16, color=BLUE)
        u1_label_new.move_to(axes.c2p(*(u1_hat * 1.3 + np.array([0.1, -0.2, 0.1]))))
        self.add_fixed_orientation_mobjects(u1_label_new)

        u2_arrow = Arrow3D(start=origin, end=axes.c2p(*u2_hat), color=PURPLE, resolution=8)
        u2_label_new = Text("\u00fb\u2082", font_size=16, color=PURPLE)
        u2_label_new.move_to(axes.c2p(*(u2_hat * 1.3 + np.array([-0.2, 0.2, 0.1]))))
        self.add_fixed_orientation_mobjects(u2_label_new)

        self.play(
            FadeOut(s1_bg), FadeOut(s1),
            FadeOut(v1_arrow), FadeOut(v1_label),
            FadeOut(v2_arrow), FadeOut(v2_label),
            Create(u1_arrow), FadeIn(u1_label_new),
            Create(u2_arrow), FadeIn(u2_label_new),
            FadeIn(norm_box), Write(norm_lines),
            FadeIn(s2_bg), Write(s2),
            run_time=2,
        )
        self.wait(1.5)

        # ============================================================
        # STEP 3: Show target vector y
        # ============================================================
        s3_bg, s3 = make_panel(
            "Step 3: y = (2, 1, 3) — the vector to project (has a component outside the plane)"
        )
        self.add_fixed_in_frame_mobjects(s3_bg, s3)

        y_arrow = Arrow3D(start=origin, end=axes.c2p(*y), color=YELLOW, resolution=8)
        y_label = Text("y = (2, 1, 3)", font_size=14, color=YELLOW)
        y_label.move_to(axes.c2p(*(y + np.array([0.3, 0.3, 0.3]))))
        self.add_fixed_orientation_mobjects(y_label)

        self.play(
            FadeOut(s2_bg), FadeOut(s2),
            FadeOut(norm_box), FadeOut(norm_lines),
            Create(y_arrow), FadeIn(y_label),
            FadeIn(s3_bg), Write(s3),
            run_time=1.5,
        )
        self.wait(1.5)

        # ============================================================
        # STEP 4: Compute weight w1 = y . u_hat_1, project onto u_hat_1
        # ============================================================
        w1_proj = w1 * u1_hat  # component along u_hat_1
        s4_bg, s4 = make_panel(
            f"Step 4: w\u2081 = y \u00b7 \u00fb\u2081 = {w1:.4f}  (weight = dot product since ||\u00fb\u2081||=1)"
        )
        self.add_fixed_in_frame_mobjects(s4_bg, s4)

        w1_box, w1_lines = make_info_box([
            ("w\u2081 = y \u00b7 \u00fb\u2081", WHITE),
            (f"   = 2({u1_hat[0]:.4f}) + 1({u1_hat[1]:.4f}) + 3({u1_hat[2]:.1f})", GREY_B),
            (f"   = {w1:.4f}", BLUE),
            (f"w\u2081\u00fb\u2081 = ({w1_proj[0]:.2f}, {w1_proj[1]:.2f}, {w1_proj[2]:.2f})", GREEN),
        ])
        self.add_fixed_in_frame_mobjects(w1_box, w1_lines)

        w1u1_arrow = Arrow3D(start=origin, end=axes.c2p(*w1_proj), color=GREEN, resolution=8)
        w1u1_label = Text(
            f"w\u2081\u00fb\u2081 = ({w1_proj[0]:.2f}, {w1_proj[1]:.2f}, {w1_proj[2]:.2f})",
            font_size=12, color=GREEN,
        )
        w1u1_label.move_to(axes.c2p(*(w1_proj + np.array([0.1, -0.4, 0.1]))))
        self.add_fixed_orientation_mobjects(w1u1_label)

        drop1 = DashedLine(
            axes.c2p(*y), axes.c2p(*w1_proj),
            color=GREEN, stroke_width=1.2, stroke_opacity=0.35, dash_length=0.08,
        )

        self.play(
            FadeOut(s3_bg), FadeOut(s3),
            Create(w1u1_arrow), FadeIn(w1u1_label),
            Create(drop1),
            FadeIn(w1_box), Write(w1_lines),
            FadeIn(s4_bg), Write(s4),
            run_time=2,
        )
        self.wait(1.5)

        # ============================================================
        # STEP 5: Compute weight w2 = y . u_hat_2, project onto u_hat_2
        # ============================================================
        w2_proj = w2 * u2_hat
        s5_bg, s5 = make_panel(f"Step 5: w\u2082 = y \u00b7 \u00fb\u2082 = {w2:.4f}")
        self.add_fixed_in_frame_mobjects(s5_bg, s5)

        w2_box, w2_lines = make_info_box([
            ("w\u2082 = y \u00b7 \u00fb\u2082", WHITE),
            (f"   = 2({u2_hat[0]:.4f}) + 1({u2_hat[1]:.4f}) + 3({u2_hat[2]:.1f})", GREY_B),
            (f"   = {w2:.4f}", PURPLE),
            (f"w\u2082\u00fb\u2082 = ({w2_proj[0]:.2f}, {w2_proj[1]:.2f}, {w2_proj[2]:.2f})", MAROON_B),
        ])
        self.add_fixed_in_frame_mobjects(w2_box, w2_lines)

        w2u2_arrow = Arrow3D(start=origin, end=axes.c2p(*w2_proj), color=MAROON_B, resolution=8)
        w2u2_label = Text(
            f"w\u2082\u00fb\u2082 = ({w2_proj[0]:.2f}, {w2_proj[1]:.2f}, {w2_proj[2]:.2f})",
            font_size=12, color=MAROON_B,
        )
        w2u2_label.move_to(axes.c2p(*(w2_proj + np.array([-0.5, 0.3, 0.1]))))
        self.add_fixed_orientation_mobjects(w2u2_label)

        drop2 = DashedLine(
            axes.c2p(*y), axes.c2p(*w2_proj),
            color=MAROON_B, stroke_width=1.2, stroke_opacity=0.35, dash_length=0.08,
        )

        self.play(
            FadeOut(s4_bg), FadeOut(s4),
            FadeOut(w1_box), FadeOut(w1_lines),
            Create(w2u2_arrow), FadeIn(w2u2_label),
            Create(drop2),
            FadeIn(w2_box), Write(w2_lines),
            FadeIn(s5_bg), Write(s5),
            run_time=2,
        )
        self.wait(1.5)

        # ============================================================
        # STEP 6: Reconstruct y_hat = w1*u_hat_1 + w2*u_hat_2
        # ============================================================
        s6_bg, s6 = make_panel(
            f"Step 6: \u0177 = w\u2081\u00fb\u2081 + w\u2082\u00fb\u2082 = ({y_hat[0]:.2f}, {y_hat[1]:.2f}, {y_hat[2]:.2f})"
        )
        self.add_fixed_in_frame_mobjects(s6_bg, s6)

        # Parallelogram
        par1 = DashedLine(
            axes.c2p(*w1_proj), axes.c2p(*y_hat),
            color=MAROON_B, stroke_width=1.5, stroke_opacity=0.5, dash_length=0.1,
        )
        par2 = DashedLine(
            axes.c2p(*w2_proj), axes.c2p(*y_hat),
            color=GREEN, stroke_width=1.5, stroke_opacity=0.5, dash_length=0.1,
        )

        yhat_dot = Dot3D(axes.c2p(*y_hat), color=WHITE, radius=0.06)
        yhat_label = Text(
            f"\u0177 = ({y_hat[0]:.2f}, {y_hat[1]:.2f}, {y_hat[2]:.2f})",
            font_size=14, color=WHITE,
        )
        yhat_label.move_to(axes.c2p(*(y_hat + np.array([0.5, -0.3, 0.0]))))
        self.add_fixed_orientation_mobjects(yhat_label)

        recon_box, recon_lines = make_info_box([
            ("\u0177 = w\u2081\u00fb\u2081 + w\u2082\u00fb\u2082", WHITE),
            (f"  = {w1:.2f}\u00fb\u2081 + {w2:.2f}\u00fb\u2082", GREY_B),
            (f"  = ({y_hat[0]:.2f}, {y_hat[1]:.2f}, {y_hat[2]:.2f})", WHITE),
            ("\u0177 lies in the plane", BLUE_B),
        ])
        self.add_fixed_in_frame_mobjects(recon_box, recon_lines)

        self.play(
            FadeOut(s5_bg), FadeOut(s5),
            FadeOut(w2_box), FadeOut(w2_lines),
            Create(par1), Create(par2),
            FadeIn(yhat_dot), FadeIn(yhat_label),
            FadeIn(recon_box), Write(recon_lines),
            FadeIn(s6_bg), Write(s6),
            run_time=2,
        )
        self.wait(1.5)

        # ============================================================
        # STEP 7: Error e = y - y_hat (perpendicular to the plane)
        # ============================================================
        s7_bg, s7 = make_panel(
            f"Step 7: e = y - \u0177 = ({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}),  ||e|| = {e_mag:.2f}"
        )
        self.add_fixed_in_frame_mobjects(s7_bg, s7)

        e_arrow = Arrow3D(
            start=axes.c2p(*y_hat), end=axes.c2p(*y),
            color=RED, resolution=8,
        )
        e_label = Text(
            f"e = ({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f})",
            font_size=13, color=RED,
        )
        e_mid = (y_hat + y) / 2
        e_label.move_to(axes.c2p(*(e_mid + np.array([0.6, 0.3, 0.0]))))
        self.add_fixed_orientation_mobjects(e_label)

        # Right-angle marker
        ms = 0.2
        e_dir = e / e_mag
        in_plane_dir = y_hat / np.linalg.norm(y_hat)
        ra_pts = [
            y_hat + ms * in_plane_dir,
            y_hat + ms * in_plane_dir + ms * e_dir,
            y_hat + ms * e_dir,
        ]
        right_angle = VGroup(
            Line(axes.c2p(*ra_pts[0]), axes.c2p(*ra_pts[1]), color=WHITE, stroke_width=1.5),
            Line(axes.c2p(*ra_pts[1]), axes.c2p(*ra_pts[2]), color=WHITE, stroke_width=1.5),
        )

        err_box, err_lines = make_info_box([
            (f"e \u00b7 \u00fb\u2081 = {np.dot(e, u1_hat):.6f}  (= 0)", TEAL),
            (f"e \u00b7 \u00fb\u2082 = {np.dot(e, u2_hat):.6f}  (= 0)", TEAL),
            ("e \u22a5 plane  (perpendicular)", TEAL),
            (f"||e|| = {e_mag:.4f}", RED),
        ])
        self.add_fixed_in_frame_mobjects(err_box, err_lines)

        self.play(
            FadeOut(s6_bg), FadeOut(s6),
            FadeOut(recon_box), FadeOut(recon_lines),
            Create(e_arrow),
            Create(right_angle),
            FadeIn(e_label),
            FadeIn(err_box), Write(err_lines),
            FadeIn(s7_bg), Write(s7),
            run_time=2,
        )
        self.wait(2)

        # ============================================================
        # FINAL: Full decomposition y = y_hat + e + camera orbit
        # ============================================================
        summary = VGroup(
            Text("y = \u0177 + e  =  w\u2081\u00fb\u2081 + w\u2082\u00fb\u2082 + e", font_size=15, color=WHITE),
            Text(
                f"(2,1,3) = {w1:.2f}\u00fb\u2081 + {w2:.2f}\u00fb\u2082 + ({e[0]:.2f},{e[1]:.2f},{e[2]:.2f})",
                font_size=13, color=GREY_B,
            ),
            Text(
                "w\u1d62 = y \u00b7 \u00fb\u1d62  (orthonormal \u2192 weight is just the dot product)",
                font_size=13, color=TEAL,
            ),
        )
        summary.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        summary.to_edge(DOWN, buff=0.3)
        sum_bg = BackgroundRectangle(summary, fill_opacity=0.85, buff=0.1)
        self.add_fixed_in_frame_mobjects(sum_bg, summary)

        self.play(
            FadeOut(s7_bg), FadeOut(s7),
            FadeOut(err_box), FadeOut(err_lines),
            FadeIn(sum_bg), Write(summary),
            run_time=1,
        )

        # Slow orbit
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(5)
        self.stop_ambient_camera_rotation()


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
