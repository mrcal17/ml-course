from manim import *
import numpy as np


class RNNUnrolling(Scene):
    def construct(self):
        # Title
        title = Text("RNN Unrolling & Vanishing Gradient", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.5)

        # ---- Phase 1: Single RNN cell with self-loop ----
        cell = RoundedRectangle(
            width=1.6, height=1.0, corner_radius=0.15,
            color=BLUE, stroke_width=3, fill_color=BLUE, fill_opacity=0.15,
        ).move_to(ORIGIN)
        cell_label = Text("RNN", font_size=20, color=BLUE)
        cell_label.move_to(cell.get_center())

        # Self-loop: curved arrow arcing above the cell
        loop_start = cell.get_top() + RIGHT * 0.35
        loop_end = cell.get_top() + LEFT * 0.35
        self_loop = CurvedArrow(
            loop_start, loop_end,
            angle=-TAU / 2, color=YELLOW, stroke_width=2.5,
        ).shift(UP * 0.15)
        loop_label = MathTex(r"h_t", font_size=20, color=YELLOW)
        loop_label.next_to(self_loop, UP, buff=0.1)

        # Input arrow from below
        in_arrow = Arrow(
            cell.get_bottom() + DOWN * 0.7, cell.get_bottom(),
            color=WHITE, buff=0, stroke_width=2,
        )
        in_label = MathTex(r"x_t", font_size=20, color=WHITE)
        in_label.next_to(in_arrow, DOWN, buff=0.1)

        # Output arrow going up
        out_arrow = Arrow(
            cell.get_top(), cell.get_top() + UP * 0.7,
            color=WHITE, buff=0, stroke_width=2,
        )
        out_label = MathTex(r"y_t", font_size=20, color=WHITE)
        out_label.next_to(out_arrow, UP, buff=0.1)

        folded = VGroup(
            cell, cell_label, self_loop, loop_label,
            in_arrow, in_label, out_arrow, out_label,
        )

        self.play(Create(cell), FadeIn(cell_label), run_time=0.6)
        self.play(
            Create(self_loop), FadeIn(loop_label),
            GrowArrow(in_arrow), FadeIn(in_label),
            GrowArrow(out_arrow), FadeIn(out_label),
            run_time=0.8,
        )
        self.wait(0.6)

        # ---- Phase 2: Unroll into 4 time steps ----
        self.play(FadeOut(folded), run_time=0.4)

        num_steps = 4
        box_w, box_h = 1.2, 0.8
        gap = 2.2
        center_y = 0.0
        start_x = -(num_steps - 1) * gap / 2

        boxes = []
        w_labels = []
        t_labels = []
        x_arrows = []
        x_labels = []
        h_arrows = []
        h_labels = []
        conn_arrows = []

        for t in range(num_steps):
            x = start_x + t * gap
            pos = np.array([x, center_y, 0])

            box = RoundedRectangle(
                width=box_w, height=box_h, corner_radius=0.1,
                color=BLUE, stroke_width=3, fill_color=BLUE, fill_opacity=0.15,
            ).move_to(pos)
            boxes.append(box)

            wl = Text("RNN", font_size=16, color=BLUE)
            wl.move_to(box.get_center())
            w_labels.append(wl)

            tl = MathTex(f"t={t + 1}", font_size=16, color=GREY_B)
            tl.next_to(box, DOWN, buff=0.25)
            t_labels.append(tl)

            # Input from below (below time label)
            x_start_pt = box.get_bottom() + DOWN * 1.0
            x_arr = Arrow(x_start_pt, box.get_bottom(), color=WHITE, buff=0.05, stroke_width=2)
            x_arrows.append(x_arr)
            xl = MathTex(f"x_{t + 1}", font_size=20, color=WHITE)
            xl.next_to(x_arr, DOWN, buff=0.1)
            x_labels.append(xl)

            # Output above
            h_end_pt = box.get_top() + UP * 1.0
            h_arr = Arrow(box.get_top(), h_end_pt, color=WHITE, buff=0.05, stroke_width=2)
            h_arrows.append(h_arr)
            hl = MathTex(f"h_{t + 1}", font_size=20, color=WHITE)
            hl.next_to(h_arr, UP, buff=0.1)
            h_labels.append(hl)

            # Horizontal connection
            if t > 0:
                conn = Arrow(
                    boxes[t - 1].get_right(), box.get_left(),
                    color=YELLOW, buff=0.08, stroke_width=2.5,
                )
                conn_arrows.append(conn)

        # Animate unrolled cells
        self.play(
            *[Create(b) for b in boxes],
            *[FadeIn(l) for l in w_labels],
            *[FadeIn(t) for t in t_labels],
            run_time=0.8,
        )
        self.play(
            *[GrowArrow(a) for a in x_arrows],
            *[FadeIn(l) for l in x_labels],
            *[GrowArrow(a) for a in h_arrows],
            *[FadeIn(l) for l in h_labels],
            run_time=0.7,
        )
        self.play(*[GrowArrow(a) for a in conn_arrows], run_time=0.6)

        # Shared weights note
        shared = Text("Same W", font_size=16, color=BLUE)
        shared.next_to(VGroup(*boxes), UP, buff=1.8)
        brace = Brace(VGroup(*boxes), UP, buff=1.5, color=BLUE)
        brace.scale(np.array([1, 0.5, 1]))  # make brace shorter vertically
        shared.next_to(brace, UP, buff=0.1)
        self.play(Create(brace), FadeIn(shared), run_time=0.5)
        self.wait(0.5)

        # ---- Phase 3: Vanishing gradient ----
        self.play(FadeOut(brace), FadeOut(shared), run_time=0.3)

        # Backward gradient arrows (right to left) getting thinner and more transparent
        grad_widths = [5.0, 3.5, 2.0, 0.8]
        grad_opacities = [1.0, 0.7, 0.4, 0.15]
        grad_arrows = []

        for i in range(len(conn_arrows)):
            idx = len(conn_arrows) - 1 - i  # start from rightmost
            c = conn_arrows[idx]
            grad = Arrow(
                c.get_end(), c.get_start(),
                color=RED, buff=0.08,
                stroke_width=grad_widths[i],
                stroke_opacity=grad_opacities[i],
            ).shift(DOWN * 0.35)
            grad_arrows.append(grad)
            self.play(GrowArrow(grad), run_time=0.4)

        # Gradient magnitude labels above each cell
        grad_vals = [0.1, 0.3, 0.6, 1.0]
        grad_labels = []
        for t in range(num_steps):
            gl = MathTex(f"|\\nabla| = {grad_vals[t]}", font_size=16, color=RED)
            gl.set_opacity(grad_opacities[num_steps - 1 - t] if t < num_steps else 1.0)
            gl.next_to(h_labels[t], RIGHT * 0, buff=0).shift(UP * 0.35)
            grad_labels.append(gl)

        self.play(*[FadeIn(g) for g in grad_labels], run_time=0.5)

        # Bottom label
        vanish_label = Text("Gradients vanish over long sequences", font_size=16, color=RED)
        vanish_label.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(vanish_label), run_time=0.5)

        self.wait(1.5)
