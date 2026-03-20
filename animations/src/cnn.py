from manim import *
import numpy as np


class ConvolutionSliding(Scene):
    def construct(self):
        # Title
        title = Text("2D Convolution", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)

        cell = 0.5  # cell size

        # 5x5 input values
        input_vals = np.array([
            [1, 0, 2, 1, 0],
            [0, 1, 0, 2, 1],
            [2, 0, 1, 0, 2],
            [1, 2, 0, 1, 0],
            [0, 1, 2, 0, 1],
        ])

        # 3x3 kernel
        kernel_vals = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ])

        output_size = 3

        def build_grid(values, origin, cell_sz, base_color=WHITE):
            """Build grid of rectangles + text. Returns (squares_vgroup, texts_list)."""
            rows, cols = values.shape
            squares = VGroup()
            texts = []
            for r in range(rows):
                for c in range(cols):
                    sq = Square(
                        side_length=cell_sz,
                        color=base_color,
                        stroke_width=1.5,
                        fill_color=GREY_E,
                        fill_opacity=0.3,
                    )
                    sq.move_to(origin + np.array([c * cell_sz, -r * cell_sz, 0]))
                    squares.add(sq)
                    txt = Text(str(int(values[r, c])), font_size=16, color=WHITE)
                    txt.move_to(sq.get_center())
                    texts.append(txt)
            return squares, texts

        # Positions: input LEFT, kernel ABOVE-CENTER, output RIGHT
        input_origin = np.array([-5.0, 1.0, 0])
        kernel_origin = np.array([-1.0, 2.8, 0])
        output_origin = np.array([3.0, 0.5, 0])

        # Build grids
        in_squares, in_texts = build_grid(input_vals, input_origin, cell)
        k_squares, k_texts = build_grid(kernel_vals, kernel_origin, cell, base_color=BLUE)
        # Empty output grid
        out_vals_display = np.full((output_size, output_size), 0)
        out_squares, out_texts = build_grid(out_vals_display, output_origin, cell, base_color=GREEN)

        in_group = VGroup(in_squares, *in_texts)
        k_group = VGroup(k_squares, *k_texts)
        out_group = VGroup(out_squares, *out_texts)

        # Labels
        in_label = Text("Input (5x5)", font_size=16, color=WHITE)
        in_label.next_to(in_group, UP, buff=0.25)
        k_label = Text("Kernel (3x3)", font_size=16, color=BLUE)
        k_label.next_to(k_group, UP, buff=0.25)
        out_label = Text("Output (3x3)", font_size=16, color=GREEN)
        out_label.next_to(out_group, UP, buff=0.25)

        self.play(
            FadeIn(in_group), FadeIn(in_label),
            FadeIn(k_group), FadeIn(k_label),
            FadeIn(out_group), FadeIn(out_label),
            run_time=1.2,
        )

        # Highlight rectangle for the 3x3 region on input
        highlight = Rectangle(
            width=3 * cell, height=3 * cell,
            color=YELLOW, stroke_width=3, fill_opacity=0,
        )

        # Computation display at bottom
        comp_area = Text("", font_size=16, color=YELLOW)
        comp_area.to_edge(DOWN, buff=0.6)

        # Positions to demonstrate (5 key positions covering the grid)
        positions = [(0, 0), (0, 1), (0, 2), (1, 1), (2, 2)]

        prev_comp = None

        for step, (pr, pc) in enumerate(positions):
            # Center of the 3x3 region
            center_r, center_c = pr + 1, pc + 1
            center_pos = input_origin + np.array([center_c * cell, -center_r * cell, 0])
            highlight.move_to(center_pos)

            # Compute convolution
            region = input_vals[pr:pr + 3, pc:pc + 3]
            conv_val = int(np.sum(region * kernel_vals))

            # Build compact computation string
            products = []
            for dr in range(3):
                for dc in range(3):
                    products.append(f"{region[dr, dc]}x{kernel_vals[dr, dc]}")
            comp_str = " + ".join(products[:4]) + f" + ... = {conv_val}"
            new_comp = Text(comp_str, font_size=16, color=YELLOW)
            new_comp.to_edge(DOWN, buff=0.6)

            # Output cell index
            out_idx = pr * output_size + pc
            new_out_txt = Text(str(conv_val), font_size=16, color=YELLOW)
            new_out_txt.move_to(out_squares[out_idx].get_center())

            # Highlight the output cell
            out_highlight = out_squares[out_idx].copy()
            out_highlight.set_stroke(YELLOW, width=2.5)
            out_highlight.set_fill(YELLOW, opacity=0.15)

            if step == 0:
                self.play(Create(highlight), FadeIn(new_comp), run_time=0.8)
            else:
                anims = [highlight.animate.move_to(center_pos)]
                if prev_comp is not None:
                    anims.append(FadeOut(prev_comp))
                anims.append(FadeIn(new_comp))
                if hasattr(self, '_prev_out_hl'):
                    anims.append(FadeOut(self._prev_out_hl))
                self.play(*anims, run_time=0.7)

            # Write result to output
            self.play(
                FadeIn(out_highlight),
                Transform(out_texts[out_idx], new_out_txt),
                run_time=0.5,
            )

            prev_comp = new_comp
            self._prev_out_hl = out_highlight

        self.wait(2)
