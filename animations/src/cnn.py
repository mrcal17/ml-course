from manim import *
import numpy as np


class ConvolutionSliding(Scene):
    def construct(self):
        title = Text("2D Convolution", color=WHITE).scale(0.55)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.5)

        cell_size = 0.55

        # 5x5 input grid values
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

        # Output grid is 3x3
        output_size = 3

        def make_grid(values, x_offset, y_offset, cell_sz, label_text=None):
            """Create a grid of squares with numbers."""
            rows, cols = values.shape
            squares = VGroup()
            texts = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=cell_sz, color=WHITE, stroke_width=1.5)
                    sq.move_to([
                        x_offset + c * cell_sz,
                        y_offset - r * cell_sz,
                        0,
                    ])
                    squares.add(sq)
                    val = int(values[r, c])
                    txt = Text(str(val), color=WHITE).scale(0.3)
                    txt.move_to(sq.get_center())
                    texts.add(txt)
            group = VGroup(squares, texts)
            if label_text:
                lbl = Text(label_text, color=WHITE).scale(0.35)
                lbl.next_to(group, UP, buff=0.2)
                return group, lbl
            return group

        # Position grids
        input_grid, input_label = make_grid(
            input_vals, -4.5, 1.2, cell_size, "Input (5x5)"
        )
        kernel_grid, kernel_label = make_grid(
            kernel_vals, -1.0, 0.6, cell_size, "Kernel (3x3)"
        )
        # Empty output grid
        output_vals = np.zeros((output_size, output_size), dtype=int)
        output_grid, output_label = make_grid(
            output_vals, 2.5, 0.6, cell_size, "Output (3x3)"
        )
        # We'll update output texts dynamically
        output_text_mobs = list(output_grid[1])  # the Text mobjects

        self.play(
            FadeIn(input_grid), FadeIn(input_label),
            FadeIn(kernel_grid), FadeIn(kernel_label),
            FadeIn(output_grid), FadeIn(output_label),
            run_time=1.5,
        )

        # Highlight rectangle for input region
        highlight = Rectangle(
            width=3 * cell_size,
            height=3 * cell_size,
            color=YELLOW,
            stroke_width=3,
        )

        # Positions to demonstrate (5 key positions)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]

        # Computation text area
        comp_text = Text("", color=YELLOW).scale(0.3)
        comp_text.to_edge(DOWN, buff=0.5)

        for step, (pr, pc) in enumerate(positions):
            # Position highlight on input grid
            center_r = pr + 1  # center of 3x3 region
            center_c = pc + 1
            input_squares = input_grid[0]
            # Get center position of the middle cell of the region
            mid_idx = center_r * 5 + center_c
            highlight.move_to(input_squares[mid_idx].get_center())

            # Compute convolution value
            region = input_vals[pr : pr + 3, pc : pc + 3]
            conv_val = int(np.sum(region * kernel_vals))

            # Highlight input squares in the region
            region_highlights = VGroup()
            for dr in range(3):
                for dc in range(3):
                    idx = (pr + dr) * 5 + (pc + dc)
                    sq_copy = input_squares[idx].copy()
                    sq_copy.set_fill(YELLOW, opacity=0.25)
                    region_highlights.add(sq_copy)

            # Build computation string
            terms = []
            for dr in range(3):
                for dc in range(3):
                    iv = int(region[dr, dc])
                    kv = int(kernel_vals[dr, dc])
                    terms.append(f"{iv}*{kv}")
            comp_str = " + ".join(terms[:5]) + " + ... = " + str(conv_val)
            new_comp = Text(comp_str, color=YELLOW).scale(0.28)
            new_comp.to_edge(DOWN, buff=0.5)

            # Update output text
            out_idx = pr * output_size + pc
            new_out_text = Text(str(conv_val), color=YELLOW).scale(0.3)
            new_out_text.move_to(output_text_mobs[out_idx].get_center())

            if step == 0:
                self.play(
                    Create(highlight),
                    FadeIn(region_highlights),
                    FadeIn(new_comp),
                    run_time=1,
                )
            else:
                self.play(
                    highlight.animate.move_to(
                        input_squares[mid_idx].get_center()
                    ),
                    FadeOut(prev_highlights),
                    FadeIn(region_highlights),
                    Transform(comp_text_mob, new_comp),
                    run_time=0.8,
                )

            # Write output value
            self.play(
                Transform(output_text_mobs[out_idx], new_out_text),
                run_time=0.5,
            )

            if step == 0:
                comp_text_mob = new_comp
            prev_highlights = region_highlights

        self.wait(1.5)
