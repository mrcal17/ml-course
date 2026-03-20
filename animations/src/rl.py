from manim import *
import numpy as np


class TDLearningUpdate(Scene):
    def construct(self):
        # Title
        title = Text("TD Learning in Grid World", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.5)

        # Grid parameters - LEFT side of screen
        ROWS, COLS = 4, 4
        CELL = 0.85
        grid_center = LEFT * 3.5 + DOWN * 0.3

        goal = (3, 3)
        obstacle = (1, 2)

        # Initialize values
        values = np.zeros((ROWS, COLS))
        values[goal] = 1.0

        # Build grid
        cells = {}
        val_texts = {}

        def cell_pos(r, c):
            return (
                grid_center
                + RIGHT * (c - COLS / 2 + 0.5) * CELL
                + DOWN * (r - ROWS / 2 + 0.5) * CELL
            )

        for r in range(ROWS):
            for c in range(COLS):
                pos = cell_pos(r, c)
                if (r, c) == goal:
                    fill_c, fill_o = GREEN, 0.4
                elif (r, c) == obstacle:
                    fill_c, fill_o = RED, 0.4
                else:
                    fill_c, fill_o = DARK_GRAY, 0.2

                cell = Square(
                    side_length=CELL,
                    fill_color=fill_c, fill_opacity=fill_o,
                    stroke_color=WHITE, stroke_width=1.5,
                )
                cell.move_to(pos)
                cells[(r, c)] = cell

                # Value text or special label
                if (r, c) == goal:
                    vt = Text("G", font_size=16, color=GREEN)
                elif (r, c) == obstacle:
                    vt = Text("X", font_size=16, color=RED)
                else:
                    vt = Text(f"{values[r, c]:.1f}", font_size=16, color=WHITE)
                vt.move_to(pos)
                val_texts[(r, c)] = vt

        # Show value under G label for goal
        goal_val = Text("1.0", font_size=12, color=GREEN)
        goal_val.next_to(val_texts[goal], DOWN, buff=0.05)

        grid_group = VGroup(*cells.values())
        val_group = VGroup(*val_texts.values())
        self.play(Create(grid_group), run_time=0.6)
        self.play(FadeIn(val_group), FadeIn(goal_val), run_time=0.5)

        # Agent dot at (0, 0)
        agent_rc = [0, 0]
        agent = Dot(
            point=cell_pos(0, 0), radius=0.12, color=YELLOW, z_index=5,
        )
        self.play(FadeIn(agent), run_time=0.4)

        # TD equation on RIGHT side
        eq_x = RIGHT * 3.0
        td_eq = MathTex(
            r"V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]",
            font_size=20,
        ).move_to(eq_x + UP * 2.5)
        self.play(FadeIn(td_eq), run_time=0.5)

        params = MathTex(r"\alpha = 0.5, \quad \gamma = 0.9", font_size=16)
        params.next_to(td_eq, DOWN, buff=0.25)
        self.play(FadeIn(params), run_time=0.4)

        # Agent path avoiding obstacle
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]
        alpha = 0.5
        gamma = 0.9
        n_steps = 4  # show 4 TD updates

        comp_area_center = eq_x + DOWN * 0.5

        for step in range(n_steps):
            s = path[step]
            s_next = path[step + 1]
            reward = 1.0 if s_next == goal else 0.0

            old_v = values[s]
            next_v = values[s_next]
            td_error = reward + gamma * next_v - old_v
            new_v = old_v + alpha * td_error
            values[s] = new_v

            # Move agent
            self.play(
                agent.animate.move_to(cell_pos(*s_next)),
                run_time=0.5,
            )

            # Show computation on the right
            line1 = Text(
                f"s=({s[0]},{s[1]}) -> s'=({s_next[0]},{s_next[1]})",
                font_size=14, color=GREY_B,
            )
            line2 = MathTex(
                f"R={reward:.0f},\\; V(s')={next_v:.2f}",
                font_size=16,
            )
            line3 = MathTex(
                f"\\delta = {reward:.0f} + {gamma}\\times{next_v:.2f} - {old_v:.2f} = {td_error:.2f}",
                font_size=16,
            )
            line4 = MathTex(
                f"V(s) \\leftarrow {old_v:.2f} + {alpha}\\times{td_error:.2f} = {new_v:.2f}",
                font_size=16, color=YELLOW,
            )
            comp = VGroup(line1, line2, line3, line4).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
            comp.move_to(comp_area_center)

            self.play(FadeIn(comp), run_time=0.5)
            self.wait(0.4)

            # Update cell value in grid (skip goal and obstacle)
            if s != goal and s != obstacle:
                new_vt = Text(f"{new_v:.2f}", font_size=16, color=YELLOW)
                new_vt.move_to(cell_pos(*s))
                # Flash cell
                flash = cells[s].copy().set_fill(YELLOW, opacity=0.4)
                self.play(
                    FadeIn(flash),
                    ReplacementTransform(val_texts[s], new_vt),
                    run_time=0.4,
                )
                val_texts[s] = new_vt
                self.play(FadeOut(flash), run_time=0.3)

            self.play(FadeOut(comp), run_time=0.3)

        # Final message
        final_msg = Text(
            "Values propagate backward from goal",
            font_size=16, color=GREEN,
        )
        final_msg.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(final_msg), run_time=0.5)

        self.wait(1.5)
