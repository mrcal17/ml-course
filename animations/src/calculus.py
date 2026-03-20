from manim import *
import numpy as np


class GradientVector2D(Scene):
    """Contour plot of f(x,y) = x^2 + 2y^2 with gradient arrows and gradient descent."""

    def construct(self):
        # Title and function label
        title = Text("Gradient Vectors & Gradient Descent", font_size=28)
        title.to_edge(UP, buff=0.3)
        func_label = MathTex(r"f(x,y) = x^2 + 2y^2", font_size=20)
        func_label.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), run_time=0.8)
        self.play(Write(func_label), run_time=0.5)

        # Axes (no tick numbers to avoid clutter)
        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        )
        axes.shift(DOWN * 0.5)
        x_lab = MathTex("x", font_size=20).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_lab = MathTex("y", font_size=20).next_to(axes.y_axis, UP, buff=0.1)
        self.play(Create(axes), Write(x_lab), Write(y_lab), run_time=0.8)

        # Contour ellipses for f(x,y) = x^2 + 2y^2 = c
        contour_colors = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, TEAL_A]
        levels = [0.5, 1.5, 3.0, 5.0, 8.0, 12.0]
        contours = VGroup()
        for c_val, color in zip(levels, contour_colors):
            a = np.sqrt(c_val)
            b = np.sqrt(c_val / 2)
            ellipse = axes.plot_parametric_curve(
                lambda t, a=a, b=b: np.array([a * np.cos(t), b * np.sin(t), 0]),
                t_range=[0, 2 * np.pi, 0.02],
                color=color,
                stroke_width=1.5,
            )
            contours.add(ellipse)
        self.play(Create(contours, lag_ratio=0.08), run_time=1.5)

        # Gradient arrows at 8 points, normalized length
        # grad f = (2x, 4y)
        gradient_points = [
            (1.5, 1.0), (-1.5, 1.0), (1.5, -1.0), (-1.5, -1.0),
            (2.2, 0.0), (0.0, 1.5), (-2.2, 0.0), (0.0, -1.5),
        ]
        arrows = VGroup()
        for px, py in gradient_points:
            gx, gy = 2 * px, 4 * py
            mag = np.sqrt(gx**2 + gy**2)
            if mag < 0.01:
                continue
            # Normalize to fixed screen-space length
            unit_gx, unit_gy = gx / mag, gy / mag
            arrow_len = 0.6  # screen-space length in axis units
            start = axes.c2p(px, py)
            end = axes.c2p(px + unit_gx * arrow_len, py + unit_gy * arrow_len)
            arrow = Arrow(
                start, end, buff=0, color=RED, stroke_width=3,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(arrow)

        # Legend in lower-left corner
        legend = MathTex(
            r"\nabla f = \text{direction of steepest ascent}",
            font_size=16, color=RED,
        )
        legend.to_corner(DL, buff=0.4)
        legend_bg = BackgroundRectangle(legend, fill_opacity=0.8, buff=0.1)

        self.play(
            Create(arrows, lag_ratio=0.05),
            FadeIn(legend_bg), Write(legend),
            run_time=1.5,
        )
        self.wait(1)

        # Transition to gradient descent
        self.play(FadeOut(arrows), FadeOut(legend), FadeOut(legend_bg), run_time=0.5)

        descent_label = Text("Gradient Descent", font_size=20, color=YELLOW)
        descent_label.to_corner(DL, buff=0.4)
        descent_bg = BackgroundRectangle(descent_label, fill_opacity=0.8, buff=0.1)
        self.play(FadeIn(descent_bg), Write(descent_label), run_time=0.5)

        # Starting point and gradient descent path
        lr = 0.1
        pos = np.array([2.5, 1.5])
        dot = Dot(axes.c2p(*pos), color=YELLOW, radius=0.1)
        start_label = Text("Start", font_size=16, color=YELLOW)
        start_label.next_to(dot, UR, buff=0.15)
        self.play(FadeIn(dot), Write(start_label), run_time=0.5)

        path_points = [axes.c2p(*pos)]
        for _ in range(25):
            grad = np.array([2 * pos[0], 4 * pos[1]])
            pos = pos - lr * grad
            path_points.append(axes.c2p(*pos))

        traced_path = VMobject(color=YELLOW, stroke_width=3)
        traced_path.set_points_as_corners(path_points)
        self.play(
            FadeOut(start_label),
            MoveAlongPath(dot, traced_path),
            Create(traced_path),
            run_time=3,
        )

        min_label = MathTex(r"\text{min}", font_size=16, color=GREEN)
        min_label.next_to(axes.c2p(0, 0), DR, buff=0.15)
        self.play(Write(min_label), run_time=0.5)
        self.wait(1)


class ChainRuleComputationGraph(Scene):
    """Computation graph: x -> [*w] -> [+b] -> [sigma] -> L with forward/backward pass."""

    def construct(self):
        title = Text("Chain Rule via Computation Graph", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        # Node setup - wide spacing across screen
        node_names = ["x", r"\times w", "+ b", r"\sigma", "L"]
        x_positions = [-5.5, -2.75, 0, 2.75, 5.5]
        node_y = 0.0

        nodes = VGroup()
        node_texts = VGroup()
        for i, (name, xp) in enumerate(zip(node_names, x_positions)):
            rect = RoundedRectangle(
                width=1.4, height=0.8, corner_radius=0.12,
                stroke_color=WHITE, fill_color=GREY_E, fill_opacity=0.9,
            )
            rect.move_to([xp, node_y, 0])
            label = MathTex(name, font_size=20)
            label.move_to(rect.get_center())
            nodes.add(rect)
            node_texts.add(label)

        self.play(
            *[Create(n) for n in nodes],
            *[Write(t) for t in node_texts],
            run_time=1,
        )

        # Forward pass arrows and values (ABOVE nodes)
        fwd_label = Text("Forward Pass", font_size=20, color=BLUE)
        fwd_label.next_to(title, DOWN, buff=0.2)
        self.play(Write(fwd_label), run_time=0.4)

        # Compute values: x=2, w=0.7, b=0.5
        # x=2 -> 2*0.7=1.4 -> 1.4+0.5=1.9 -> sigma(1.9)=0.87 -> L
        fwd_values = ["2.0", "1.4", "1.9", "0.87", "L"]
        fwd_arrows = VGroup()
        fwd_val_labels = VGroup()

        for i in range(len(nodes)):
            # Value label above node
            val = Text(fwd_values[i], font_size=16, color=BLUE)
            val.next_to(nodes[i], UP, buff=0.3)
            fwd_val_labels.add(val)

            if i > 0:
                arrow = Arrow(
                    nodes[i - 1].get_right() + UP * 0.15,
                    nodes[i].get_left() + UP * 0.15,
                    buff=0.08, color=BLUE, stroke_width=2.5,
                    max_tip_length_to_length_ratio=0.15,
                )
                fwd_arrows.add(arrow)
                self.play(GrowArrow(arrow), FadeIn(val), run_time=0.35)
            else:
                self.play(FadeIn(val), run_time=0.3)

        self.wait(0.3)

        # Backward pass arrows and values (BELOW nodes)
        self.play(FadeOut(fwd_label), run_time=0.3)
        bwd_label = Text("Backward Pass (Chain Rule)", font_size=20, color=RED)
        bwd_label.next_to(title, DOWN, buff=0.2)
        self.play(Write(bwd_label), run_time=0.4)

        # Gradient values (simplified, illustrative)
        grad_values = [
            r"\frac{\partial L}{\partial x}",
            r"\frac{\partial L}{\partial(xw)}",
            r"\frac{\partial L}{\partial z}",
            r"\frac{\partial L}{\partial \hat{y}}",
        ]
        grad_nums = ["-0.26", "-0.13", "-0.13", "-1.15"]

        bwd_arrows = VGroup()
        grad_labels = VGroup()

        for idx in range(len(nodes) - 1):
            i = len(nodes) - 1 - idx  # go right to left
            arrow = Arrow(
                nodes[i].get_left() + DOWN * 0.15,
                nodes[i - 1].get_right() + DOWN * 0.15,
                buff=0.08, color=RED, stroke_width=2.5,
                max_tip_length_to_length_ratio=0.15,
            )
            bwd_arrows.add(arrow)

            grad_text = Text(grad_nums[idx], font_size=14, color=RED)
            grad_text.next_to(arrow, DOWN, buff=0.2)
            grad_labels.add(grad_text)

            self.play(GrowArrow(arrow), FadeIn(grad_text), run_time=0.35)

        # Chain rule formula at bottom
        chain_formula = MathTex(
            r"\frac{\partial L}{\partial w} = "
            r"\frac{\partial L}{\partial \hat{y}} \cdot "
            r"\frac{\partial \hat{y}}{\partial z} \cdot "
            r"\frac{\partial z}{\partial w}",
            font_size=20,
            color=YELLOW,
        )
        chain_formula.to_edge(DOWN, buff=0.5)
        chain_bg = BackgroundRectangle(chain_formula, fill_opacity=0.8, buff=0.1)
        self.play(FadeIn(chain_bg), Write(chain_formula), run_time=1)
        self.wait(1.5)
