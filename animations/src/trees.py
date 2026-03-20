from manim import *
import numpy as np


class DecisionTreeGrowth(Scene):
    def construct(self):
        # Section headers
        feat_title = Text("Feature Space", font_size=20, color=WHITE)
        feat_title.move_to(LEFT * 3.5 + UP * 3.3)
        tree_title = Text("Decision Tree", font_size=20, color=WHITE)
        tree_title.move_to(RIGHT * 3.5 + UP * 3.3)

        self.play(FadeIn(feat_title), FadeIn(tree_title), run_time=0.5)

        # --- LEFT: Scatter plot ---
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=5,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.2, "color": GREY_C},
        ).move_to(LEFT * 3.5 + DOWN * 0.2)
        x_lab = MathTex(r"x_1", font_size=20).next_to(axes.x_axis, DR, buff=0.1)
        y_lab = MathTex(r"x_2", font_size=20).next_to(axes.y_axis, UL, buff=0.1)
        self.play(Create(axes), FadeIn(x_lab), FadeIn(y_lab), run_time=0.8)

        # Two-class data
        np.random.seed(7)
        class_a = np.array([
            [1.0, 1.5], [1.5, 2.0], [0.8, 3.5], [2.0, 1.0],
            [1.2, 4.2], [2.5, 0.8], [1.8, 2.5], [0.5, 2.0],
            [2.2, 3.2], [1.0, 0.5],
        ])
        class_b = np.array([
            [4.0, 4.5], [5.0, 3.0], [4.5, 5.0], [3.5, 4.2],
            [5.5, 4.5], [4.2, 2.0], [5.0, 5.5], [3.8, 3.5],
            [4.5, 1.5], [5.2, 2.2],
        ])

        dots_a = VGroup(*[
            Dot(axes.c2p(p[0], p[1]), radius=0.06, color=BLUE) for p in class_a
        ])
        dots_b = VGroup(*[
            Dot(axes.c2p(p[0], p[1]), radius=0.06, color=ORANGE) for p in class_b
        ])
        self.play(FadeIn(dots_a), FadeIn(dots_b), run_time=0.8)

        # Helper: create a tree node
        def make_node(text, pos, color=WHITE, radius=0.32):
            circ = Circle(radius=radius, color=color, stroke_width=2)
            circ.move_to(pos)
            label = MathTex(text, font_size=14, color=color).move_to(pos)
            return VGroup(circ, label)

        # Tree positions
        root_pos = RIGHT * 3.5 + UP * 2.0
        l1_left = root_pos + DOWN * 1.5 + LEFT * 1.5
        l1_right = root_pos + DOWN * 1.5 + RIGHT * 1.5
        l2_left = l1_right + DOWN * 1.5 + LEFT * 1.0
        l2_right = l1_right + DOWN * 1.5 + RIGHT * 1.0
        l3_left = l2_right + DOWN * 1.2 + LEFT * 0.7
        l3_right = l2_right + DOWN * 1.2 + RIGHT * 0.7

        # ===================== SPLIT 1: x1 = 3 vertical =====================
        split1_x = 3.0
        split1_line = DashedLine(
            axes.c2p(split1_x, 0), axes.c2p(split1_x, 6),
            color=YELLOW, stroke_width=2.5, dash_length=0.08,
        )

        # Region fills
        left_fill = Polygon(
            axes.c2p(0, 0), axes.c2p(split1_x, 0),
            axes.c2p(split1_x, 6), axes.c2p(0, 6),
            fill_color=BLUE, fill_opacity=0.08, stroke_width=0,
        )
        right_fill = Polygon(
            axes.c2p(split1_x, 0), axes.c2p(6, 0),
            axes.c2p(6, 6), axes.c2p(split1_x, 6),
            fill_color=ORANGE, fill_opacity=0.08, stroke_width=0,
        )

        # Tree: root node
        root = make_node(r"x_1 < 3?", root_pos, YELLOW)
        left_leaf1 = make_node(r"\text{A}", l1_left, BLUE, radius=0.25)
        right_node1 = make_node(r"?", l1_right, ORANGE, radius=0.25)
        edge_l = Line(root[0].get_bottom(), left_leaf1[0].get_top(), stroke_width=1.5)
        edge_r = Line(root[0].get_bottom(), right_node1[0].get_top(), stroke_width=1.5)

        self.play(
            Create(split1_line), FadeIn(left_fill), FadeIn(right_fill),
            Create(root), run_time=1.2,
        )
        self.play(
            Create(edge_l), Create(edge_r),
            FadeIn(left_leaf1), FadeIn(right_node1),
            run_time=0.8,
        )
        self.wait(0.5)

        # ===================== SPLIT 2: x2 = 3 in right region =====================
        split2_y = 3.0
        split2_line = DashedLine(
            axes.c2p(split1_x, split2_y), axes.c2p(6, split2_y),
            color=TEAL, stroke_width=2.5, dash_length=0.08,
        )

        # Update fills: right splits into bottom (blue-ish) and top (orange)
        right_bottom_fill = Polygon(
            axes.c2p(split1_x, 0), axes.c2p(6, 0),
            axes.c2p(6, split2_y), axes.c2p(split1_x, split2_y),
            fill_color=BLUE, fill_opacity=0.08, stroke_width=0,
        )
        right_top_fill = Polygon(
            axes.c2p(split1_x, split2_y), axes.c2p(6, split2_y),
            axes.c2p(6, 6), axes.c2p(split1_x, 6),
            fill_color=ORANGE, fill_opacity=0.12, stroke_width=0,
        )

        # Tree: right node becomes internal
        right_internal = make_node(r"x_2 < 3?", l1_right, TEAL, radius=0.32)
        gc_left = make_node(r"\text{A}", l2_left, BLUE, radius=0.22)
        gc_right = make_node(r"\text{B}", l2_right, ORANGE, radius=0.22)
        gc_edge_l = Line(right_internal[0].get_bottom(), gc_left[0].get_top(), stroke_width=1.5)
        gc_edge_r = Line(right_internal[0].get_bottom(), gc_right[0].get_top(), stroke_width=1.5)

        self.play(
            Create(split2_line),
            FadeOut(right_fill), FadeIn(right_bottom_fill), FadeIn(right_top_fill),
            ReplacementTransform(right_node1, right_internal),
            run_time=1,
        )
        self.play(
            Create(gc_edge_l), Create(gc_edge_r),
            FadeIn(gc_left), FadeIn(gc_right),
            run_time=0.8,
        )
        self.wait(0.5)

        # ===================== SPLIT 3: x1 = 4.5 in right-top region =====================
        split3_x = 4.5
        split3_line = DashedLine(
            axes.c2p(split3_x, split2_y), axes.c2p(split3_x, 6),
            color=PURPLE, stroke_width=2.5, dash_length=0.08,
        )

        rt_left_fill = Polygon(
            axes.c2p(split1_x, split2_y), axes.c2p(split3_x, split2_y),
            axes.c2p(split3_x, 6), axes.c2p(split1_x, 6),
            fill_color=BLUE, fill_opacity=0.1, stroke_width=0,
        )
        rt_right_fill = Polygon(
            axes.c2p(split3_x, split2_y), axes.c2p(6, split2_y),
            axes.c2p(6, 6), axes.c2p(split3_x, 6),
            fill_color=ORANGE, fill_opacity=0.15, stroke_width=0,
        )

        # Tree: gc_right becomes internal
        gc_right_internal = make_node(r"x_1 < 4.5?", l2_right, PURPLE, radius=0.32)
        ggc_left = make_node(r"\text{A}", l3_left, BLUE, radius=0.2)
        ggc_right = make_node(r"\text{B}", l3_right, ORANGE, radius=0.2)
        ggc_edge_l = Line(gc_right_internal[0].get_bottom(), ggc_left[0].get_top(), stroke_width=1.5)
        ggc_edge_r = Line(gc_right_internal[0].get_bottom(), ggc_right[0].get_top(), stroke_width=1.5)

        self.play(
            Create(split3_line),
            FadeOut(right_top_fill), FadeIn(rt_left_fill), FadeIn(rt_right_fill),
            ReplacementTransform(gc_right, gc_right_internal),
            run_time=1,
        )
        self.play(
            Create(ggc_edge_l), Create(ggc_edge_r),
            FadeIn(ggc_left), FadeIn(ggc_right),
            run_time=0.8,
        )

        self.wait(2)
