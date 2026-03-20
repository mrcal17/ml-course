from manim import *
import numpy as np


class AttentionWeights(Scene):
    def construct(self):
        # Title at top
        title = Text("Self-Attention Mechanism", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.5)

        # Token boxes centered vertically, slightly above center
        tokens = ["The", "cat", "sat", "down"]
        token_y = 0.3
        token_spacing = 2.5
        start_x = -(len(tokens) - 1) * token_spacing / 2

        token_boxes = []
        token_labels = []
        for i, tok in enumerate(tokens):
            x = start_x + i * token_spacing
            box = RoundedRectangle(
                width=1.3, height=0.7, corner_radius=0.1,
                stroke_color=WHITE, stroke_width=2,
                fill_color=DARK_GRAY, fill_opacity=0.3,
            ).move_to([x, token_y, 0])
            label = Text(tok, font_size=20, color=WHITE)
            label.move_to(box.get_center())
            token_boxes.append(box)
            token_labels.append(label)

        self.play(
            *[Create(b) for b in token_boxes],
            *[FadeIn(l) for l in token_labels],
            run_time=0.8,
        )
        self.wait(0.3)

        # Highlight "sat" as query (index 2)
        query_idx = 2
        query_highlight = SurroundingRectangle(
            token_boxes[query_idx], color=BLUE, buff=0.08, stroke_width=3,
        )
        query_tag = Text("query", font_size=16, color=BLUE)
        query_tag.next_to(token_boxes[query_idx], DOWN, buff=0.25)
        self.play(Create(query_highlight), FadeIn(query_tag), run_time=0.6)

        # ---- Step 1: Dot product scores ----
        step_label = Text("Dot product scores", font_size=16, color=YELLOW)
        step_label.to_corner(UL, buff=0.6).shift(DOWN * 0.5)
        self.play(FadeIn(step_label), run_time=0.4)

        raw_scores = [0.5, 1.2, 2.8, 0.3]
        score_texts = []
        score_y = token_y + 0.75
        for i in range(len(tokens)):
            x = start_x + i * token_spacing
            st = Text(f"{raw_scores[i]:.1f}", font_size=20, color=YELLOW)
            st.move_to([x, score_y, 0])
            score_texts.append(st)

        self.play(*[FadeIn(s) for s in score_texts], run_time=0.6)
        self.wait(0.5)

        # ---- Step 2: Softmax → attention weights ----
        new_step_label = Text("After softmax", font_size=16, color=YELLOW)
        new_step_label.move_to(step_label.get_center())

        exp_scores = np.exp(raw_scores)
        attn_weights = exp_scores / exp_scores.sum()
        weight_vals = [round(float(w), 2) for w in attn_weights]

        weight_texts = []
        for i in range(len(tokens)):
            x = start_x + i * token_spacing
            wt = Text(f"{weight_vals[i]:.2f}", font_size=20, color=YELLOW)
            wt.move_to([x, score_y, 0])
            weight_texts.append(wt)

        self.play(
            ReplacementTransform(step_label, new_step_label),
            *[ReplacementTransform(score_texts[i], weight_texts[i]) for i in range(4)],
            run_time=0.8,
        )
        self.wait(0.5)

        # ---- Step 3: Attention arrows from "sat" to each token ----
        attn_arrows = []
        query_box = token_boxes[query_idx]
        for i in range(len(tokens)):
            w = weight_vals[i]
            thickness = 1.5 + w * 10
            opacity = 0.25 + w * 0.75
            if i == query_idx:
                # Self-attention: curved arrow looping above
                arr = CurvedArrow(
                    query_box.get_top() + RIGHT * 0.25,
                    query_box.get_top() + LEFT * 0.25,
                    angle=-TAU / 3, color=BLUE,
                    stroke_width=thickness, stroke_opacity=opacity,
                ).shift(UP * 0.6)
                attn_arrows.append(arr)
            else:
                target_box = token_boxes[i]
                # Arrow from query to target at the box top level
                arr = Arrow(
                    query_box.get_top() + UP * 0.15,
                    target_box.get_top() + UP * 0.15,
                    color=BLUE, buff=0.15,
                    stroke_width=thickness, stroke_opacity=opacity,
                    max_tip_length_to_length_ratio=0.08,
                )
                attn_arrows.append(arr)

        self.play(*[Create(a) for a in attn_arrows], run_time=0.8)
        self.wait(0.5)

        # ---- Step 4: Output equation at bottom ----
        output_eq = MathTex(
            r"\text{Output} = \sum_i \alpha_i V_i",
            font_size=20, color=GREEN,
        )
        output_eq.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(output_eq), run_time=0.6)

        self.wait(1.5)
