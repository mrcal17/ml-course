from manim import *
import numpy as np


class BayesTheoremUpdate(Scene):
    """Animate prior -> posterior update using Bayes' theorem with a medical test example."""

    def construct(self):
        title = Text("Bayes' Theorem: Medical Test", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        # Parameters
        prior_disease = 0.01
        prior_healthy = 0.99
        sensitivity = 0.95  # P(+|disease)
        specificity = 0.95  # P(-|healthy) => P(+|healthy) = 0.05
        fp_rate = 1 - specificity
        p_pos = sensitivity * prior_disease + fp_rate * prior_healthy
        post_disease = (sensitivity * prior_disease) / p_pos
        post_healthy = 1 - post_disease

        # Setup labels
        setup_text = VGroup(
            MathTex(r"P(\text{disease}) = 0.01", font_size=16),
            MathTex(r"\text{Sensitivity} = 0.95", font_size=16),
            MathTex(r"\text{Specificity} = 0.95", font_size=16),
        ).arrange(RIGHT, buff=0.8)
        setup_text.next_to(title, DOWN, buff=0.25)
        self.play(Write(setup_text), run_time=0.8)

        # --- PRIOR bar chart (left third) ---
        prior_title = Text("Prior", font_size=20, color=WHITE)
        prior_title.move_to(LEFT * 4.2 + UP * 1.5)

        bar_max_h = 3.0
        bar_w = 0.9

        # Healthy bar (prior)
        prior_h_bar = Rectangle(
            width=bar_w, height=bar_max_h * prior_healthy,
            fill_color=GREEN, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1,
        )
        prior_h_bar.move_to(LEFT * 4.8 + DOWN * 0.5, DOWN)
        prior_h_val = Text(f"{prior_healthy:.2f}", font_size=16, color=GREEN)
        prior_h_val.next_to(prior_h_bar, UP, buff=0.1)
        prior_h_lab = Text("Healthy", font_size=14)
        prior_h_lab.next_to(prior_h_bar, DOWN, buff=0.15)

        # Disease bar (prior)
        prior_d_bar = Rectangle(
            width=bar_w, height=max(bar_max_h * prior_disease, 0.05),
            fill_color=RED, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1,
        )
        prior_d_bar.move_to(LEFT * 3.5 + DOWN * 0.5, DOWN)
        prior_d_val = Text(f"{prior_disease:.2f}", font_size=16, color=RED)
        prior_d_val.next_to(prior_d_bar, UP, buff=0.1)
        prior_d_lab = Text("Disease", font_size=14)
        prior_d_lab.next_to(prior_d_bar, DOWN, buff=0.15)

        self.play(
            Write(prior_title),
            GrowFromEdge(prior_h_bar, DOWN), GrowFromEdge(prior_d_bar, DOWN),
            Write(prior_h_val), Write(prior_d_val),
            Write(prior_h_lab), Write(prior_d_lab),
            run_time=1.5,
        )
        self.wait(0.5)

        # --- Evidence text (center) ---
        evidence = Text("Positive test\nresult!", font_size=20, color=YELLOW)
        evidence.move_to(ORIGIN + UP * 0.3)
        evidence_box = SurroundingRectangle(evidence, color=YELLOW, buff=0.2, stroke_width=2)
        self.play(Write(evidence), Create(evidence_box), run_time=1)
        self.wait(0.5)

        # Arrow
        arrow = Arrow(LEFT * 1.2, RIGHT * 1.2, color=YELLOW, stroke_width=3)
        arrow.move_to(ORIGIN + DOWN * 0.5)
        self.play(GrowArrow(arrow), run_time=0.5)

        # --- POSTERIOR bar chart (right third) ---
        post_title = Text("Posterior", font_size=20, color=WHITE)
        post_title.move_to(RIGHT * 4.2 + UP * 1.5)

        # Healthy bar (posterior)
        post_h_bar = Rectangle(
            width=bar_w, height=bar_max_h * post_healthy,
            fill_color=GREEN, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1,
        )
        post_h_bar.move_to(RIGHT * 3.5 + DOWN * 0.5, DOWN)
        post_h_val = Text(f"{post_healthy:.2f}", font_size=16, color=GREEN)
        post_h_val.next_to(post_h_bar, UP, buff=0.1)
        post_h_lab = Text("Healthy", font_size=14)
        post_h_lab.next_to(post_h_bar, DOWN, buff=0.15)

        # Disease bar (posterior)
        post_d_bar = Rectangle(
            width=bar_w, height=bar_max_h * post_disease,
            fill_color=RED, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1,
        )
        post_d_bar.move_to(RIGHT * 4.8 + DOWN * 0.5, DOWN)
        post_d_val = Text(f"{post_disease:.2f}", font_size=16, color=RED)
        post_d_val.next_to(post_d_bar, UP, buff=0.1)
        post_d_lab = Text("Disease", font_size=14)
        post_d_lab.next_to(post_d_bar, DOWN, buff=0.15)

        self.play(
            Write(post_title),
            GrowFromEdge(post_h_bar, DOWN), GrowFromEdge(post_d_bar, DOWN),
            Write(post_h_val), Write(post_d_val),
            Write(post_h_lab), Write(post_d_lab),
            run_time=1.5,
        )
        self.wait(0.5)

        # Bayes formula with numbers at bottom
        bayes_formula = MathTex(
            r"P(D|+) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99}"
            r"= \frac{0.0095}{0.059} \approx 0.16",
            font_size=16,
            color=WHITE,
        )
        bayes_formula.to_edge(DOWN, buff=0.7)
        bayes_bg = BackgroundRectangle(bayes_formula, fill_opacity=0.85, buff=0.1)
        self.play(FadeIn(bayes_bg), Write(bayes_formula), run_time=1)

        # Key insight
        insight = Text(
            "Even with 95% accurate test, only 16% chance of disease",
            font_size=16, color=YELLOW,
        )
        insight.to_edge(DOWN, buff=0.3)
        insight_bg = BackgroundRectangle(insight, fill_opacity=0.85, buff=0.1)
        self.play(FadeIn(insight_bg), Write(insight), run_time=1)
        self.wait(2)


class CentralLimitTheorem(Scene):
    """Demonstrate CLT: sample means from Uniform(0,1) converge to Gaussian."""

    def construct(self):
        title = Text("Central Limit Theorem", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        subtitle = Text("Sample means from Uniform(0, 1)", font_size=16, color=BLUE)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(subtitle), run_time=0.5)

        # n label (will update)
        n_label = MathTex("n = 1", font_size=20, color=YELLOW)
        n_label.next_to(subtitle, DOWN, buff=0.2)
        self.play(Write(n_label), run_time=0.3)

        # Axes - use y_range that fits all histograms (n=30 is tallest ~4.5)
        axes = Axes(
            x_range=[-0.1, 1.1, 0.25],
            y_range=[0, 6, 1],
            x_length=9,
            y_length=4,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        )
        axes.shift(DOWN * 0.8)
        x_label = MathTex(r"\bar{X}", font_size=16)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.15)
        y_label = Text("Density", font_size=14)
        y_label.next_to(axes.y_axis, UP, buff=0.15)
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.8)

        # Generate data
        np.random.seed(42)
        sample_sizes = [1, 2, 5, 30]
        num_samples = 5000
        n_bins = 35

        prev_bars = None
        prev_gauss = None

        for s_idx, n in enumerate(sample_sizes):
            sample_means = np.mean(
                np.random.uniform(0, 1, size=(num_samples, n)), axis=1
            )

            counts, bin_edges = np.histogram(sample_means, bins=n_bins, range=(0, 1))
            bin_width = bin_edges[1] - bin_edges[0]

            bars = VGroup()
            for i in range(n_bins):
                x0, x1 = bin_edges[i], bin_edges[i + 1]
                density = counts[i] / (num_samples * bin_width)
                if density < 0.01:
                    continue
                # Clamp to axis range
                clamped = min(density, 5.8)
                bar = Polygon(
                    axes.c2p(x0, 0),
                    axes.c2p(x0, clamped),
                    axes.c2p(x1, clamped),
                    axes.c2p(x1, 0),
                    fill_color=BLUE, fill_opacity=0.6,
                    stroke_color=BLUE_A, stroke_width=0.8,
                )
                bars.add(bar)

            new_n_label = MathTex(f"n = {n}", font_size=20, color=YELLOW)
            new_n_label.next_to(subtitle, DOWN, buff=0.2)

            if prev_bars is None:
                self.play(
                    Create(bars, lag_ratio=0.02),
                    Transform(n_label, new_n_label),
                    run_time=1.2,
                )
                prev_bars = bars
            else:
                anims = [
                    Transform(prev_bars, bars),
                    Transform(n_label, new_n_label),
                ]
                # Remove previous gaussian if exists
                if prev_gauss is not None:
                    anims.append(FadeOut(prev_gauss))
                    prev_gauss = None
                self.play(*anims, run_time=1.2)

            # Overlay Gaussian for n >= 5
            if n >= 5:
                mu = 0.5
                sigma = 1.0 / np.sqrt(12.0 * n)
                gauss_curve = axes.plot(
                    lambda x, mu=mu, sigma=sigma: (
                        np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                        / (sigma * np.sqrt(2 * np.pi))
                    ),
                    x_range=[max(0.01, mu - 4 * sigma), min(0.99, mu + 4 * sigma), 0.005],
                    color=RED,
                    stroke_width=2.5,
                )
                gauss_label_tex = MathTex(
                    r"\mathcal{N}\left(\frac{1}{2},\, \frac{1}{12n}\right)",
                    font_size=16, color=RED,
                )
                gauss_label_tex.to_corner(UR, buff=0.5)
                gauss_label_tex.shift(DOWN * 0.8)
                gauss_bg = BackgroundRectangle(gauss_label_tex, fill_opacity=0.8, buff=0.08)

                gauss_group = VGroup(gauss_bg, gauss_label_tex, gauss_curve)
                self.play(Create(gauss_curve), FadeIn(gauss_bg), Write(gauss_label_tex), run_time=0.8)
                prev_gauss = gauss_group

            self.wait(0.6)

        # Conclusion
        conclusion = Text(
            "Sample means converge to a Normal distribution",
            font_size=16, color=GREEN,
        )
        conclusion.to_edge(DOWN, buff=0.4)
        conc_bg = BackgroundRectangle(conclusion, fill_opacity=0.85, buff=0.1)
        self.play(FadeIn(conc_bg), Write(conclusion), run_time=1)
        self.wait(1.5)
