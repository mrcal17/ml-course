from manim import *
import numpy as np


class BackpropFlow(Scene):
    def construct(self):
        title = Text("Backpropagation", color=WHITE).scale(0.55)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.5)

        # Network layout: 2 input -> 3 hidden -> 1 output
        layer_sizes = [2, 3, 1]
        layer_x = [-4, 0, 4]
        spacing_y = 1.5

        neurons = []  # list of lists of Circle mobjects
        neuron_labels = []  # parallel structure for value labels

        for li, (size, x) in enumerate(zip(layer_sizes, layer_x)):
            layer = []
            labels = []
            y_start = -(size - 1) * spacing_y / 2
            for ni in range(size):
                y = y_start + ni * spacing_y
                circle = Circle(
                    radius=0.35, color=WHITE, stroke_width=2
                ).move_to([x, y, 0])
                layer.append(circle)
                labels.append(None)  # placeholder
            neurons.append(layer)
            neuron_labels.append(labels)

        # Draw neurons
        all_circles = VGroup(*[c for layer in neurons for c in layer])
        self.play(Create(all_circles), run_time=1)

        # Layer labels
        input_label = Text("Input", color=WHITE).scale(0.35)
        input_label.next_to(VGroup(*neurons[0]), DOWN, buff=0.4)
        hidden_label = Text("Hidden", color=WHITE).scale(0.35)
        hidden_label.next_to(VGroup(*neurons[1]), DOWN, buff=0.4)
        output_label = Text("Output", color=WHITE).scale(0.35)
        output_label.next_to(VGroup(*neurons[2]), DOWN, buff=0.4)
        self.play(
            FadeIn(input_label), FadeIn(hidden_label), FadeIn(output_label),
            run_time=0.5,
        )

        # Draw connections (lines between layers)
        connections = []  # list of (line, from_layer, from_idx, to_layer, to_idx)
        all_lines = VGroup()
        for li in range(len(layer_sizes) - 1):
            layer_conns = []
            for fi in range(layer_sizes[li]):
                for ti in range(layer_sizes[li + 1]):
                    line = Line(
                        neurons[li][fi].get_right(),
                        neurons[li + 1][ti].get_left(),
                        color=GREY,
                        stroke_width=1.5,
                        stroke_opacity=0.6,
                    )
                    all_lines.add(line)
                    layer_conns.append((line, li, fi, li + 1, ti))
            connections.append(layer_conns)

        self.play(Create(all_lines), run_time=1)

        # ---- FORWARD PASS ----
        fwd_label = Text("Forward Pass", color=BLUE).scale(0.45)
        fwd_label.to_edge(LEFT, buff=0.3).shift(UP * 2.5)
        self.play(FadeIn(fwd_label), run_time=0.5)

        # Input values
        input_vals = [1, 2]
        hidden_vals = [3, 1, 4]
        output_val = [5]
        all_vals = [input_vals, hidden_vals, output_val]

        # Animate forward layer by layer
        for li, vals in enumerate(all_vals):
            val_texts = []
            for ni, val in enumerate(vals):
                txt = Text(str(val), color=BLUE).scale(0.4)
                txt.move_to(neurons[li][ni].get_center())
                val_texts.append(txt)

            if li > 0:
                # Animate blue pulses along connections from previous layer
                pulses = []
                for line, fl, fi, tl, ti in connections[li - 1]:
                    pulse = Dot(color=BLUE, radius=0.06)
                    pulse.move_to(line.get_start())
                    pulses.append((pulse, line))

                pulse_dots = VGroup(*[p for p, _ in pulses])
                self.play(FadeIn(pulse_dots), run_time=0.2)
                self.play(
                    *[
                        p.animate.move_to(l.get_end())
                        for p, l in pulses
                    ],
                    run_time=0.8,
                )
                self.play(FadeOut(pulse_dots), run_time=0.2)

            self.play(*[FadeIn(t) for t in val_texts], run_time=0.5)
            # Store for later removal
            for ni, txt in enumerate(val_texts):
                neuron_labels[li][ni] = txt

        self.wait(0.5)

        # ---- BACKWARD PASS ----
        self.play(FadeOut(fwd_label), run_time=0.3)
        bwd_label = Text("Backward Pass", color=RED).scale(0.45)
        bwd_label.to_edge(LEFT, buff=0.3).shift(UP * 2.5)
        self.play(FadeIn(bwd_label), run_time=0.5)

        # Remove forward values
        all_val_mobs = VGroup(
            *[t for layer in neuron_labels for t in layer if t is not None]
        )
        self.play(FadeOut(all_val_mobs), run_time=0.5)

        # Gradient values (simplified)
        grad_vals = [
            [0.2, 0.1],   # input gradients
            [0.5, 0.3, 0.8],  # hidden gradients
            [1.0],             # output gradient (dL/dL = 1)
        ]

        # Animate backward: output -> hidden -> input
        for li in reversed(range(len(layer_sizes))):
            val_texts = []
            for ni, val in enumerate(grad_vals[li]):
                txt = Text(str(val), color=RED).scale(0.4)
                txt.move_to(neurons[li][ni].get_center())
                val_texts.append(txt)

            if li < len(layer_sizes) - 1:
                # Red pulses flowing right to left
                pulses = []
                for line, fl, fi, tl, ti in connections[li]:
                    pulse = Dot(color=RED, radius=0.06)
                    pulse.move_to(line.get_end())
                    pulses.append((pulse, line))

                pulse_dots = VGroup(*[p for p, _ in pulses])
                self.play(FadeIn(pulse_dots), run_time=0.2)
                self.play(
                    *[
                        p.animate.move_to(l.get_start())
                        for p, l in pulses
                    ],
                    run_time=0.8,
                )
                self.play(FadeOut(pulse_dots), run_time=0.2)

            self.play(*[FadeIn(t) for t in val_texts], run_time=0.5)

        self.wait(1.5)
