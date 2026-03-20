from manim import *
import numpy as np


class BackpropFlow(Scene):
    def construct(self):
        # Title
        phase_label = Text("Forward Pass", font_size=28, color=BLUE)
        phase_label.to_edge(UP, buff=0.4)
        self.play(FadeIn(phase_label), run_time=0.5)

        # Network: 2 input -> 3 hidden -> 1 output
        layer_sizes = [2, 3, 1]
        layer_x = [-4.5, 0, 4.5]
        layer_names = ["Input", "Hidden", "Output"]

        # Compute neuron positions
        neuron_pos = []  # neuron_pos[layer][index] = np.array
        neurons = []     # Circle mobjects
        spacing_y = 1.4

        for li, (size, x) in enumerate(zip(layer_sizes, layer_x)):
            positions = []
            circles = []
            y_start = (size - 1) * spacing_y / 2
            for ni in range(size):
                y = y_start - ni * spacing_y
                pos = np.array([x, y, 0])
                positions.append(pos)
                circ = Circle(
                    radius=0.38, color=WHITE, stroke_width=2,
                    fill_color=BLACK, fill_opacity=1,
                ).move_to(pos)
                circles.append(circ)
            neuron_pos.append(positions)
            neurons.append(circles)

        all_circles = VGroup(*[c for layer in neurons for c in layer])
        self.play(Create(all_circles), run_time=1)

        # Layer labels below
        layer_labels = VGroup()
        for li, name in enumerate(layer_names):
            lbl = Text(name, font_size=16, color=GREY_B)
            # Position below the lowest neuron in the layer
            bottom_y = min(p[1] for p in neuron_pos[li])
            lbl.move_to([layer_x[li], bottom_y - 0.7, 0])
            layer_labels.add(lbl)
        self.play(FadeIn(layer_labels), run_time=0.5)

        # Connections (grey lines)
        conn_lines = []  # conn_lines[layer_gap] = list of Lines
        all_lines = VGroup()
        for li in range(len(layer_sizes) - 1):
            gap_lines = []
            for fi in range(layer_sizes[li]):
                for ti in range(layer_sizes[li + 1]):
                    line = Line(
                        neurons[li][fi].get_right(),
                        neurons[li + 1][ti].get_left(),
                        color=GREY_D, stroke_width=1.2, stroke_opacity=0.5,
                    )
                    all_lines.add(line)
                    gap_lines.append(line)
            conn_lines.append(gap_lines)
        self.play(Create(all_lines), run_time=0.8)

        # ===================== FORWARD PASS =====================
        fwd_values = [
            ["1", "2"],
            ["0.8", "0.3", "1.2"],
            ["0.7"],
        ]

        fwd_texts = []  # store for later removal

        for li, vals in enumerate(fwd_values):
            # Pulse dots along connections from previous layer
            if li > 0:
                pulses = VGroup()
                pulse_targets = []
                for line in conn_lines[li - 1]:
                    dot = Dot(color=BLUE, radius=0.05).move_to(line.get_start())
                    pulses.add(dot)
                    pulse_targets.append(line.get_end())

                self.play(FadeIn(pulses), run_time=0.15)
                self.play(
                    *[pulses[j].animate.move_to(pulse_targets[j]) for j in range(len(pulses))],
                    run_time=0.6,
                )
                self.play(FadeOut(pulses), run_time=0.15)

            # Show values inside circles
            layer_texts = []
            for ni, val in enumerate(vals):
                txt = Text(val, font_size=16, color=BLUE)
                txt.move_to(neurons[li][ni].get_center())
                layer_texts.append(txt)
            fwd_texts.extend(layer_texts)
            self.play(*[FadeIn(t) for t in layer_texts], run_time=0.4)

        self.wait(1)

        # ===================== BACKWARD PASS =====================
        # Fade forward values
        self.play(*[FadeOut(t) for t in fwd_texts], run_time=0.5)

        # Update phase label
        bwd_label = Text("Backward Pass", font_size=28, color=RED)
        bwd_label.to_edge(UP, buff=0.4)
        self.play(ReplacementTransform(phase_label, bwd_label), run_time=0.5)

        bwd_values = [
            ["0.2", "0.1"],
            ["0.5", "0.3", "0.8"],
            ["1.0"],
        ]

        # Go right to left: output, hidden, input
        for li in reversed(range(len(layer_sizes))):
            # Red pulses from next layer back
            if li < len(layer_sizes) - 1:
                pulses = VGroup()
                pulse_targets = []
                for line in conn_lines[li]:
                    dot = Dot(color=RED, radius=0.05).move_to(line.get_end())
                    pulses.add(dot)
                    pulse_targets.append(line.get_start())

                self.play(FadeIn(pulses), run_time=0.15)
                self.play(
                    *[pulses[j].animate.move_to(pulse_targets[j]) for j in range(len(pulses))],
                    run_time=0.6,
                )
                self.play(FadeOut(pulses), run_time=0.15)

            # Show gradient values
            layer_texts = []
            for ni, val in enumerate(bwd_values[li]):
                txt = Text(val, font_size=16, color=RED)
                txt.move_to(neurons[li][ni].get_center())
                layer_texts.append(txt)
            self.play(*[FadeIn(t) for t in layer_texts], run_time=0.4)

        self.wait(2)
