"""Animated explanation of Algorithm 1 and root-drift prefetching policies.

Render a continuous 1080p video:
    conda run -n specdiff manim -qh slides/algorithm_one.py AlgorithmOneSpeculativeTree

Render/present with interactive slide boundaries:
    conda run -n specdiff manim-slides render -qh \
        slides/algorithm_one.py AlgorithmOneSpeculativeTree
    conda run -n specdiff manim-slides present AlgorithmOneSpeculativeTree
"""

from __future__ import annotations

import numpy as np

from manim import (
    AnimationGroup,
    Arrow,
    Circle,
    Create,
    DashedLine,
    DashedVMobject,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromCenter,
    Indicate,
    LEFT,
    Line,
    ManimColor,
    MathTex,
    ParametricFunction,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text as ManimText,
    TransformFromCopy,
    UP,
    VGroup,
    config,
)

try:
    from manim_slides import Slide
except ImportError:  # Keep the file usable with plain Manim.
    Slide = Scene


config.background_color = "#0B1020"

BG = "#0B1020"
INK = "#F5F7FF"
MUTED = "#9AA7BD"
FAINT = "#33415C"
DRAFT = "#58A6FF"
VERIFY = "#B794F4"
ACCEPT = "#71D99E"
RESIDUAL = "#FFB454"
DANGER = "#FF6B6B"
TEXT_FONT = "Helvetica"


def Text(content: str, *args, **kwargs) -> ManimText | VGroup:
    """Create Helvetica text with stable, explicit inter-word spacing.

    Pango stores ordinary spaces only as positional advances rather than SVG
    glyphs.  Those narrow advances can look inconsistent after Manim groups or
    transforms the surrounding glyphs.  Rendering each word independently
    makes the gap geometric and therefore stable throughout every animation.
    """
    kwargs.setdefault("font", TEXT_FONT)
    kwargs.setdefault("disable_ligatures", True)

    if "\n" in content or " " not in content:
        return ManimText(content, *args, **kwargs)

    font_size = float(kwargs.get("font_size", 48))
    word_gap = 0.0065 * font_size
    raw_parts = content.split(" ")
    words = VGroup()
    spaces_before = 0
    previous = None
    for part in raw_parts:
        if not part:
            spaces_before += 1
            continue
        word = ManimText(part, *args, **kwargs)
        if previous is None:
            word.move_to((0, 0, 0))
        else:
            gap_factor = 1.0 + 0.45 * spaces_before
            word.next_to(
                previous,
                RIGHT,
                buff=word_gap * gap_factor,
            )
        words.add(word)
        previous = word
        spaces_before = 0
    words.move_to((0, 0, 0))
    return words


class AlgorithmOneSpeculativeTree(Slide):
    """A slow, centered walkthrough of drafting, verification, and prefetching."""

    def construct(self) -> None:
        self.camera.background_color = ManimColor(BG)
        header = self._header("ALGORITHM 1  ·  ONE SPECULATIVE ROUND")
        phases = self._phase_labels()
        tree = self._tree_geometry()

        self.play(FadeIn(header, shift=0.15 * DOWN), run_time=1.3)
        self.play(FadeIn(phases, shift=0.12 * DOWN), run_time=0.9)
        self.play(GrowFromCenter(tree["nodes"][0]), run_time=1.1)
        root_caption = Text("current state", font_size=24, color=ACCEPT)
        root_caption.next_to(tree["nodes"][0], LEFT, buff=0.28)
        self.play(FadeIn(root_caption, shift=0.1 * RIGHT), run_time=0.7)
        self._pause(1.6)

        # ------------------------------------------------ Phase 1: drafting
        self.play(
            self._activate_phase(phases, 0),
            FadeOut(root_caption),
            run_time=1.0,
        )
        # First expose the deterministic proposal mean at the current realised
        # parent. The delayed increment is frozen throughout the draft tree.
        root_mean_formula = MathTex(
            r"m^p_n(y_n)=y_n+\gamma b^q_{t_n}(\widetilde y_n)",
            font_size=29,
            color=INK,
        )
        root_mean_formula.set_color_by_tex("m^p_n", DRAFT)
        root_mean_formula.move_to((-3.72, 1.56, 0))
        self.play(FadeIn(root_mean_formula, shift=0.12 * RIGHT), run_time=1.4)
        self._pause(1.5)

        # The bell curve is a one-dimensional view of the isotropic Gaussian.
        # Its centre is the proposal mean computed immediately above.
        root_gaussian = self._gaussian_packet(
            center_x=0.0,
            baseline_y=0.48,
            width=3.55,
            height=0.66,
            sigma=0.62,
            show_mean=True,
            mean_tex=r"m^p_n(y_n)",
            mean_font_size=21,
        )
        gaussian_label = MathTex(
            r"\mathcal N\!\left(m^p_n(y_n),\sigma_n^2 I\right)",
            font_size=27,
            color=DRAFT,
        )
        gaussian_label.move_to((2.65, 1.02, 0))
        level_one_edges = tree["edges"][:2]
        self.play(
            FadeIn(root_gaussian),
            FadeIn(gaussian_label),
            run_time=1.8,
        )
        self._pause(1.6)

        # Sample two deliberately asymmetric x-axis locations. Each marked
        # realisation becomes one child only after its edge is drawn.
        first_root_sample = Dot(
            np.array([-0.94, 0.48, 0.0]), radius=0.082, color=DRAFT
        )
        first_root_tick = Line(
            (-0.94, 0.37, 0), (-0.94, 0.59, 0), color=DRAFT, stroke_width=3.0
        )
        self.play(
            FadeIn(first_root_sample, scale=1.8),
            Create(first_root_tick),
            run_time=1.0,
        )
        self.play(Indicate(first_root_sample, color=DRAFT, scale_factor=1.9), run_time=1.0)
        self.play(
            Create(level_one_edges[0]),
            ReplacementTransform(first_root_sample, tree["nodes"][1]),
            FadeOut(first_root_tick),
            run_time=1.9,
        )

        second_root_sample = Dot(
            np.array([0.53, 0.48, 0.0]), radius=0.082, color=DRAFT
        )
        second_root_tick = Line(
            (0.53, 0.37, 0), (0.53, 0.59, 0), color=DRAFT, stroke_width=3.0
        )
        self.play(
            FadeIn(second_root_sample, scale=1.8),
            Create(second_root_tick),
            run_time=1.0,
        )
        self.play(Indicate(second_root_sample, color=DRAFT, scale_factor=1.9), run_time=1.0)
        self.play(
            Create(level_one_edges[1]),
            ReplacementTransform(second_root_sample, tree["nodes"][2]),
            FadeOut(second_root_tick),
            run_time=1.9,
        )
        self._pause(2.0)
        self.play(
            FadeOut(root_gaussian),
            FadeOut(gaussian_label),
            FadeOut(root_mean_formula),
            run_time=1.0,
        )

        # Every realised parent at depth one repeats the same sequence. The
        # frozen drift is unchanged; only the parent state being translated is
        # different.
        level_two_edges = tree["edges"][2:]
        left_mean_formula = MathTex(
            r"m^p_{n+1}(y^{(1)}_{n+1})"
            r"=y^{(1)}_{n+1}+\gamma b^q_{t_n}(\widetilde y_n)",
            font_size=22,
            color=INK,
        )
        left_mean_formula.set_color_by_tex(r"m^p_{n+1}", DRAFT)
        left_mean_formula.move_to((-4.46, 0.12, 0))
        self.play(FadeIn(left_mean_formula, shift=0.10 * RIGHT), run_time=1.3)

        left_gaussian = self._gaussian_packet(
            -2.05,
            -1.06,
            2.05,
            0.48,
            0.40,
            show_mean=True,
            mean_tex=r"m^p_{n+1}(y^{(1)}_{n+1})",
            mean_font_size=15,
        )
        left_distribution = MathTex(
            r"\mathcal N\!\left(m^p_{n+1}(y^{(1)}_{n+1}),\sigma_{n+1}^2I\right)",
            font_size=18,
            color=DRAFT,
        )
        left_distribution.move_to((-4.55, -0.70, 0))
        self.play(
            FadeIn(left_gaussian),
            FadeIn(left_distribution),
            run_time=1.5,
        )
        self._pause(1.4)

        left_sample_one = Dot(
            np.array([-2.67, -1.06, 0.0]), radius=0.070, color=DRAFT
        )
        left_tick_one = Line(
            (-2.67, -1.15, 0), (-2.67, -0.97, 0), color=DRAFT, stroke_width=2.8
        )
        self.play(FadeIn(left_sample_one, scale=1.7), Create(left_tick_one), run_time=0.9)
        self.play(Indicate(left_sample_one, color=DRAFT, scale_factor=1.8), run_time=0.9)
        self.play(
            Create(level_two_edges[0]),
            ReplacementTransform(left_sample_one, tree["nodes"][3]),
            FadeOut(left_tick_one),
            run_time=1.7,
        )

        left_sample_two = Dot(
            np.array([-1.71, -1.06, 0.0]), radius=0.070, color=DRAFT
        )
        left_tick_two = Line(
            (-1.71, -1.15, 0), (-1.71, -0.97, 0), color=DRAFT, stroke_width=2.8
        )
        self.play(FadeIn(left_sample_two, scale=1.7), Create(left_tick_two), run_time=0.9)
        self.play(Indicate(left_sample_two, color=DRAFT, scale_factor=1.8), run_time=0.9)
        self.play(
            Create(level_two_edges[1]),
            ReplacementTransform(left_sample_two, tree["nodes"][4]),
            FadeOut(left_tick_two),
            run_time=1.7,
        )
        self._pause(1.6)
        self.play(
            FadeOut(left_gaussian),
            FadeOut(left_distribution),
            FadeOut(left_mean_formula),
            run_time=0.9,
        )

        right_mean_formula = MathTex(
            r"m^p_{n+1}(y^{(2)}_{n+1})"
            r"=y^{(2)}_{n+1}+\gamma b^q_{t_n}(\widetilde y_n)",
            font_size=22,
            color=INK,
        )
        right_mean_formula.set_color_by_tex(r"m^p_{n+1}", DRAFT)
        right_mean_formula.move_to((4.46, 0.12, 0))
        self.play(FadeIn(right_mean_formula, shift=0.10 * LEFT), run_time=1.3)

        right_gaussian = self._gaussian_packet(
            2.05,
            -1.06,
            2.05,
            0.48,
            0.40,
            show_mean=True,
            mean_tex=r"m^p_{n+1}(y^{(2)}_{n+1})",
            mean_font_size=15,
        )
        right_distribution = MathTex(
            r"\mathcal N\!\left(m^p_{n+1}(y^{(2)}_{n+1}),\sigma_{n+1}^2I\right)",
            font_size=18,
            color=DRAFT,
        )
        right_distribution.move_to((4.55, -0.70, 0))
        self.play(
            FadeIn(right_gaussian),
            FadeIn(right_distribution),
            run_time=1.5,
        )
        self._pause(1.4)

        right_sample_one = Dot(
            np.array([1.63, -1.06, 0.0]), radius=0.070, color=DRAFT
        )
        right_tick_one = Line(
            (1.63, -1.15, 0), (1.63, -0.97, 0), color=DRAFT, stroke_width=2.8
        )
        self.play(FadeIn(right_sample_one, scale=1.7), Create(right_tick_one), run_time=0.9)
        self.play(Indicate(right_sample_one, color=DRAFT, scale_factor=1.8), run_time=0.9)
        self.play(
            Create(level_two_edges[2]),
            ReplacementTransform(right_sample_one, tree["nodes"][5]),
            FadeOut(right_tick_one),
            run_time=1.7,
        )

        right_sample_two = Dot(
            np.array([2.74, -1.06, 0.0]), radius=0.070, color=DRAFT
        )
        right_tick_two = Line(
            (2.74, -1.15, 0), (2.74, -0.97, 0), color=DRAFT, stroke_width=2.8
        )
        self.play(FadeIn(right_sample_two, scale=1.7), Create(right_tick_two), run_time=0.9)
        self.play(Indicate(right_sample_two, color=DRAFT, scale_factor=1.8), run_time=0.9)
        self.play(
            Create(level_two_edges[3]),
            ReplacementTransform(right_sample_two, tree["nodes"][6]),
            FadeOut(right_tick_two),
            run_time=1.7,
        )
        self._pause(1.8)
        self.play(
            FadeOut(right_gaussian),
            FadeOut(right_distribution),
            FadeOut(right_mean_formula),
            run_time=0.9,
        )

        # --------------------------------------------- Phase 2: verification
        # Compress the branch spacing slightly while the tree moves left.  The
        # circles keep their size; only their horizontal positions and the
        # connecting arrows change.
        verification_tree = self._tree_geometry(x_scale=0.78, x_shift=-3.45)
        self.play(
            self._activate_phase(phases, 1),
            ReplacementTransform(tree["edges"], verification_tree["edges"]),
            *(
                node.animate.move_to(target)
                for node, target in zip(tree["nodes"], verification_tree["nodes"])
            ),
            run_time=2.0,
        )
        tree["edges"] = verification_tree["edges"]

        internal_halos = VGroup(
            *(self._mean_halo(node) for node in tree["nodes"][:3])
        )
        internal_caption = MathTex(
            r"\mathcal{I}(T_n)=\{u:C(u)\neq\varnothing\}",
            font_size=25,
            color=VERIFY,
        ).move_to((-3.45, 2.22, 0))
        self.play(
            AnimationGroup(*(Create(halo) for halo in internal_halos), lag_ratio=0.18),
            FadeIn(internal_caption),
            AnimationGroup(
                *(self._pulse(node, VERIFY) for node in tree["nodes"][:3]),
                lag_ratio=0.15,
            ),
            run_time=2.2,
        )

        stack_nodes = VGroup(*(node.copy().scale(0.72) for node in tree["nodes"][:3]))
        row_y = (1.05, 0.18, -0.69)
        for node, y_pos in zip(stack_nodes, row_y):
            node.move_to((0.12, y_pos, 0))
        stack_label = VGroup(
            Text("ONE BATCH", font_size=22, weight="BOLD", color=VERIFY),
            MathTex(r"u\in\mathcal{I}(T_n)", font_size=22, color=MUTED),
        ).arrange(DOWN, buff=0.05)
        stack_label.move_to((0.12, 1.78, 0))
        self.play(
            AnimationGroup(
                *(TransformFromCopy(source, target) for source, target in zip(tree["nodes"][:3], stack_nodes)),
                lag_ratio=0.22,
            ),
            FadeIn(stack_label, shift=0.10 * UP),
            run_time=2.5,
        )

        target_model = self._target_model(VERIFY, width=1.90, height=3.10)
        target_model.move_to((3.18, 0.18, 0))
        batch_arrow = Arrow(
            (stack_nodes.get_right()[0] + 0.10, 0.18, 0),
            (target_model.get_left()[0] - 0.10, 0.18, 0),
            buff=0.18,
            stroke_width=3.4,
            max_tip_length_to_length_ratio=0.08,
            color=VERIFY,
        )
        self.play(FadeIn(target_model), GrowArrow(batch_arrow), run_time=1.7)

        drift_rows = VGroup(
            MathTex(r"b^q_{t_n}(y_n)", font_size=22, color=VERIFY),
            MathTex(r"b^q_{t_{n+1}}(y^{(1)}_{n+1})", font_size=20, color=VERIFY),
            MathTex(r"b^q_{t_{n+1}}(y^{(2)}_{n+1})", font_size=20, color=VERIFY),
        )
        output_arrow = Arrow(
            np.zeros(3),
            0.72 * RIGHT,
            buff=0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.09,
            color=VERIFY,
        )
        output_arrow.move_to(
            (target_model.get_right()[0] + 0.44, row_y[0], 0)
        )
        drift_arrows = VGroup(
            *(output_arrow.copy().set_y(y_pos) for y_pos in row_y)
        )
        for row, arrow in zip(drift_rows, drift_arrows):
            row.next_to(arrow, RIGHT, buff=0.12)
        for arrow, row in zip(drift_arrows, drift_rows):
            self.play(Create(arrow), FadeIn(row, shift=0.12 * RIGHT), run_time=1.0)
        self._pause(2.2)

        # Each computed drift now lands in its own explicit target-mean formula.
        # The left-hand sides are deliberately isolated as MathTex parts so
        # they can travel into the target cards in the next transition.
        target_mean_rows = VGroup(
            MathTex(
                r"m^q_n(y_n)",
                "=",
                r"y_n+\gamma b^q_{t_n}(y_n)",
                font_size=22,
                color=INK,
            ),
            MathTex(
                r"m^q_{n+1}(y^{(1)}_{n+1})",
                "=",
                r"y^{(1)}_{n+1}+\gamma b^q_{t_{n+1}}(y^{(1)}_{n+1})",
                font_size=19,
                color=INK,
            ),
            MathTex(
                r"m^q_{n+1}(y^{(2)}_{n+1})",
                "=",
                r"y^{(2)}_{n+1}+\gamma b^q_{t_{n+1}}(y^{(2)}_{n+1})",
                font_size=19,
                color=INK,
            ),
        )
        for row, y_pos in zip(target_mean_rows, row_y):
            row.move_to((3.15, y_pos, 0))
            row[0].set_color(VERIFY)
            row.set_color_by_tex("b^q", VERIFY)
        self.play(
            FadeOut(target_model),
            FadeOut(batch_arrow),
            FadeOut(stack_nodes),
            FadeOut(stack_label),
            FadeOut(drift_arrows),
            *(
                ReplacementTransform(drift, mean)
                for drift, mean in zip(drift_rows, target_mean_rows)
            ),
            run_time=2.2,
        )
        self._pause(2.0)

        # Preserve visual identity: each displayed m^q left-hand side is the
        # very same object inside its target card.  Only its size and position
        # change; no replacement/morph is used for this transition.
        proposal_mean_tex = (
            r"m^p_n(y_n)",
            r"m^p_{n+1}(y^{(1)}_{n+1})",
            r"m^p_{n+1}(y^{(2)}_{n+1})",
        )
        target_mean_tex = (
            r"m^q_n(y_n)",
            r"m^q_{n+1}(y^{(1)}_{n+1})",
            r"m^q_{n+1}(y^{(2)}_{n+1})",
        )
        proposal_cards = VGroup(
            *(
                self._mean_card(tex, DRAFT, font_size=19 if i == 0 else 16)
                for i, tex in enumerate(proposal_mean_tex)
            )
        )
        target_cards = VGroup(
            *(
                self._mean_card(tex, VERIFY, font_size=19 if i == 0 else 16)
                for i, tex in enumerate(target_mean_tex)
            )
        )
        mean_card_pairs = VGroup(
            *(
                VGroup(proposal, target).arrange(RIGHT, buff=0.14)
                for proposal, target in zip(proposal_cards, target_cards)
            )
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        mean_card_pairs.move_to((1.48, 0.44, 0))
        card_headers = VGroup(
            Text("PROPOSAL", font_size=16, weight="BOLD", color=DRAFT),
            Text("TARGET", font_size=16, weight="BOLD", color=VERIFY),
        )
        card_headers[0].next_to(proposal_cards[0], UP, buff=0.12)
        card_headers[1].next_to(target_cards[0], UP, buff=0.12)

        # The labels created by _mean_card serve only as invisible layout
        # guides.  The visible labels remain the original left-hand sides of
        # the three equations above.
        for target_card in target_cards:
            target_card[1].set_opacity(0)
        target_lhs = VGroup(*(equation[0] for equation in target_mean_rows))

        equation_remainders = [
            VGroup(equation[1], equation[2]) for equation in target_mean_rows
        ]
        self.play(
            *(FadeOut(remainder) for remainder in equation_remainders),
            run_time=0.8,
        )
        self.play(
            *(
                lhs.animate.scale(
                    target_card[1].height / lhs.height,
                    about_point=lhs.get_center(),
                ).move_to(target_card[1].get_center())
                for lhs, target_card in zip(target_lhs, target_cards)
            ),
            *(FadeIn(target_card[0]) for target_card in target_cards),
            *(
                FadeIn(proposal_card, shift=0.18 * RIGHT)
                for proposal_card in proposal_cards
            ),
            FadeIn(card_headers, shift=0.08 * DOWN),
            run_time=1.9,
        )
        self._pause(1.5)

        # The remaining Verify arguments enter vertically from below only
        # after all three proposal/target pairs have visibly formed.
        verify_engine = self._verify_box(width=1.72, height=2.62, font_size=24)
        verify_engine.move_to((5.62, 0.44, 0))
        pair_y = mean_card_pairs.get_center()[1]
        verify_arrow = Arrow(
            (mean_card_pairs.get_right()[0] + 0.08, pair_y, 0),
            (verify_engine.get_left()[0] - 0.08, pair_y, 0),
            buff=0.16,
            stroke_width=3.4,
            max_tip_length_to_length_ratio=0.08,
            color=VERIFY,
        )
        child_batch_nodes = VGroup(
            *(node.copy().scale(0.34) for node in tree["nodes"][1:])
        )
        for node in child_batch_nodes:
            node[0].set_stroke(width=1.35)
        child_batch_nodes.arrange_in_grid(rows=2, cols=3, buff=(0.13, 0.10))
        child_batch_label = Text(
            "DRAFTED CHILDREN",
            font_size=15,
            weight="BOLD",
            color=DRAFT,
        )
        sigma_input = MathTex(
            r"\{\sigma_{n+|u|}\}_{u\in\mathcal{I}(T_n)}",
            font_size=19,
            color=MUTED,
        )
        auxiliary_inputs = VGroup(
            child_batch_label,
            child_batch_nodes,
            sigma_input,
        ).arrange(DOWN, buff=0.10)
        auxiliary_inputs.move_to((5.62, -2.45, 0))
        auxiliary_arrow = Arrow(
            (5.62, auxiliary_inputs.get_top()[1] + 0.06, 0),
            (5.62, verify_engine.get_bottom()[1] - 0.06, 0),
            buff=0.12,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.10,
            color=VERIFY,
        )
        verify_contract = MathTex(
            r"(y,\ \mathrm{accepted},\ v^\star)",
            font_size=23,
            color=INK,
        ).move_to((5.62, 2.15, 0))
        self.play(
            FadeIn(verify_engine),
            GrowArrow(verify_arrow),
            FadeIn(verify_contract, shift=0.10 * DOWN),
            run_time=1.0,
        )
        self.play(
            AnimationGroup(
                *(
                    TransformFromCopy(source, target)
                    for source, target in zip(
                        tree["nodes"][1:], child_batch_nodes
                    )
                ),
                lag_ratio=0.09,
            ),
            FadeIn(child_batch_label, shift=0.08 * UP),
            FadeIn(sigma_input, shift=0.08 * UP),
            run_time=1.8,
        )
        self.play(GrowArrow(auxiliary_arrow), run_time=0.7)
        self._pause(2.3)

        # ------------------------------------------------ Phase 3: acceptance
        # Keep the verification tree compressed and left-aligned.  Acceptance
        # happens beside it, preserving the spatial context established above.
        self.play(
            self._activate_phase(phases, 2),
            FadeOut(internal_halos),
            FadeOut(internal_caption),
            FadeOut(mean_card_pairs),
            FadeOut(target_lhs),
            FadeOut(card_headers),
            FadeOut(verify_engine),
            FadeOut(verify_arrow),
            FadeOut(auxiliary_inputs),
            FadeOut(auxiliary_arrow),
            FadeOut(verify_contract),
            run_time=1.7,
        )

        # VERIFY consumes the complete first depth in one squeeze.  The first
        # proposal rejects and the second accepts, so only one node turns green.
        node_verify = self._verify_box(width=1.52, height=0.70, font_size=18)
        node_verify.move_to((0.72, 0.02, 0))
        depth_one_label = Text("DEPTH 1", font_size=20, weight="BOLD", color=MUTED)
        depth_one_label.next_to(node_verify, UP, buff=0.15)
        level_one_pair = VGroup(
            tree["nodes"][1].copy(),
            tree["nodes"][2].copy(),
        ).scale(0.52)
        level_one_pair.arrange(RIGHT, buff=0.08).move_to(node_verify)
        self.play(
            FadeIn(node_verify),
            FadeIn(depth_one_label),
            TransformFromCopy(
                VGroup(tree["nodes"][1], tree["nodes"][2]),
                level_one_pair,
            ),
            run_time=1.2,
        )
        self.play(
            node_verify.animate.stretch(0.52, 0),
            level_one_pair.animate.stretch(0.24, 0),
            run_time=0.55,
        )
        depth_one_verdicts = VGroup(
            Text("REJECT", font_size=17, weight="BOLD", color=RESIDUAL),
            Text("ACCEPT", font_size=17, weight="BOLD", color=ACCEPT),
        )
        depth_one_verdicts[0].next_to(tree["nodes"][1], DOWN, buff=0.10)
        depth_one_verdicts[1].next_to(tree["nodes"][2], DOWN, buff=0.10)
        self.play(
            node_verify.animate.stretch(1 / 0.52, 0),
            FadeOut(level_one_pair),
            FadeIn(depth_one_verdicts, shift=0.08 * UP),
            tree["nodes"][1][0].animate.set_stroke(RESIDUAL, width=4).set_fill(
                RESIDUAL, opacity=0.22
            ),
            tree["edges"][1].animate.set_color(ACCEPT).set_stroke(width=5),
            tree["nodes"][2][0].animate.set_stroke(ACCEPT, width=4).set_fill(
                ACCEPT, opacity=0.22
            ),
            run_time=0.85,
        )
        self._pause(1.8)

        # At depth two, VERIFY again receives both children together.  The two
        # possible outcomes are then kept side-by-side in a persistent table.
        depth_two_label = Text("DEPTH 2", font_size=20, weight="BOLD", color=MUTED)
        depth_two_position = np.array([0.72, -1.72, 0.0])
        depth_two_label.move_to(depth_two_position + 0.58 * UP)
        self.play(
            FadeOut(depth_one_label),
            FadeOut(depth_one_verdicts),
            node_verify.animate.move_to(depth_two_position),
            FadeIn(depth_two_label),
            run_time=0.9,
        )
        level_two_pair = VGroup(
            tree["nodes"][5].copy(),
            tree["nodes"][6].copy(),
        ).scale(0.52)
        level_two_pair.arrange(RIGHT, buff=0.08).move_to(node_verify)
        self.play(
            TransformFromCopy(
                VGroup(tree["nodes"][5], tree["nodes"][6]),
                level_two_pair,
            ),
            run_time=0.9,
        )
        self.play(
            node_verify.animate.stretch(0.52, 0),
            level_two_pair.animate.stretch(0.24, 0),
            run_time=0.55,
        )
        self.play(
            node_verify.animate.stretch(1 / 0.52, 0),
            FadeOut(level_two_pair),
            run_time=0.65,
        )

        case_a_panel = RoundedRectangle(
            width=2.20,
            height=2.25,
            corner_radius=0.12,
            fill_color=ACCEPT,
            fill_opacity=0.045,
            stroke_color=ACCEPT,
            stroke_width=1.8,
            stroke_opacity=0.55,
        ).move_to((3.05, -0.25, 0))
        case_b_panel = RoundedRectangle(
            width=2.20,
            height=2.25,
            corner_radius=0.12,
            fill_color=RESIDUAL,
            fill_opacity=0.045,
            stroke_color=RESIDUAL,
            stroke_width=1.8,
            stroke_opacity=0.55,
        ).move_to((5.55, -0.25, 0))
        case_a_header = VGroup(
            Text("CASE A", font_size=21, weight="BOLD", color=ACCEPT),
            Text("accepted leaf", font_size=18, color=MUTED),
        ).arrange(DOWN, buff=0.05).move_to((3.05, 0.50, 0))
        case_b_header = VGroup(
            Text("CASE B", font_size=21, weight="BOLD", color=RESIDUAL),
            Text("both leaves rejected", font_size=16, color=MUTED),
        ).arrange(DOWN, buff=0.05).move_to((5.55, 0.50, 0))
        case_a_outcome = self._state_node(r"y_{n+2}", ACCEPT, radius=0.30)
        case_a_outcome.move_to((3.05, -0.55, 0))
        case_b_outcome = self._state_node(r"y_{n+2}", RESIDUAL, radius=0.30)
        case_b_outcome.move_to((5.55, -0.55, 0))
        case_a_note = Text("drafted leaf", font_size=17, color=ACCEPT)
        case_a_note.next_to(case_a_outcome, DOWN, buff=0.12)
        case_b_note = Text("residual sample", font_size=17, color=RESIDUAL)
        case_b_note.next_to(case_b_outcome, DOWN, buff=0.12)

        self.play(
            FadeIn(case_a_panel),
            FadeIn(case_b_panel),
            FadeIn(case_a_header, shift=0.08 * DOWN),
            FadeIn(case_b_header, shift=0.08 * DOWN),
            TransformFromCopy(tree["nodes"][5], case_a_outcome),
            TransformFromCopy(tree["nodes"][5], case_b_outcome),
            FadeIn(case_a_note),
            FadeIn(case_b_note),
            run_time=1.6,
        )
        case_table = VGroup(
            case_a_panel,
            case_b_panel,
            case_a_header,
            case_b_header,
            case_a_outcome,
            case_b_outcome,
            case_a_note,
            case_b_note,
        )
        self._pause(2.0)

        # Both possible outputs lack the target drift needed at the next root.
        in_both_cases = Text(
            "IN BOTH CASES",
            font_size=20,
            weight="BOLD",
            color=MUTED,
        ).move_to((4.30, -1.73, 0))
        missing_drift = MathTex(
            r"b^q_{t_{n+2}}(y_{n+2})",
            font_size=27,
            color=MUTED,
        ).move_to((4.30, -2.18, 0))
        missing_cross = self._cross(
            missing_drift.get_center(),
            size=0.28,
            color=DANGER,
        )
        self.play(FadeIn(in_both_cases, shift=0.08 * UP), run_time=0.8)
        self.play(
            FadeIn(missing_drift),
            FadeIn(missing_cross, scale=1.25),
            run_time=1.0,
        )
        self._pause(1.5)
        prefetching_label = Text(
            "PREFETCHING",
            font_size=25,
            weight="BOLD",
            color=VERIFY,
        ).move_to((4.30, -3.00, 0))
        prefetching_rule = Line(
            (-1.0, 0, 0),
            (1.0, 0, 0),
            color=VERIFY,
            stroke_width=2.6,
        ).match_width(prefetching_label)
        prefetching_rule.next_to(prefetching_label, DOWN, buff=0.09)
        prefetching_teaser = VGroup(prefetching_label, prefetching_rule)
        self.play(FadeIn(prefetching_teaser, shift=0.12 * UP), run_time=1.0)
        self._pause(1.8)

        # ------------------------------------ Between rounds: prefetching
        prefetch_header = self._header("LEAF OUTPUTS  ·  PREFETCH THE NEXT ROOT DRIFT")
        self.play(
            ReplacementTransform(header, prefetch_header),
            FadeOut(phases),
            FadeOut(node_verify),
            FadeOut(depth_two_label),
            FadeOut(case_table),
            FadeOut(in_both_cases),
            FadeOut(missing_drift),
            FadeOut(missing_cross),
            FadeOut(prefetching_teaser),
            tree["nodes"][3:].animate.set_opacity(0.13),
            tree["edges"][2:].animate.set_opacity(0.13),
            run_time=1.6,
        )

        # The two available level-one states are the nearest-prefetch
        # candidates.  Their actual graphical nodes replace placeholders inside
        # the mathematical expression on the free right-hand side.
        level_one_halos = VGroup(
            self._mean_halo(tree["nodes"][1]),
            self._mean_halo(tree["nodes"][2]),
        )
        self.play(
            Create(level_one_halos),
            AnimationGroup(
                self._pulse(tree["nodes"][1], VERIFY),
                self._pulse(tree["nodes"][2], VERIFY),
                lag_ratio=0.18,
            ),
            run_time=1.7,
        )

        formula_head = MathTex(
            r"v^\star\in\arg\min",
            font_size=34,
            color=INK,
        )
        placeholder_one = DashedVMobject(
            Circle(radius=0.27, stroke_color=FAINT, stroke_width=2.0),
            num_dashes=10,
        )
        placeholder_two = placeholder_one.copy()
        first_left = MathTex(r"\lVert y_{n+2}-", font_size=29, color=INK)
        first_right = MathTex(r"\rVert,", font_size=29, color=INK)
        second_left = MathTex(r"\lVert y_{n+2}-", font_size=29, color=INK)
        second_right = MathTex(r"\rVert", font_size=29, color=INK)
        set_left = MathTex(r"\{", font_size=42, color=INK)
        set_right = MathTex(r"\}", font_size=42, color=INK)
        nearest_formula_shell = VGroup(
            formula_head,
            set_left,
            first_left,
            placeholder_one,
            first_right,
            second_left,
            placeholder_two,
            second_right,
            set_right,
        ).arrange(RIGHT, buff=0.06)
        nearest_formula_shell.scale_to_fit_width(6.55)
        nearest_formula_shell.move_to((3.55, 0.15, 0))

        formula_node_one = tree["nodes"][1].copy()
        formula_node_two = tree["nodes"][2].copy()
        formula_node_one.scale_to_fit_height(placeholder_one.height * 1.02)
        formula_node_two.scale_to_fit_height(placeholder_two.height * 1.02)
        formula_node_one.move_to(placeholder_one)
        formula_node_two.move_to(placeholder_two)
        formula_without_placeholders = VGroup(
            formula_head,
            set_left,
            set_right,
            first_left,
            first_right,
            second_left,
            second_right,
        )
        self.play(
            FadeIn(formula_without_placeholders, shift=0.08 * UP),
            run_time=1.4,
        )
        self.play(
            TransformFromCopy(tree["nodes"][1], formula_node_one),
            TransformFromCopy(tree["nodes"][2], formula_node_two),
            run_time=1.8,
        )
        self._pause(1.8)

        # Select the nearer graphical candidate, then name it as the prefetched
        # state used to construct the following proposal mean.
        self.play(
            first_left.animate.set_opacity(0.28),
            first_right.animate.set_opacity(0.28),
            formula_node_one.animate.set_opacity(0.28),
            Indicate(formula_node_two, color=VERIFY, scale_factor=1.20),
            run_time=1.0,
        )
        tilde_prefix = MathTex(
            r"\widetilde y_{n+2}=",
            font_size=36,
            color=VERIFY,
        )
        tilde_node = formula_node_two.copy().scale(1.35)
        tilde_group = VGroup(tilde_prefix, tilde_node).arrange(RIGHT, buff=0.14)
        tilde_group.move_to((3.55, -2.18, 0))
        tilde_caption = Text(
            "node chosen by prefetch",
            font_size=20,
            color=MUTED,
        ).next_to(tilde_group, DOWN, buff=0.12)
        self.play(
            FadeIn(tilde_prefix, shift=0.10 * UP),
            TransformFromCopy(formula_node_two, tilde_node),
            FadeIn(tilde_caption, shift=0.08 * UP),
            run_time=1.2,
        )
        self._pause(2.3)

        # The next round repeats the exact grammar of Phase 1: reveal the mean,
        # reveal its Gaussian, then sample and connect one child at a time.
        next_round_header = self._header("NEXT ROUND  ·  DRAFT FROM THE PREFETCHED DRIFT")
        next_root = self._state_node(r"y_{n+2}", ACCEPT, radius=0.36)
        next_root.move_to((0.0, 1.55, 0))
        prefetch_screen = VGroup(
            tree["nodes"],
            tree["edges"],
            level_one_halos,
            formula_without_placeholders,
            formula_node_one,
            formula_node_two,
            tilde_prefix,
            tilde_node,
            tilde_caption,
        )
        self.play(
            ReplacementTransform(prefetch_header, next_round_header),
            FadeOut(prefetch_screen),
            GrowFromCenter(next_root),
            run_time=1.6,
        )
        prefetched_mean = MathTex(
            r"m^p_{n+2}(y_{n+2})=y_{n+2}+\gamma "
            r"b^q_{t_{n+2}}(\widetilde y_{n+2})",
            font_size=29,
            color=INK,
        )
        prefetched_mean.set_color_by_tex(r"m^p_{n+2}", DRAFT)
        prefetched_mean.set_color_by_tex(
            r"b^q_{t_{n+2}}(\widetilde y_{n+2})", VERIFY
        )
        prefetched_mean.move_to((-3.25, 1.55, 0))
        self.play(FadeIn(prefetched_mean, shift=0.12 * RIGHT), run_time=1.4)
        self._pause(1.4)

        next_gaussian = self._gaussian_packet(
            center_x=0.0,
            baseline_y=0.44,
            width=3.55,
            height=0.66,
            sigma=0.62,
            show_mean=True,
            mean_tex=r"m^p_{n+2}(y_{n+2})",
            mean_font_size=20,
        )
        next_distribution = MathTex(
            r"\mathcal N\!\left(m^p_{n+2}(y_{n+2}),\sigma_{n+2}^2I\right)",
            font_size=26,
            color=DRAFT,
        ).move_to((2.95, 1.00, 0))
        next_children = VGroup(
            self._state_node(r"y^{(1)}_{n+3}", DRAFT, radius=0.31),
            self._state_node(r"y^{(2)}_{n+3}", DRAFT, radius=0.31),
        )
        next_children[0].move_to((-1.75, -1.28, 0))
        next_children[1].move_to((1.75, -1.28, 0))
        next_edges = VGroup(
            Arrow(
                next_root.get_bottom(),
                next_children[0].get_top(),
                buff=0.10,
                stroke_width=2.6,
                max_tip_length_to_length_ratio=0.09,
                color=FAINT,
            ),
            Arrow(
                next_root.get_bottom(),
                next_children[1].get_top(),
                buff=0.10,
                stroke_width=2.6,
                max_tip_length_to_length_ratio=0.09,
                color=FAINT,
            ),
        )
        self.play(FadeIn(next_gaussian), FadeIn(next_distribution), run_time=1.8)
        self._pause(1.5)

        next_sample_one = Dot((-0.92, 0.44, 0), radius=0.082, color=DRAFT)
        next_tick_one = Line(
            (-0.92, 0.33, 0), (-0.92, 0.55, 0), color=DRAFT, stroke_width=3.0
        )
        self.play(FadeIn(next_sample_one, scale=1.8), Create(next_tick_one), run_time=1.0)
        self.play(Indicate(next_sample_one, color=DRAFT, scale_factor=1.9), run_time=1.0)
        self.play(
            Create(next_edges[0]),
            ReplacementTransform(next_sample_one, next_children[0]),
            FadeOut(next_tick_one),
            run_time=1.9,
        )

        next_sample_two = Dot((0.61, 0.44, 0), radius=0.082, color=DRAFT)
        next_tick_two = Line(
            (0.61, 0.33, 0), (0.61, 0.55, 0), color=DRAFT, stroke_width=3.0
        )
        self.play(FadeIn(next_sample_two, scale=1.8), Create(next_tick_two), run_time=1.0)
        self.play(Indicate(next_sample_two, color=DRAFT, scale_factor=1.9), run_time=1.0)
        self.play(
            Create(next_edges[1]),
            ReplacementTransform(next_sample_two, next_children[1]),
            FadeOut(next_tick_two),
            run_time=1.9,
        )
        self._pause(2.0)

        next_round_screen = VGroup(
            next_root,
            prefetched_mean,
            next_gaussian,
            next_distribution,
            next_children,
            next_edges,
        )
        # ----------------------------- evaluate_leaves=True: full target batch
        full_header = self._header(
            "FULL-TREE TARGET EVALUATION  ·  evaluate_leaves=True"
        )
        full_phases = self._phase_labels()
        for i, phase in enumerate(full_phases):
            phase.set_opacity(1.0 if i == 1 else 0.24)
        self.play(
            ReplacementTransform(next_round_header, full_header),
            FadeOut(next_round_screen),
            FadeIn(full_phases),
            run_time=1.5,
        )

        full_tree = self._tree_geometry()
        self.play(
            AnimationGroup(*(Create(edge) for edge in full_tree["edges"]), lag_ratio=0.04),
            AnimationGroup(*(GrowFromCenter(node) for node in full_tree["nodes"]), lag_ratio=0.04),
            run_time=1.8,
        )
        # Match the earlier verification grammar: first compress and move the
        # tree left, then mark every evaluated node, then assemble one batch.
        full_verification_tree = self._tree_geometry(x_scale=0.78, x_shift=-3.45)
        self.play(
            ReplacementTransform(
                full_tree["edges"], full_verification_tree["edges"]
            ),
            *(
                node.animate.move_to(target)
                for node, target in zip(
                    full_tree["nodes"], full_verification_tree["nodes"]
                )
            ),
            run_time=1.8,
        )
        full_tree["edges"] = full_verification_tree["edges"]

        internal_full_halos = VGroup(
            *(self._mean_halo(node) for node in full_tree["nodes"][:3])
        )
        leaf_full_halos = VGroup(
            *(
                DashedVMobject(
                    Circle(
                        radius=node.width / 2 + 0.19,
                        stroke_color=VERIFY,
                        stroke_width=3.4,
                    ).move_to(node),
                    num_dashes=14,
                )
                for node in full_tree["nodes"][3:]
            )
        )
        for halo in leaf_full_halos:
            halo.set_z_index(12)
        full_halos = VGroup(*internal_full_halos, *leaf_full_halos)
        self.play(
            AnimationGroup(
                *(Create(halo) for halo in internal_full_halos),
                lag_ratio=0.12,
            ),
            AnimationGroup(
                *(Create(halo) for halo in leaf_full_halos),
                lag_ratio=0.12,
            ),
            run_time=2.2,
        )
        self.add(*internal_full_halos, *leaf_full_halos)

        full_batch_nodes = VGroup(
            *(node.copy().scale(0.52) for node in full_tree["nodes"])
        )
        for node in full_batch_nodes:
            node[0].set_stroke(width=1.55)
        full_batch_nodes.arrange_in_grid(rows=4, cols=2, buff=(0.14, 0.11))
        full_batch_nodes.move_to((0.08, 0.08, 0))
        full_batch = VGroup(
            Text("ONE BATCH", font_size=22, weight="BOLD", color=VERIFY),
            Text("all 7 nodes", font_size=18, color=MUTED),
        ).arrange(DOWN, buff=0.04)
        full_batch.next_to(full_batch_nodes, UP, buff=0.14)
        self.play(
            AnimationGroup(
                *(
                    TransformFromCopy(source, target)
                    for source, target in zip(
                        full_tree["nodes"], full_batch_nodes
                    )
                ),
                lag_ratio=0.10,
            ),
            FadeIn(full_batch, shift=0.08 * UP),
            run_time=2.2,
        )

        full_target = self._target_model(VERIFY, width=1.90, height=3.10)
        full_target.move_to((3.18, 0.08, 0))
        full_batch_arrow = Arrow(
            (full_batch_nodes.get_right()[0] + 0.10, 0.08, 0),
            (full_target.get_left()[0] - 0.10, 0.08, 0),
            buff=0.18,
            stroke_width=3.4,
            max_tip_length_to_length_ratio=0.08,
            color=VERIFY,
        )
        self.play(
            FadeIn(full_target),
            GrowArrow(full_batch_arrow),
            run_time=1.7,
        )
        self._pause(3.0)

        # Phase 3 repeats the original acceptance path: first resolve depth 1,
        # colour the accepted branch, and only then move VERIFY to the leaves.
        self.play(
            self._activate_phase(full_phases, 2),
            FadeOut(full_target),
            FadeOut(full_batch),
            FadeOut(full_batch_nodes),
            FadeOut(full_batch_arrow),
            run_time=1.5,
        )
        full_node_verify = self._verify_box(width=1.52, height=0.70, font_size=18)
        full_node_verify.move_to((0.72, 0.02, 0))
        full_depth_one_label = Text(
            "DEPTH 1", font_size=20, weight="BOLD", color=MUTED
        )
        full_depth_one_label.next_to(full_node_verify, UP, buff=0.15)
        full_level_one_pair = VGroup(
            full_tree["nodes"][1].copy(),
            full_tree["nodes"][2].copy(),
        ).scale(0.52)
        full_level_one_pair.arrange(RIGHT, buff=0.08).move_to(full_node_verify)
        self.play(
            FadeIn(full_node_verify),
            FadeIn(full_depth_one_label),
            TransformFromCopy(
                VGroup(full_tree["nodes"][1], full_tree["nodes"][2]),
                full_level_one_pair,
            ),
            run_time=1.1,
        )
        self.play(
            full_node_verify.animate.stretch(0.52, 0),
            full_level_one_pair.animate.stretch(0.24, 0),
            run_time=0.55,
        )
        full_depth_one_verdicts = VGroup(
            Text("REJECT", font_size=17, weight="BOLD", color=RESIDUAL),
            Text("ACCEPT", font_size=17, weight="BOLD", color=ACCEPT),
        )
        full_depth_one_verdicts[0].next_to(
            full_tree["nodes"][1], DOWN, buff=0.10
        )
        full_depth_one_verdicts[1].next_to(
            full_tree["nodes"][2], DOWN, buff=0.10
        )
        self.play(
            full_node_verify.animate.stretch(1 / 0.52, 0),
            FadeOut(full_level_one_pair),
            FadeIn(full_depth_one_verdicts, shift=0.08 * UP),
            full_tree["nodes"][1][0].animate.set_stroke(
                RESIDUAL, width=4
            ).set_fill(RESIDUAL, opacity=0.22),
            full_tree["edges"][1].animate.set_color(ACCEPT).set_stroke(width=5),
            full_tree["nodes"][2][0].animate.set_stroke(
                ACCEPT, width=4
            ).set_fill(ACCEPT, opacity=0.22),
            run_time=0.85,
        )
        self._pause(1.6)

        full_depth_position = np.array([0.72, -1.72, 0.0])
        full_depth_label = Text(
            "DEPTH 2", font_size=20, weight="BOLD", color=MUTED
        )
        full_depth_label.move_to(full_depth_position + 0.58 * UP)
        self.play(
            FadeOut(full_depth_one_label),
            FadeOut(full_depth_one_verdicts),
            full_node_verify.animate.move_to(full_depth_position),
            FadeIn(full_depth_label),
            run_time=0.9,
        )
        full_level_pair = VGroup(
            full_tree["nodes"][5].copy(),
            full_tree["nodes"][6].copy(),
        ).scale(0.52)
        full_level_pair.arrange(RIGHT, buff=0.08).move_to(full_node_verify)
        self.play(
            TransformFromCopy(
                VGroup(full_tree["nodes"][5], full_tree["nodes"][6]),
                full_level_pair,
            ),
            run_time=1.1,
        )
        self.play(
            full_node_verify.animate.stretch(0.52, 0),
            full_level_pair.animate.stretch(0.24, 0),
            run_time=0.55,
        )
        self.play(
            full_node_verify.animate.stretch(1 / 0.52, 0),
            FadeOut(full_level_pair),
            run_time=0.65,
        )

        full_case_a_panel = RoundedRectangle(
            width=2.20,
            height=2.25,
            corner_radius=0.12,
            fill_color=ACCEPT,
            fill_opacity=0.045,
            stroke_color=ACCEPT,
            stroke_width=1.8,
            stroke_opacity=0.55,
        ).move_to((3.05, -0.25, 0))
        full_case_b_panel = RoundedRectangle(
            width=2.20,
            height=2.25,
            corner_radius=0.12,
            fill_color=RESIDUAL,
            fill_opacity=0.045,
            stroke_color=RESIDUAL,
            stroke_width=1.8,
            stroke_opacity=0.55,
        ).move_to((5.55, -0.25, 0))
        full_case_a_header = VGroup(
            Text("CASE A", font_size=21, weight="BOLD", color=ACCEPT),
            Text("accepted leaf", font_size=18, color=MUTED),
        ).arrange(DOWN, buff=0.05).move_to((3.05, 0.50, 0))
        full_case_b_header = VGroup(
            Text("CASE B", font_size=21, weight="BOLD", color=RESIDUAL),
            Text("both leaves rejected", font_size=16, color=MUTED),
        ).arrange(DOWN, buff=0.05).move_to((5.55, 0.50, 0))
        full_case_a_outcome = self._state_node(r"y_{n+2}", ACCEPT, radius=0.30)
        full_case_a_outcome.move_to((3.05, -0.55, 0))
        full_case_b_outcome = self._state_node(r"y_{n+2}", RESIDUAL, radius=0.30)
        full_case_b_outcome.move_to((5.55, -0.55, 0))
        full_case_a_note = Text("own drift available", font_size=17, color=ACCEPT)
        full_case_a_note.next_to(full_case_a_outcome, DOWN, buff=0.12)
        full_case_b_note = Text("nearest at depth 2", font_size=17, color=RESIDUAL)
        full_case_b_note.next_to(full_case_b_outcome, DOWN, buff=0.12)
        self.play(
            FadeIn(full_case_a_panel),
            FadeIn(full_case_b_panel),
            FadeIn(full_case_a_header, shift=0.08 * DOWN),
            FadeIn(full_case_b_header, shift=0.08 * DOWN),
            TransformFromCopy(full_tree["nodes"][6], full_case_a_outcome),
            TransformFromCopy(full_tree["nodes"][6], full_case_b_outcome),
            FadeIn(full_case_a_note),
            FadeIn(full_case_b_note),
            run_time=1.6,
        )
        full_case_table = VGroup(
            full_case_a_panel,
            full_case_b_panel,
            full_case_a_header,
            full_case_b_header,
            full_case_a_outcome,
            full_case_b_outcome,
            full_case_a_note,
            full_case_b_note,
        )
        self._pause(1.5)
        full_case_a_drift = MathTex(
            r"b^q_{t_{n+2}}(y_{n+2})",
            font_size=23,
            color=ACCEPT,
        ).move_to((3.05, -1.78, 0))
        full_case_b_drift = MathTex(
            r"b^q_{t_{n+2}}(y_{n+2})",
            font_size=23,
            color=MUTED,
        ).move_to((5.55, -1.78, 0))
        full_case_b_cross = self._cross(
            full_case_b_drift.get_center(),
            size=0.25,
            color=DANGER,
        )
        full_case_consequences = VGroup(
            full_case_a_drift,
            full_case_b_drift,
            full_case_b_cross,
        )
        self.play(
            FadeIn(full_case_a_drift, shift=0.08 * UP),
            FadeIn(full_case_b_drift, shift=0.08 * UP),
            FadeIn(full_case_b_cross, scale=1.20),
            run_time=1.2,
        )
        self._pause(2.5)

        # If the final leaf transition rejects, the residual is not a node in
        # the batch. All leaf siblings were nevertheless evaluated, so nearest
        # chooses one at the correct time index instead of the parent.
        leaf_reject_header = self._header(
            "LEAF REJECTION  ·  NEAREST AT THE SAME DENOISING TIME"
        )
        self.play(
            ReplacementTransform(full_header, leaf_reject_header),
            FadeOut(full_node_verify),
            FadeOut(full_depth_label),
            FadeOut(full_case_table),
            FadeOut(full_case_consequences),
            run_time=1.4,
        )
        # Restore the tree's visibility and repeat the graphical nearest-node
        # grammar from the earlier prefetch slide, now with depth-two leaves.
        self.play(
            full_tree["edges"].animate.set_opacity(1.0).set_color(FAINT).set_stroke(width=2.6),
            full_halos.animate.set_opacity(1.0),
            full_tree["nodes"][0][0].animate.set_stroke(ACCEPT, width=2.7, opacity=1.0).set_fill(ACCEPT, opacity=0.12),
            full_tree["nodes"][0][1].animate.set_opacity(1.0),
            AnimationGroup(
                *(
                    AnimationGroup(
                        node[0].animate.set_stroke(DRAFT, width=2.7, opacity=1.0).set_fill(DRAFT, opacity=0.12),
                        node[1].animate.set_opacity(1.0),
                        lag_ratio=0,
                    )
                    for node in full_tree["nodes"][1:]
                ),
                lag_ratio=0,
            ),
            run_time=1.2,
        )
        leaf_residual = self._state_node(r"y_{n+2}", RESIDUAL, radius=0.31)
        leaf_residual.move_to((3.55, 1.33, 0))
        leaf_residual_label = Text("residual output", font_size=21, color=RESIDUAL)
        leaf_residual_label.next_to(leaf_residual, DOWN, buff=0.14)
        self.play(
            GrowFromCenter(leaf_residual),
            FadeIn(leaf_residual_label),
            AnimationGroup(
                self._pulse(full_tree["nodes"][5], VERIFY),
                self._pulse(full_tree["nodes"][6], VERIFY),
                lag_ratio=0.18,
            ),
            run_time=1.6,
        )

        leaf_formula_head = MathTex(
            r"v^\star\in\arg\min",
            font_size=34,
            color=INK,
        )
        leaf_placeholder_one = DashedVMobject(
            Circle(radius=0.27, stroke_color=FAINT, stroke_width=2.0),
            num_dashes=10,
        )
        leaf_placeholder_two = leaf_placeholder_one.copy()
        leaf_first_left = MathTex(r"\lVert y_{n+2}-", font_size=29, color=INK)
        leaf_first_right = MathTex(r"\rVert,", font_size=29, color=INK)
        leaf_second_left = MathTex(r"\lVert y_{n+2}-", font_size=29, color=INK)
        leaf_second_right = MathTex(r"\rVert", font_size=29, color=INK)
        leaf_set_left = MathTex(r"\{", font_size=42, color=INK)
        leaf_set_right = MathTex(r"\}", font_size=42, color=INK)
        leaf_formula_shell = VGroup(
            leaf_formula_head,
            leaf_set_left,
            leaf_first_left,
            leaf_placeholder_one,
            leaf_first_right,
            leaf_second_left,
            leaf_placeholder_two,
            leaf_second_right,
            leaf_set_right,
        ).arrange(RIGHT, buff=0.06)
        leaf_formula_shell.scale_to_fit_width(6.45)
        leaf_formula_shell.move_to((3.45, -0.15, 0))
        leaf_formula_visible = VGroup(
            leaf_formula_head,
            leaf_set_left,
            leaf_set_right,
            leaf_first_left,
            leaf_first_right,
            leaf_second_left,
            leaf_second_right,
        )
        leaf_formula_node_one = full_tree["nodes"][5].copy()
        leaf_formula_node_two = full_tree["nodes"][6].copy()
        leaf_formula_node_one.scale_to_fit_height(leaf_placeholder_one.height * 1.02)
        leaf_formula_node_two.scale_to_fit_height(leaf_placeholder_two.height * 1.02)
        leaf_formula_node_one.move_to(leaf_placeholder_one)
        leaf_formula_node_two.move_to(leaf_placeholder_two)
        self.play(
            FadeIn(leaf_formula_visible, shift=0.08 * UP),
            run_time=1.2,
        )
        self.play(
            TransformFromCopy(full_tree["nodes"][5], leaf_formula_node_one),
            TransformFromCopy(full_tree["nodes"][6], leaf_formula_node_two),
            run_time=1.8,
        )
        self.play(
            leaf_first_left.animate.set_opacity(0.28),
            leaf_first_right.animate.set_opacity(0.28),
            leaf_formula_node_one.animate.set_opacity(0.28),
            Indicate(leaf_formula_node_two, color=VERIFY, scale_factor=1.20),
            run_time=1.0,
        )
        leaf_tilde_prefix = MathTex(
            r"\widetilde y_{n+2}=",
            font_size=36,
            color=VERIFY,
        )
        leaf_tilde_node = leaf_formula_node_two.copy().scale(1.35)
        leaf_tilde_group = VGroup(
            leaf_tilde_prefix, leaf_tilde_node
        ).arrange(RIGHT, buff=0.14)
        leaf_tilde_group.move_to((3.55, -2.10, 0))
        leaf_tilde_caption = VGroup(
            Text(
                "nearest evaluated leaf at depth 2  ·  same time",
                font_size=20,
                color=MUTED,
            ),
            MathTex(r"t_{n+2}", font_size=24, color=VERIFY),
        ).arrange(RIGHT, buff=0.10)
        leaf_tilde_caption.next_to(leaf_tilde_group, DOWN, buff=0.12)
        self.play(
            FadeIn(leaf_tilde_prefix, shift=0.10 * UP),
            TransformFromCopy(leaf_formula_node_two, leaf_tilde_node),
            FadeIn(leaf_tilde_caption, shift=0.08 * UP),
            run_time=1.3,
        )
        self._pause(3.2)

    # ---------------------------------------------------------------- layout
    @staticmethod
    def _header(text: str) -> ManimText:
        header = Text(text, font_size=42, weight="BOLD", color=INK)
        if header.width > 12.0:
            header.scale_to_fit_width(12.0)
        header.to_edge(UP, buff=0.25)
        return header

    def _phase_labels(self) -> VGroup:
        labels = VGroup(
            self._phase_label("1  DRAFT", DRAFT),
            self._phase_label("2  VERIFY", VERIFY),
            self._phase_label("3  ACCEPT", ACCEPT),
        ).arrange(RIGHT, buff=0.85)
        labels.move_to((0.0, 2.75, 0))
        for label in labels:
            label.set_opacity(0.32)
        return labels

    @staticmethod
    def _phase_label(text: str, color: str) -> VGroup:
        word = Text(text, font_size=23, weight="BOLD", color=color)
        underline = Line(LEFT, RIGHT, color=color, stroke_width=3).match_width(word)
        underline.next_to(word, DOWN, buff=0.08)
        return VGroup(word, underline)

    @staticmethod
    def _activate_phase(phases: VGroup, index: int) -> AnimationGroup:
        return AnimationGroup(
            *[
                phase.animate.set_opacity(1.0 if i == index else 0.24)
                for i, phase in enumerate(phases)
            ],
            lag_ratio=0,
        )

    def _tree_geometry(
        self,
        x_scale: float = 1.0,
        x_shift: float = 0.0,
    ) -> dict[str, VGroup]:
        base_positions = [
            (0.0, 1.55, 0),
            (-2.05, 0.02, 0),
            (2.05, 0.02, 0),
            (-3.10, -1.72, 0),
            (-1.02, -1.72, 0),
            (1.02, -1.72, 0),
            (3.10, -1.72, 0),
        ]
        positions = [
            (x_scale * x + x_shift, y, z)
            for x, y, z in base_positions
        ]
        labels = [
            "y_n",
            r"y^{(1)}_{n+1}",
            r"y^{(2)}_{n+1}",
            r"y^{(1,1)}_{n+2}",
            r"y^{(1,2)}_{n+2}",
            r"y^{(2,1)}_{n+2}",
            r"y^{(2,2)}_{n+2}",
        ]
        nodes = VGroup(
            self._state_node(labels[0], ACCEPT, radius=0.36),
            *(self._state_node(label, DRAFT, radius=0.31) for label in labels[1:3]),
            *(self._state_node(label, DRAFT, radius=0.29) for label in labels[3:]),
        )
        for node, position in zip(nodes, positions):
            node.move_to(position)

        pairs = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        edges = VGroup(
            *[
                Arrow(
                    nodes[parent].get_bottom(),
                    nodes[child].get_top(),
                    buff=0.09,
                    stroke_width=2.6,
                    max_tip_length_to_length_ratio=0.10,
                    color=FAINT,
                )
                for parent, child in pairs
            ]
        )
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _state_node(label: str, color: str, radius: float) -> VGroup:
        circle = Circle(radius=radius, stroke_color=color, stroke_width=2.7)
        circle.set_fill(color, opacity=0.12)
        text = MathTex(label, font_size=22 if radius < 0.31 else 25, color=INK)
        text.move_to(circle)
        return VGroup(circle, text)

    @staticmethod
    def _gaussian_packet(
        center_x: float,
        baseline_y: float,
        width: float,
        height: float,
        sigma: float,
        show_mean: bool = False,
        mean_tex: str = "m",
        mean_font_size: int = 24,
    ) -> VGroup:
        curve = ParametricFunction(
            lambda t: np.array(
                [
                    center_x + t,
                    baseline_y + height * np.exp(-0.5 * (t / sigma) ** 2),
                    0.0,
                ]
            ),
            t_range=[-width / 2, width / 2, 0.025],
            stroke_color=DRAFT,
            stroke_width=3.0,
        )
        baseline = Line(
            (center_x - width / 2, baseline_y, 0),
            (center_x + width / 2, baseline_y, 0),
            color=DRAFT,
            stroke_width=1.6,
            stroke_opacity=0.50,
        )
        packet = VGroup(baseline, curve)
        if show_mean:
            mean_line = DashedLine(
                (center_x, baseline_y - 0.02, 0),
                (center_x, baseline_y + height + 0.04, 0),
                color=DRAFT,
                stroke_width=2.0,
                dash_length=0.08,
            )
            mean_label = MathTex(mean_tex, font_size=mean_font_size, color=DRAFT)
            mean_label.move_to((center_x, baseline_y - 0.20, 0))
            packet.add(mean_line, mean_label)
        return packet

    @staticmethod
    def _gaussian_point(
        center_x: float,
        baseline_y: float,
        height: float,
        sigma: float,
        offset: float,
    ) -> np.ndarray:
        return np.array(
            [
                center_x + offset,
                baseline_y + height * np.exp(-0.5 * (offset / sigma) ** 2),
                0.0,
            ]
        )

    @staticmethod
    def _target_model(
        color: str,
        width: float = 1.75,
        height: float = 0.82,
    ) -> VGroup:
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.15,
            fill_color=color,
            fill_opacity=0.10,
            stroke_color=color,
            stroke_width=2.3,
        )
        label = VGroup(
            Text("TARGET", font_size=19, weight="BOLD", color=color),
            MathTex(r"m^q(\cdot)", font_size=24, color=INK),
        ).arrange(DOWN, buff=0.04)
        label.move_to(box)
        return VGroup(box, label)

    @staticmethod
    def _verify_box(width: float, height: float, font_size: int) -> VGroup:
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.15,
            fill_color=VERIFY,
            fill_opacity=0.10,
            stroke_color=VERIFY,
            stroke_width=2.4,
        )
        label = Text("VERIFY", font_size=font_size, weight="BOLD", color=VERIFY)
        label.move_to(box)
        return VGroup(box, label)

    @staticmethod
    def _mean_card(tex: str, color: str, font_size: int) -> VGroup:
        """Compact color-coded container for one proposal or target mean."""
        label = MathTex(tex, font_size=font_size, color=color)
        box = RoundedRectangle(
            width=max(1.45, label.width + 0.34),
            height=max(0.52, label.height + 0.18),
            corner_radius=0.10,
            fill_color=color,
            fill_opacity=0.12,
            stroke_color=color,
            stroke_width=2.0,
        )
        label.move_to(box)
        return VGroup(box, label)

    @staticmethod
    def _mean_halo(node: VGroup) -> DashedVMobject:
        """Purple halo: the target drift at this node is in the batch."""
        ring = Circle(
            radius=node.width / 2 + 0.13,
            stroke_color=VERIFY,
            stroke_width=2.8,
        ).move_to(node)
        return DashedVMobject(ring, num_dashes=14)

    @staticmethod
    def _pulse(node: VGroup, color: str) -> AnimationGroup:
        return AnimationGroup(
            Indicate(node[0], color=color, scale_factor=1.28),
            Indicate(node[1], color=color, scale_factor=1.08),
            lag_ratio=0,
        )

    @staticmethod
    def _cross(center, size: float = 0.18, color: str = RESIDUAL) -> VGroup:
        one = Line(
            center + size * (UP + LEFT),
            center + size * (DOWN + RIGHT),
            color=color,
            stroke_width=4,
        )
        two = Line(
            center + size * (UP + RIGHT),
            center + size * (DOWN + LEFT),
            color=color,
            stroke_width=4,
        )
        return VGroup(one, two)

    def _pause(self, seconds: float = 1.5) -> None:
        """Hold the frame in video mode, then create an interactive slide stop."""
        self.wait(seconds)
        if hasattr(self, "next_slide"):
            self.next_slide()
