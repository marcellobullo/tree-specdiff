"""Algorithm 1 drafting followed by target-informed Picard refinement.

Render a continuous 1080p video:
    conda run -n specdiff manim -qh \
        slides/algorithm_one_with_picard_iters.py AlgorithmOneWithPicardIters

The drafting sequence is inherited verbatim from ``algorithm_one.py`` and is
stopped exactly where its VERIFY phase would begin.  This keeps the two scenes
visually synchronized while the Picard-specific continuation lives here.
"""

from __future__ import annotations

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    Line,
    MathTex,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Transform,
    TransformFromCopy,
    UP,
    VGroup,
)

try:  # Support both ``manim slides/file.py`` and package-style imports.
    from .algorithm_one import (
        ACCEPT,
        DRAFT,
        FAINT,
        INK,
        MUTED,
        VERIFY,
        AlgorithmOneSpeculativeTree,
        Text,
    )
except ImportError:
    from algorithm_one import (
        ACCEPT,
        DRAFT,
        FAINT,
        INK,
        MUTED,
        VERIFY,
        AlgorithmOneSpeculativeTree,
        Text,
    )


class _DraftComplete(RuntimeError):
    """Internal control-flow marker raised before VERIFY begins."""


class AlgorithmOneWithPicardIters(AlgorithmOneSpeculativeTree):
    """Reuse Algorithm 1 drafting, then explain J Picard refinements."""

    def construct(self) -> None:
        self._draft_tree = None
        self._draft_header = None
        self._draft_phases = None

        # Run precisely the shared DRAFT portion.  The overridden phase switch
        # below stops the parent scene before any VERIFY animation is played.
        try:
            super().construct()
        except _DraftComplete:
            pass

        if self._draft_tree is None or self._draft_header is None:
            raise RuntimeError("Could not capture the shared drafting scene")

        self._construct_picard_refinement(
            self._draft_header,
            self._draft_phases,
            self._draft_tree,
        )

    # ------------------------------------------------ shared-scene interception
    def _tree_geometry(self, *args, **kwargs):
        geometry = super()._tree_geometry(*args, **kwargs)
        if self._draft_tree is None:
            self._draft_tree = geometry
        return geometry

    def _header(self, text: str):
        header = super()._header(text)
        if self._draft_header is None:
            self._draft_header = header
        return header

    def _phase_labels(self):
        phases = super()._phase_labels()
        if self._draft_phases is None:
            self._draft_phases = phases
        return phases

    def _activate_phase(self, phases, index: int):
        if index == 1:
            raise _DraftComplete
        return super()._activate_phase(phases, index)

    # --------------------------------------------------------- Picard narrative
    def _construct_picard_refinement(self, header, phases, tree) -> None:
        self._pause(1.4)

        picard_header = self._header(
            "PICARD ITERATIONS  ·  TRADE NFE FOR TARGET-INFORMED DRAFTS"
        )
        tradeoff = VGroup(
            MathTex(r"J\ \text{extra batched target evaluations}", font_size=27),
            MathTex(r"\Longrightarrow", font_size=34, color=VERIFY),
            Text("drafts closer to the target dynamics", font_size=23, color=INK),
        ).arrange(RIGHT, buff=0.18)
        tradeoff.move_to((0.0, 2.72, 0))
        self.play(
            ReplacementTransform(header, picard_header),
            FadeOut(phases),
            FadeIn(tradeoff, shift=0.10 * DOWN),
            run_time=1.6,
        )

        # Match the established VERIFY layout: compressed tree at far left,
        # two-column batch in the middle, target model on the right.
        compressed = super()._tree_geometry(x_scale=0.78, x_shift=-3.45)
        self.play(
            ReplacementTransform(tree["edges"], compressed["edges"]),
            *(
                node.animate.move_to(target)
                for node, target in zip(tree["nodes"], compressed["nodes"])
            ),
            run_time=1.8,
        )
        tree["edges"] = compressed["edges"]

        mismatch_title = Text(
            "draft–target mismatch",
            font_size=19,
            color=MUTED,
        ).move_to((5.15, -2.48, 0))
        mismatch_track = Line(
            (4.25, -2.88, 0),
            (6.05, -2.88, 0),
            color=FAINT,
            stroke_width=9,
        )
        mismatch_bar = Line(
            mismatch_track.get_start(),
            mismatch_track.get_end(),
            color=DRAFT,
            stroke_width=9,
        )
        mismatch_label = MathTex(
            r"\lVert y^{[0]}-m^q\rVert",
            font_size=22,
            color=DRAFT,
        ).next_to(mismatch_track, DOWN, buff=0.10)
        self.play(
            FadeIn(mismatch_title),
            Create(mismatch_track),
            Create(mismatch_bar),
            FadeIn(mismatch_label),
            run_time=1.0,
        )

        iteration_label = None
        J = 3
        gap_fractions = (0.66, 0.40, 0.18)
        for j in range(1, J + 1):
            iteration_next = VGroup(
                Text(
                    f"PICARD ITERATION  {j}/{J}",
                    font_size=24,
                    weight="BOLD",
                    color=VERIFY,
                ),
                MathTex(rf"y^{{[{j - 1}]}}\mapsto y^{{[{j}]}}", font_size=24, color=MUTED),
            ).arrange(DOWN, buff=0.04)
            iteration_next.move_to((0.0, 2.10, 0))
            if iteration_label is None:
                self.play(FadeIn(iteration_next, shift=0.08 * DOWN), run_time=0.8)
            else:
                self.play(
                    ReplacementTransform(iteration_label, iteration_next),
                    run_time=0.7,
                )
            iteration_label = iteration_next

            batch_nodes = VGroup(
                *(node.copy().scale(0.50) for node in tree["nodes"])
            )
            for node in batch_nodes:
                node[0].set_stroke(width=1.45)
            batch_nodes.arrange_in_grid(rows=4, cols=2, buff=(0.13, 0.10))
            batch_nodes.move_to((0.05, -0.12, 0))
            batch_label = VGroup(
                Text("ONE BATCH", font_size=20, weight="BOLD", color=VERIFY),
                MathTex(rf"\{{y_v^{{[{j - 1}]}}\}}_{{v\in T_n}}", font_size=18, color=MUTED),
            ).arrange(DOWN, buff=0.04)
            batch_label.next_to(batch_nodes, UP, buff=0.12)
            self.play(
                AnimationGroup(
                    *(
                        TransformFromCopy(source, target)
                        for source, target in zip(tree["nodes"], batch_nodes)
                    ),
                    lag_ratio=0.09,
                ),
                FadeIn(batch_label, shift=0.08 * UP),
                run_time=1.8,
            )

            target_model = self._target_model(VERIFY, width=1.82, height=3.02)
            target_model.move_to((2.95, -0.10, 0))
            batch_arrow = Arrow(
                (batch_nodes.get_right()[0] + 0.10, -0.10, 0),
                (target_model.get_left()[0] - 0.10, -0.10, 0),
                buff=0.16,
                stroke_width=3.2,
                max_tip_length_to_length_ratio=0.08,
                color=VERIFY,
            )
            self.play(FadeIn(target_model), GrowArrow(batch_arrow), run_time=1.2)

            drift_cards = VGroup(
                *(
                    self._picard_drift_card(
                        rf"b^q(y_{{v_{i}}}^{{[{j - 1}]}})"
                    )
                    for i in range(7)
                )
            )
            drift_cards.arrange_in_grid(rows=4, cols=2, buff=(0.10, 0.09))
            drift_cards.move_to((5.25, -0.12, 0))
            drift_label = Text(
                "TARGET DRIFTS",
                font_size=18,
                weight="BOLD",
                color=VERIFY,
            )
            drift_label.next_to(drift_cards, UP, buff=0.12)
            row_centres = [
                VGroup(*drift_cards[2 * row : min(2 * row + 2, 7)]).get_center()[1]
                for row in range(4)
            ]
            output_start_x = target_model.get_right()[0] + 0.10
            output_end_x = drift_cards.get_left()[0] - 0.12
            output_arrows = VGroup(
                *(
                    Arrow(
                        (output_start_x, y, 0),
                        (output_end_x, y, 0),
                        buff=0,
                        stroke_width=2.4,
                        max_tip_length_to_length_ratio=0.11,
                        color=VERIFY,
                    )
                    for y in row_centres
                )
            )
            self.play(FadeIn(drift_label, shift=0.06 * DOWN), run_time=0.5)
            for row, arrow in enumerate(output_arrows):
                row_cards = VGroup(*drift_cards[2 * row : min(2 * row + 2, 7)])
                self.play(
                    Create(arrow),
                    FadeIn(row_cards, shift=0.08 * RIGHT),
                    run_time=0.55,
                )
            self._pause(0.8)

            update_box = self._picard_update_box(j)
            update_box.move_to((2.95, -0.10, 0))
            self.play(
                FadeOut(target_model),
                FadeOut(batch_arrow),
                FadeOut(output_arrows),
                FadeIn(update_box, scale=0.92),
                run_time=0.8,
            )
            self.play(
                batch_nodes.animate.scale(0.18).move_to(update_box),
                drift_cards.animate.scale(0.18).move_to(update_box),
                update_box.animate.stretch(0.78, 0),
                AnimationGroup(
                    *(
                        AnimationGroup(
                            node[0].animate.stretch(0.72, 0),
                            node[1].animate.stretch(0.72, 0),
                            lag_ratio=0,
                        )
                        for node in tree["nodes"]
                    ),
                    lag_ratio=0,
                ),
                FadeOut(batch_label),
                FadeOut(drift_label),
                run_time=1.1,
            )

            refined_labels = VGroup(
                *(
                    self._picard_tree_label(node, tex, j)
                    for node, tex in zip(
                        tree["nodes"],
                        (
                            r"y_n^{[%d]}" % j,
                            r"y_{n+1}^{(1)\,[%d]}" % j,
                            r"y_{n+1}^{(2)\,[%d]}" % j,
                            r"y_{n+2}^{(1,1)\,[%d]}" % j,
                            r"y_{n+2}^{(1,2)\,[%d]}" % j,
                            r"y_{n+2}^{(2,1)\,[%d]}" % j,
                            r"y_{n+2}^{(2,2)\,[%d]}" % j,
                        ),
                    )
                )
            )
            self.play(
                update_box.animate.stretch(1 / 0.78, 0),
                *(
                    node[0].animate.stretch(1 / 0.72, 0)
                    for node in tree["nodes"]
                ),
                *(
                    Transform(node[1], label)
                    for node, label in zip(tree["nodes"], refined_labels)
                ),
                run_time=0.9,
            )

            new_gap_end = mismatch_track.get_start() + gap_fractions[j - 1] * (
                mismatch_track.get_end() - mismatch_track.get_start()
            )
            mismatch_bar_next = Line(
                mismatch_track.get_start(),
                new_gap_end,
                color=VERIFY if j == J else DRAFT,
                stroke_width=9,
            )
            mismatch_label_next = MathTex(
                rf"\lVert y^{{[{j}]}}-m^q\rVert",
                font_size=22,
                color=VERIFY if j == J else DRAFT,
            ).next_to(mismatch_track, DOWN, buff=0.10)
            self.play(
                FadeOut(batch_nodes),
                FadeOut(drift_cards),
                FadeOut(update_box),
                ReplacementTransform(mismatch_bar, mismatch_bar_next),
                ReplacementTransform(mismatch_label, mismatch_label_next),
                run_time=1.2,
            )
            mismatch_bar = mismatch_bar_next
            mismatch_label = mismatch_label_next
            self._pause(1.0)

        conclusion_top = VGroup(
            MathTex(r"J=3", font_size=34, color=VERIFY),
            Text("additional batched target passes", font_size=23, color=MUTED),
        ).arrange(RIGHT, buff=0.14)
        conclusion = VGroup(
            conclusion_top,
            MathTex(r"\Downarrow", font_size=32, color=VERIFY),
            Text("target-informed proposal samples", font_size=25, color=INK),
        ).arrange(DOWN, buff=0.10)
        conclusion.move_to((3.25, 0.85, 0))
        self.play(FadeIn(conclusion, shift=0.10 * UP), run_time=1.2)
        self._pause(3.0)

    @staticmethod
    def _picard_drift_card(tex: str) -> VGroup:
        label = MathTex(tex, font_size=14, color=VERIFY)
        box = RoundedRectangle(
            width=max(1.02, label.width + 0.18),
            height=0.42,
            corner_radius=0.08,
            fill_color=VERIFY,
            fill_opacity=0.10,
            stroke_color=VERIFY,
            stroke_width=1.5,
        )
        label.move_to(box)
        return VGroup(box, label)

    @staticmethod
    def _picard_update_box(iteration: int) -> VGroup:
        box = RoundedRectangle(
            width=2.35,
            height=1.10,
            corner_radius=0.14,
            fill_color=VERIFY,
            fill_opacity=0.12,
            stroke_color=VERIFY,
            stroke_width=2.4,
        )
        label = VGroup(
            Text("PICARD UPDATE", font_size=18, weight="BOLD", color=VERIFY),
            MathTex(rf"j={iteration}", font_size=18, color=MUTED),
        ).arrange(DOWN, buff=0.04)
        label.move_to(box)
        return VGroup(box, label)

    @staticmethod
    def _picard_tree_label(node: VGroup, tex: str, iteration: int) -> MathTex:
        """Preserve the tree superscript and distinguish Picard steps by [j]."""
        label = MathTex(
            tex,
            font_size=22,
            color=INK,
        )
        max_width = 0.84 * node[0].width / 0.72
        max_height = 0.64 * node[0].height
        if label.width > max_width:
            label.scale(max_width / label.width)
        if label.height > max_height:
            label.scale(max_height / label.height)
        label.move_to(node[0])
        return label
