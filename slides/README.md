# SpecDiff Manim slides

The opening scene starts *in medias res* from one speculative round and animates
Algorithm 1 through its three phases, then continues across round boundaries to
compare root-drift prefetching policies. It uses a binary tree with lookahead two, matching
the paper's worked topology: six drafted states, three internal nodes, and one
batched target-model evaluation. All non-mathematical labels use Helvetica;
mathematical notation remains typeset with LaTeX.

Drafting is shown as Gaussian sampling rather than deterministic branching. For
each realised parent, the local proposal mean appears first, followed by a
one-dimensional bell-curve view centred at that mean. Two asymmetric sample
locations are then highlighted on the horizontal axis one at a time; each marker
becomes a lowercase realised child only as its outgoing edge is drawn. The same
sequence is repeated separately for both parents at the second depth. Capital
letters are reserved for random variables, while tree states use lowercase `y`.

## Render

```bash
conda activate specdiff
manim-slides render -qh slides/algorithm_one.py AlgorithmOneSpeculativeTree
```

For a fast preview, replace `-qh` with `-ql`. To play the interactive slide deck:

```bash
manim-slides present AlgorithmOneSpeculativeTree
```

The standard Manim CLI also produces a continuous MP4:

```bash
manim -qh slides/algorithm_one.py AlgorithmOneSpeculativeTree
```

## Picard-iteration variant

`algorithm_one_with_picard_iters.py` reuses the complete drafting sequence and
then replaces verification/acceptance with three target-informed Picard updates.
Each iteration batches all seven tree nodes through the target model, displays
their corresponding `b^q` values, and squeezes the node/drift pairs into one
fixed Picard update block while the schematic draft--target mismatch decreases.
The tree remains fixed to emphasize state refinement rather than graph motion.
During each squeeze, every tree node compresses and its label advances from
Picard state `[j-1]` to `[j]`. Square brackets distinguish this refinement index
from the unchanged parenthesized tree index, for example
`y_{n+1}^{(1)[j]}`.

```bash
conda activate specdiff
manim -qh slides/algorithm_one_with_picard_iters.py AlgorithmOneWithPicardIters
```

## Storyboard

1. Start at the last accepted state `Y_n`.
2. Place `m_n^p(y_n) = y_n + gamma b^q_{t_n}(y_tilde_n)` beside the root.
3. Reveal the Gaussian centred at `m_n^p(y_n)`, then highlight two asymmetric
   axis samples and draw the two children sequentially.
4. Repeat mean -> Gaussian -> axis sample -> edge separately for every parent at
   the second depth; the delayed drift remains frozen across the tree.
5. Move the tree left while temporarily narrowing its branch spacing, highlight
   its internal nodes, and copy them into a three-row batch entering one tall
   target-model rectangle through a horizontal arrow.
6. Emit three horizontally aligned `b^q` outputs through three identical arrows
   and place each inside its own
   explicit target-mean formula. Resize and translate each existing target-mean
   left-hand side directly into a purple card—without replacing or morphing the
   glyphs—and pair it with a blue proposal-mean card.
7. Copy all six non-root tree nodes into a fine-stroked miniature 2-by-3 drafted-children
   batch below `VERIFY`, replacing the abstract nested child-set notation. Pass
   that graphical batch and the scale parameters vertically into the block.
8. Keep the reduced-spacing tree fixed at the far left for acceptance. A compact
   `VERIFY` box aligns with each active depth beside it; the depth-one squeeze
   rejects the first child and accepts the second.
9. Align the depth-two `VERIFY` box with the leaf level, then reveal both outcome
   columns together. A simultaneous copy of the same verified leaf lands in
   Case A as an accepted draft and in Case B as the residual-result alternative.
10. Under the persistent table, show that both outcomes lack
   `b^q_{t_{n+2}}(y_{n+2})`, then introduce prefetching before changing slides.
11. On the prefetch slide, retain the exact compressed acceptance tree at left,
    including its orange rejection and green accepted branch, and highlight both
    level-one states. The single-line nearest-neighbour expression first shows
    empty, invisible slots; copies of the complete graphical nodes then land in
    the two candidate positions.
12. Select the nearer graphical candidate and copy that node downward to complete
    the graphical definition `y_tilde_{n+2} = [selected node]`.
13. Start the next round at `y_{n+2}`, reveal the prefetched proposal mean and its
    highlighted drift term, then draft two children using the same mean -> Gaussian
    -> axis sample -> edge sequence as the opening phase.
14. Enable `evaluate_leaves=True` under phase `2 VERIFY`: narrow and move the
    full tree left, mark all seven nodes with dashed target-evaluation circles,
    then copy them into a two-column batch that enters the target model through
    the same horizontal input grammar as the earlier verification phase.
15. Switch to phase `3 ACCEPT`, resolve depth one exactly as in the original
    round—reject the left child, accept the right child, and colour the accepted
    branch green—then move `VERIFY` to the depth-two leaves.
16. Reveal Case A and Case B together. Beneath Case A, show the available
    `b^q_{t_{n+2}}(y_{n+2})` in green; beneath Case B, show the same quantity in
    grey with a red cross because the output is a residual rather than a leaf.
17. Skip the separate full-acceptance explanation and move directly to leaf
    rejection, repeating the graphical nearest-neighbour animation using
    the two evaluated leaf siblings at depth two, then copy the selected leaf
    into the next-root drift definition.

The final state therefore illustrates both possible outputs of `Verify`: an accepted
drafted child and a residual sample.

## Source mapping

- Paper: Algorithm 1 and Section 3.1, especially the three-phase description.
- Paper: Figure 2 and the `K=2`, `L=2` draft-tree example.
- Paper: Appendix C, "Proposal Construction and Root-Drift Prefetching."
- Repository: `specdiff/sampler.py::_round`, whose three blocks mirror the animation.
- Repository: `specdiff/sampler.py::_prefetch_nearest`, for the full-acceptance,
  same-depth-nearest, and parent-fallback cases.
- Repository: `specdiff/trees.py::DraftTree`, for layers, children, and internal nodes.
