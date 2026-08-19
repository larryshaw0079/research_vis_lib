# Selection rubric

`python -m rvl recommend` produces a deterministic shortlist. This document is how to
turn that shortlist into a decision, and what to do when it is empty or wrong.

## What the score already accounts for

- **Hard filters**: the reading's data kind must be in `SPEC.kinds`, and the
  category and series counts must fall inside `SPEC.categories` / `SPEC.series`.
  A template that fails either is not offered at all, and `recommend` prints why
  under "not applicable".
- **Shape comfort**: a dataset in the middle of a template's supported band scores
  higher than one at the edge.
- **Feature affinities**: ordered categories, non-negativity, sums-to-100,
  bounded scale, wide dynamic range, uncertainty, emphasis, long labels, category
  and series counts.

## What the score cannot know

Override the top pick when any of these applies.

- **The user said what they want.** "Make it a pie chart" beats any score.
- **House style.** A paper whose other figures are cartesian bars should not gain
  one polar chart for variety.
- **Venue norms.** Reviewers in some fields read radar charts as decoration.
- **The figure's job.** A chart that must support a precise numeric claim in the
  text needs a readable axis, not an impressive one.
- **Colour-blind and greyscale printing.** A template that distinguishes series
  only by hue is risky for print; mention it if the user is submitting to print.

## Decision order

Work down this list; the first rule that applies decides.

1. **Is any template applicable at all?** If `recommend` returns nothing, do not
   force a fit. Read the "not applicable" reasons, then tell the user what is
   missing — usually too few categories, too many series, or no numeric column.
2. **Does the data have an ordered axis that must be read as a sequence?**
   Dates, doses, epochs, distances. Choose a template with
   `ordered_categories=True`. Circular and polar geometries put the last point
   next to the first, which invents a cycle that is not in the data.
3. **Do the values form a whole?** If they sum to 100, or to a total the user
   cares about, prefer a composition template (`parts_of_whole`,
   `stacked_parts`, `nested_parts`) over a plain matrix. Plotting shares as
   independent bars throws away the constraint.
4. **Is the measurement a distribution or a pair?** Many observations per group
   means a distribution template, not a bar of group means; a bar chart of means
   hides spread. Paired x and y means a scatter with a fit, not two separate bars.
5. **Is the claim comparative and precise?** If the reader must rank series within
   a category, prefer cartesian. Angle and area are read less accurately than
   length along a common axis.
6. **How wide is the dynamic range?** Above roughly 100x, a length or radius
   encoding renders the small values invisible. Warn the user and suggest either a
   log axis, splitting the figure, or a template that prints values as text.
7. **Otherwise take the top score**, and mention the runner-up.

## Per-kind guidance

**matrix** — the most common reading, and the most contested, since several
templates accept it. Discriminate by counts and by what the reader must do:
few categories and few series favour a compact layout; many categories with few
series favour a radial profile; a bounded metric across many tasks favours a
smooth radar; a benchmark table with a winner favours a template that supports
`highlight`.

**parts_of_whole** — check `sums_to_100` in the profile. If the values are raw
magnitudes, decide explicitly whether to normalise and say which you did.

**stacked_parts** — only valid for non-negative values. If any segment can be
negative, this reading is wrong even though the profiler offered it.

**nested_parts** — needs the same part keys at both levels so one colour key
serves the summary and the breakdown. Confirm that before choosing it.

**series_with_totals** — the aggregate panel implies the total is meaningful.
Summing a rate over time is not an integral; if the source did that, keep the
caveat visible in the caption rather than silently presenting it as cumulative.

**xy_samples** — check that a linear fit is defensible. If the profile shows a
wide x range and the relationship is visibly curved in the rendered figure, say so
rather than reporting a slope.

**distribution_samples** — require enough observations per series. With fewer than
about 30 per group, a histogram is noise; prefer reporting the values directly.

**set_membership** — the counts must be counts of records, not proportions.
Confirm the boolean columns really are membership flags rather than 0/1 measurements.

**set_overlap** — validate that every unique count is at or below its total, and
that the shared core is genuinely shared by all groups. The builder enforces the
first; only the user can confirm the second.

## Reporting the choice

State the template, the reading, and the single reason that decided it. Name the
runner-up when the scores were within a few points. Mention any warning the
recommender raised that the user should know about, especially wide dynamic range
and long labels.

Keep it to two or three sentences. The user wants the figure, not the deliberation.
