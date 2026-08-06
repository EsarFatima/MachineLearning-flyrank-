# ML-04 — Honest Lane-Pivot Note

## Where I started

My provisional lane (ML-02) was **CTR/Engagement Opportunity Scoring**: rank pages
that have decent search visibility (good `avg_position`, enough `impressions_90d`)
but are under-capturing clicks or engagement relative to that visibility. The pitch
was that this was a well-defined, actionable gap, and the starter CSV already had
the fields to try it (`ctr`, `avg_position`, `impressions_90d`, `engagement_rate`,
`scroll_rate`) without needing the full warehouse pull.

Digging into the actual distribution killed the simplest version of that plan before
I got far: `ctr` in this file is a 0–100 scale that's heavily zero-inflated even in
the `top_3` position tier (median 0%, mean 1.48%, max 100%). A flat "CTR < 0.5"
threshold mostly just caught pages with **no recorded clicks at all**, not pages that
were meaningfully underperforming. A pure CTR-opportunity ranking would have been
ranking noise.

## The detour I scoped and didn't take

Before settling on a fix, I scoped a bigger move (ML-04's data contract exercise):
pull from the full daily-grain warehouse table (`fact_content_daily_performance`,
GSC + GA4, ~9.8M rows for a single month) and build a proxy label from raw
visibility/engagement columns instead of relying on any precomputed field. I wrote
and verified the contract — grain, time window, feature/label/excluded buckets — but
didn't carry it forward into modeling. The starter CSV already had a genuine,
pre-labeled trend signal (`trend_direction`) that let me build and validate a real
classifier without first having to hand-construct a label from scratch on a much
larger, ungated dataset. Given the 7-week window, that tradeoff wasn't close.

## Where I landed

I reframed the question from a static ranking heuristic to a validated **decline-risk
classification** problem: predict `is_declining_label` (from `trend_direction`),
using the same broad content-health signals (position, CTR, engagement, staleness,
search volume, competition, content type/intent) the CTR-opportunity lane was
already built around — but now as inputs to a model that gets checked against a
transparent rule baseline, not treated as a ranking on their own.

That gave me somewhere to be honest instead of confident:

- **Leakage check first.** The clean feature set holds at AUC 0.541; adding
  the raw 30-day windows that built the label jumps it to 0.854; adding
  `trend_pct` itself hits 1.000. All label-derived features stayed out.
- **Split matters more than the model.** A random 80/20 split let 31 of 32
  clients leak across train/test, inflating Random Forest AUC to 0.723. A
  client-grouped split — the honest version — drops that to 0.591.
- **The real number, on 7 held-out clients the model never trained on:**
  precision@50 of 0.56, against a 0.511 base rate and the rule-based baseline's
  0.38 on the same rows (AUC 0.591 vs. 0.492).

That's the number the action playbook (w07) is built on: a ranked queue of 1,498
flagged-for-review pages (of 30,000), with reason codes, and an explicit note that
only the 20.5% of rows from held-out clients carry that independent validation —
the rest were seen in training and would read optimistically if scored the same way.

## Why this is the honest framing, not the confident one

The original lane's boldest version of itself would have said "these pages are
under-monetizing their visibility." What I can actually say is narrower and more
defensible: *on a held-out sample, this model ranks pages by decline risk better
than a simple staleness-and-visibility rule, at a modest, real margin — enough to
prioritize a limited review queue, not enough to claim causal or portfolio-wide
certainty.* That's the language the paper's Results and Limitations sections carry
forward.
