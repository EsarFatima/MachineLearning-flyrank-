# Capstone Report — Content Decline-Risk Ranking

- **Author:** Esar Fatima
- **Lane:** Decline-risk classification (pivoted from CTR/Engagement Opportunity Scoring — see `work/ML-04_lane_pivot.md`)
- **Repo:** https://github.com/EsarFatima/MachineLearning-flyrank-
- **Date:** August 2026

> Mirrors `work/notebooks/capstone.ipynb`. Re-run that notebook top to bottom to reproduce every
> number below (seed = 42 throughout).

## Abstract

Editors reviewing large content portfolios can't manually check every page each cycle. Using
30,000 pseudonymized pages across 32 clients from a 90-day search/engagement dataset, I built a
Random Forest classifier to rank pages by decline risk (`trend_direction == "down"`) and
validated it on 7 clients held out entirely from training. Under an honest client-grouped split,
the model reaches AUC 0.591 and precision@50 of 0.56 — beating a transparent
staleness-and-visibility rule (AUC 0.492, precision@50 0.38) and a 0.511 base rate. A naive
random split would have overstated the same model at AUC 0.723, almost entirely from
client-level leakage. The result is a decision-support ranked queue, not a certainty about any
single page.

## 1. Problem framing

**Decision supported:** which existing content pages a client's editor should review for a
refresh this cycle, out of a portfolio too large to read by hand every time.

**Unit of analysis:** one page (one row = one URL/page for one client, 90-day GSC/GA4
aggregate window).

**Output:** a ranked review queue with a reason code per page (`review_for_refresh_priority`,
`review_for_refresh_secondary`, `spot_check_only`, `no_action`) — not an automated rewrite,
unpublish, or redirect.

**Cost of a wrong call:** a false positive costs an editor's review time on a page that wasn't
actually declining; a false negative means a genuinely declining page waits another cycle. Both
are recoverable — which is exactly why this stays a triage aid, never an automated action, and
why the recommendation is a *queue*, not a threshold.

**Why data/ML helps here:** a portfolio of thousands of pages can't be manually reviewed every
cycle. A transparent rule (Section 3) is cheap and explainable but coarse; a validated model can
rank within that coarseness — *if* it's checked honestly against the rule, not assumed better.

## 2. Data safety

**Source:** the starter CSV shipped with the internship repo (`data/raw/content_refresh_anonymized.csv`)
— 30,000 pseudonymized pages across 32 clients, one row per page, 90-day GSC/GA4 aggregates.

**Columns deliberately excluded from modeling:**
- `trend_direction`, `trend_pct` — these *build* the label; confession-tested in `w03_feature_leakage_check.ipynb` (adding `trend_pct` back in sends AUC to a fake 1.000).
- `impressions_last_30d`, `impressions_prev_30d` — the two raw components `trend_pct` is computed from (correlation with `trend_pct` = 1.000; sends AUC to 0.854 if added).
- `client_id` — used only to **group** the train/test split, never fed to the model as a feature.
- `content_id` — identifier only.

**Leakage risks considered:** the two label-derived-feature risks above (Type-1 leakage), plus
client-level leakage from a naive random split (measured directly in Section 5 — a random split
lets 31 of 32 clients appear on both sides, inflating Random Forest AUC from an honest 0.591 to
a leaky 0.723).

**Confirmed:** no client names, domains, raw URLs, or private query text appear anywhere in
`work/`. `client_id`/`content_id` are pseudonymous hashes from the released file, used for
grouping and de-duplication only.

## 3. Baseline

**The rule (from `w04_baseline_score.ipynb`):** a page is flagged if it is **stale** (≥90 days
since last update) **and visible** (90-day impressions between 300 and 30,000 — enough traffic
to matter, not so much it's already a top performer). Flagged pages are scored by
`impressions_90d` for ranking within the flag.

**Why it's a fair comparison:** it uses only two of the same signals available to the model
(staleness, visibility), is fully transparent (an editor can read the rule in one sentence), and
is scored on the exact same held-out rows and the exact same metrics as the model.

**Its numbers (client-grouped held-out split, 7 clients):** AUC 0.492, precision@20 = 0.45,
precision@50 = 0.38 — against a base rate of 0.511 on that same slice. The rule underperforms the
base rate on AUC, which is itself an honest, useful finding: a purely stale+visible cut doesn't
discriminate risk on its own; it needed a model to check whether the same underlying signals
could do better calibrated together.

## 4. Model / analysis

**Method:** a Random Forest classifier (300 trees, max depth 6, min 20 samples/leaf,
`class_weight="balanced"`), compared against a Logistic Regression on the same features. Random
Forest was chosen for its ability to capture non-monotonic signal shapes — several of the
signals audited in `w04_signal_audit.ipynb` (e.g. `position_tier`) turned out to be hump-shaped,
not linear, which a plain logistic fit can't represent as well.

**Target:** `is_declining_label = (trend_direction == "down")` — FlyRank's own trend call, used
directly rather than hand-built from scratch (see the lane-pivot note for why).

**Final feature set (13 columns, unchanged since `w03_feature_leakage_check.ipynb`):**
`content_age_days`, `days_since_last_update`, `log_impressions_90d`, `avg_position`, `ctr`,
`engagement_rate`, `search_volume`, `competition`, `word_count`, `has_keyword_data`,
`has_word_count`, `content_type`, `main_intent`.

**Left out on purpose:** anything derived from the label (Section 2), and `client_id`/`content_id`
as features (identifiers, not signal).

## 5. Evaluation

**Split:** client-grouped 80/20 (`GroupShuffleSplit`, seed 42) — 25 clients in training, 7 held
out entirely, zero client overlap. Chosen because clients are the repeating unit in this data
(many rows per client); a time-aware split wasn't available since the file is a single 90-day
snapshot, not a time series.

**Why the split matters more than the model:** the same Random Forest scored under a plain
random row-level split reads AUC 0.723 — 31 of 32 clients leaked across train/test. Under the
honest grouped split, it drops to 0.591. That +0.131 gap is memorization, not skill.

| Method | AUC | Precision@20 | Precision@50 |
|---|---|---|---|
| Base rate (majority class) | — | 0.51 | 0.51 |
| Baseline rule (w04) | 0.492 | 0.45 | 0.38 |
| Logistic Regression | 0.541 | 0.50 | 0.56 |
| **Random Forest** | **0.591** | **0.50** | **0.56** |

*(All rows: 7 held-out clients never seen in training, same split, same rows.)*

**Error shape:** the Random Forest and Logistic Regression tie at precision@20 but the Random
Forest pulls ahead at precision@50 — its edge shows up more in the top ~1.7% of a client's
portfolio than the very top sliver, consistent with a model that's calibrated across a broader
risk gradient rather than sharply separating only the most extreme cases.

## 6. Interpretation

**What the model found, in plain words:** three signals, tested individually in
`w04_signal_audit.ipynb`, tell three different stories. `engagement_rate` alone shows **no**
separation from the base rate (0.525–0.548 across all buckets) — a real negative result.
`position_tier` is **hump-shaped**: pages ranking very well (top 3) or very poorly (past
position 50) are *lower* risk than pages ranking respectably-but-not-great — those are the pages
"in motion." CTR, restricted to well-positioned pages, is the cleanest and most useful signal
found: it directly confirms the logic behind FlyRank's own `needs_ctr_fix` flag (a page ranking
well but getting comparatively few clicks is a real, monotonic risk signal — decline rate rises
from 0.475 to 0.643 as CTR falls within that slice).

**Negative result worth stating plainly:** the FlyRank research paper's own published
`needs_ctr_fix`-adjacent finding (71% holdout accuracy predicting growth) is reported without its
base rate and without confirming a brand-grouped split — exactly the two gaps this capstone
checked for on its own data and found mattered (Section 5's split gap). That's a useful,
constructive methodology note carried into this report, not a takedown of the paper.

## 7. Recommendation

**The ranked queue an editor sees tomorrow**, from the model scored across all 30,000 pages:

| Action | Pages | Meaning |
|---|---|---|
| `review_for_refresh_priority` | 1,498 | Stale + visible by the rule, AND in the model's top risk decile |
| `review_for_refresh_secondary` | 4,081 | Either the model's top decile without the rule, or the rule at moderate model risk |
| `spot_check_only` | 2,668 | Rule flags it, model doesn't agree — needs a human look, never a silent drop |
| `no_action` | 21,753 | Neither signal elevated |

**How an editor uses it:** start with the 1,498 priority pages — but check for single-client
concentration first. The hand review of the top of this queue found the top 50+ rows belonged
almost entirely to one client with a genuinely high (83.9%) decline rate and much higher typical
staleness (104 vs. 20 days median) — real, but it would monopolize a shared queue if read
straight down without a per-client cap or rotation.

**Confidence:** precision@50 of 0.56 against a 0.511 base rate is a real, modest edge — enough to
prioritize a review queue, not enough to claim a page *will* decline. Limits are explicit in
Section 5 of `work/notebooks/capstone.ipynb` and Section 8 below.

## 8. Reproducibility

**To re-run from a fresh clone:**
```bash
git clone https://github.com/EsarFatima/MachineLearning-flyrank-
cd MachineLearning-flyrank-
pip install -r requirements.txt
jupyter notebook work/notebooks/capstone.ipynb   # Run All
```
**Seed:** 42, fixed everywhere a split or model is fit (`GroupShuffleSplit`, `train_test_split`,
`RandomForestClassifier`, `LogisticRegression`).

**Environment:** pandas, numpy, scikit-learn (see `requirements.txt` for exact pins).

**Outputs regenerated by the notebook:** `work/outputs/content_action_queue.csv`,
`work/outputs/results_table.csv`, `work/figures/*.png` — all gitignored/regenerable, not
committed as data.

## 9. Acknowledgments & Data Credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai) (pseudonymized, internal
search/engagement data). Thanks to the FlyRank ML Internship program for data access and the
mentorship that shaped this analysis.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language throughout · no causal claims · no "predicted Google's algorithm" · no
> client-identifying details · numbers above match a fresh re-run of `capstone.ipynb`.
