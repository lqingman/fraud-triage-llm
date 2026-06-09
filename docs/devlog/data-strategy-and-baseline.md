# Design Note — Data Strategy & Why We Compare Against XGBoost

*An interview-oriented write-up of two design decisions: (1) why this project uses three
different datasets with distinct roles, and (2) why we benchmark the fine-tuned LLM against
a classical XGBoost baseline.*

> **Note on history.** The original plan used TeleAntiFraud-28k (primary) and redasers/difraud
> (cross-domain). Both were dropped after inspection — TeleAntiFraud is Chinese and gated;
> DIFRAUD's label is "deceptive," not "fraud." The *roles* below are unchanged; only the
> datasets filling them changed. See `phase-0c-call-corpus.md` and `phase-0d-clair-crossdomain.md`.

---

## Part 1 — The three-dataset strategy

A common misconception: "you're comparing datasets against each other." We are not. The three
datasets play **three different roles in a pipeline**, like stations on an assembly line.

| Dataset | Role | Used for |
|---|---|---|
| **BothBosu/scam-dialogue** | Prototype | Get the pipeline working end to end |
| **English call corpus (~9k)** | Primary train + test | Actually train and test the model |
| **CLAIR_email_fraud (~12k)** | Cross-domain eval | Stress-test generalization (never trained on) |

### BothBosu — the scaffold
Small (1,600 rows), text-only, CPU-friendly. Its job is to get the **whole pipeline running**
(load → format → train → eval → serve) before investing in the expensive parts. It is *not*
meant to produce a strong model — it has no gold rationales, so the `reason` field is templated.
Think of it as a small test build before constructing the real thing.

### English call corpus — the workhorse
~9,000 English phone-call transcripts, unioned and de-duplicated from four HF datasets
(`menaattia/phone-scam-dataset`, `shakeleoatmeal/phone-scam-detection-synthetic`,
`BothBosu/multi-agent-scam-conversation`, `BothBosu/single-agent-scam-conversations`). Balanced
(~50% fraud) with scam-type labels that drive `fraud_type`. We re-split it 75/10/15 and report
in-distribution results here.

*Honest limitations, worth saying out loud in an interview:*
- **It's synthetic, and there is no large English phone-call fraud dataset** — real call
  transcripts in English simply don't exist at scale (the one large *call* corpus,
  TeleAntiFraud-28k, is Chinese and gated). ~9k is small but sufficient for QLoRA on a narrow,
  format-heavy task.
- **No gold rationales.** Like BothBosu, the `reason`/`flagged_spans` are templated, so the model
  learns the *structure* of an explainable verdict, not richly-argued rationales. Recovering true
  rationale quality would mean LLM-synthesized reasons — a deliberate piece of future work.

### CLAIR — the unseen exam
~12k advance-fee ("419") scam emails vs. legitimate email (`tasksource/CLAIR_email_fraud`), with
an **explicit `FRAUD` / `NOT_FRAUD` label**. Used **only for evaluation, never for training**. It
answers the single most important question about any ML model:

> Did the model actually *learn to detect fraud*, or did it just *memorize the quirks of the
> training data*?

A different **channel** (email, not call) and a different **source** from the training set, so a
high score here is honest evidence of generalization rather than overfitting. (We chose CLAIR
over the originally-planned DIFRAUD precisely because CLAIR's positive class *is fraud* — DIFRAUD
labeled generic "deception," including non-fraud domains like fake news and opinion spam, which
would unfairly penalize a fraud model.)

**One-line summary:** *BothBosu to rehearse → call corpus to train and test → CLAIR as the final
out-of-distribution exam.*

---

## Part 2 — Train/test split: standard 7/2/1 *plus* a cross-domain test

A natural question: "I was taught to split one dataset 7/2/1 (train/val/test). Why use a
separate dataset for testing — is that better?"

The answer: **the standard split is still here and still required. We add a second, harder test
on top of it.** These are two different things.

### Two kinds of test

**(a) In-distribution split — the classic 7/2/1.**
On the primary dataset (the call corpus) we do exactly the textbook thing: split into
train / validation / test (our config: 75/10/15), with the test set held out and never seen
during training. Phase 0 produces this — a **6749 / 901 / 1350** split.

**(b) Out-of-distribution test — CLAIR.**
A separate, harder exam *in addition to* (a).

### Why (a) alone is not enough

The hidden flaw in a same-source split: train and test come from the **same origin** — same
generators, same style, same labeling conventions. So a model can score high by learning
**dataset-specific shortcuts** rather than the underlying signal of fraud.

> **Analogy.** A same-source 7/2/1 split is like a final exam drawn from *the same workbook* you
> studied. Score 95% — but did you *understand*, or did you *memorize that workbook's patterns*?
> The cross-domain test (CLAIR) is a final exam from a *completely different workbook* — different
> channel, different author. Still score high? You genuinely learned it. Collapse? You were
> memorizing. **That performance gap is the real measure of generalization** — something the
> same-source split can never reveal.

| | In-distribution (7/2/1) | Cross-domain (CLAIR) |
|---|---|---|
| Question answered | "Learned this data's patterns?" | "Works on unfamiliar scenarios?" |
| Failure mode | Score can be inflated (overfitting) | More honest, closer to production |
| Industry term | in-distribution | out-of-distribution / cross-domain |

**Not either/or — they're complementary.** An in-distribution score alone always invites the
question "are you just overfitting?" — the cross-domain test is the answer.

*Implementation detail worth mentioning in an interview:* splits are **stratified** on the
binary fraud label, so the fraud ratio is preserved across train/val/test (verified: 0.500 in
all three call-corpus splits). This prevents class imbalance from being introduced by the split
itself.

---

## Part 3 — Why benchmark against XGBoost

The real "comparison" in this project is **not between datasets** — it's between **two models**:
our fine-tuned LLM vs. a classical XGBoost classifier (TF-IDF + XGBoost).

### The question an interviewer (or skeptic) will ask

> "Fraud detection is just binary classification. Why fine-tune a 7B-parameter LLM when XGBoost
> would do?"

### What we set out to prove

- **XGBoost** outputs a number — "87% likely fraud." It **cannot explain why.**
- **Our LLM** outputs a *structured, explainable verdict*:

```json
{
  "risk": "high",
  "fraud_type": "tech_support_scam",
  "reason": "Caller claimed to be Microsoft support and requested remote access and gift-card payment.",
  "flagged_spans": ["remote access to your computer", "pay with gift cards"]
}
```

For a real fraud-triage analyst, the second form is the useful one: it can be **explained,
audited, and used as evidence.**

### How we measure it (Phase 2)

Run both models on the **same held-out test set** and report:

- **F1 / PR-AUC** on the binary fraud label — to show the LLM is **at least as accurate** as
  XGBoost (not worse). PR-AUC is the headline number because fraud is imbalanced.
- **LLM-only metrics** — JSON-validity rate and explanation quality — to show the LLM **delivers
  something XGBoost structurally cannot**: a defensible rationale.

### What the baseline already shows

The XGBoost baseline is fit and reported (persisted to `models/baseline_xgb.joblib`,
`reports/metrics.json`). It already makes the overfitting story concrete:

| | In-distribution (call test) | Cross-domain (CLAIR) |
|---|---|---|
| Precision / Recall / F1 | 0.99 / 0.99 / **0.989** | 0.43 / 0.85 / **0.572** |
| PR-AUC | **0.999** | **0.381** |

The ~0.99 → ~0.38 PR-AUC drop is exactly the "memorized the workbook" gap: a TF-IDF model trained
on synthetic call vocabulary nearly aces its own held-out split but over-flags and loses ranking
quality on real fraud emails. **That gap is the bar the fine-tuned LLM has to beat** — and the
reason a model with genuine language understanding is worth the cost.

### The actual thesis

> The pitch is **not** "the LLM crushes XGBoost on accuracy." It's: **at comparable accuracy,
> the LLM also explains its decision — and explainability is exactly what a fraud-triage
> workflow needs.** The XGBoost baseline exists to make that trade-off concrete and honest,
> rather than asserting the LLM is better by fiat.

---

## TL;DR for an interview

- **Three datasets, three jobs:** BothBosu (prototype the pipeline) → English call corpus (train +
  in-distribution test) → CLAIR fraud emails (out-of-distribution generalization test).
- **Standard 7/2/1 is used** on the primary set; the cross-domain test is an **additional**,
  more honest check against overfitting — not a replacement.
- **XGBoost is the control group**, not a rival to beat on accuracy. The baseline's in-dist 0.999
  vs. cross-domain 0.381 PR-AUC already shows the generalization gap; the LLM's job is to close
  it *while adding* the explainability the use case actually requires.
