# Design Note — Data Strategy & Why We Compare Against XGBoost

*An interview-oriented write-up of two design decisions: (1) why this project uses three
different datasets with distinct roles, and (2) why we benchmark the fine-tuned LLM against
a classical XGBoost baseline.*

---

## Part 1 — The three-dataset strategy

A common misconception: "you're comparing datasets against each other." We are not. The three
datasets play **three different roles in a pipeline**, like stations on an assembly line.

| Dataset | Role | Used for |
|---|---|---|
| **BothBosu/scam-dialogue** | Prototype | Get the pipeline working end to end |
| **TeleAntiFraud-28k** | Primary train + test | Actually train and test the model |
| **redasers/difraud** | Cross-domain eval | Stress-test generalization (never trained on) |

### BothBosu — the scaffold
Small (1,600 rows), text-only, CPU-friendly. Its job is to get the **whole pipeline running**
(load → format → train → eval → serve) before investing in the expensive parts. It is *not*
meant to produce a strong model — it has no gold rationales, so the `reason` field is templated.
Think of it as a small test build before constructing the real thing.

### TeleAntiFraud-28k — the workhorse
Large (28,511 speech-text pairs), includes **audio + transcripts**, and crucially ships with
**fraud-reasoning annotations** (the paper calls it a "slow-thinking" dataset). This is what
actually teaches the model to produce analyst-style explanations, and its audio side feeds the
Whisper ASR front-end (Phase 3). Final results are reported on this dataset.

### DIFRAUD — the unseen exam
95k samples across **7 different fraud domains**. Used **only for evaluation, never for
training**. It answers the single most important question about any ML model:

> Did the model actually *learn to detect fraud*, or did it just *memorize the quirks of the
> training data*?

By testing on a genuinely different distribution, a high score here is honest evidence of
generalization rather than overfitting.

**One-line summary:** *BothBosu to rehearse → TeleAntiFraud to train and test → DIFRAUD as the
final out-of-distribution exam.*

---

## Part 2 — Train/test split: standard 7/2/1 *plus* a cross-domain test

A natural question: "I was taught to split one dataset 7/2/1 (train/val/test). Why use a
separate dataset for testing — is that better?"

The answer: **the standard split is still here and still required. We add a second, harder test
on top of it.** These are two different things.

### Two kinds of test

**(a) In-distribution split — the classic 7/2/1.**
On the primary dataset (TeleAntiFraud) we do exactly the textbook thing: split into
train / validation / test (our config: 75/10/15), with the test set held out and never seen
during training. This is what Phase 0 already does for BothBosu — the 1199/161/240 split.

**(b) Out-of-distribution test — DIFRAUD.**
A separate, harder exam *in addition to* (a).

### Why (a) alone is not enough

The hidden flaw in a same-source split: train and test come from the **same origin** — same
collection process, same recording setup, same speaking style, same labeling conventions. So a
model can score high by learning **dataset-specific shortcuts** rather than the underlying
signal of fraud.

> **Analogy.** A same-source 7/2/1 split is like a final exam drawn from *the same workbook* you
> studied. Score 95% — but did you *understand*, or did you *memorize that workbook's patterns*?
> The cross-domain test (DIFRAUD) is a final exam from a *completely different workbook* — new
> domains, new author. Still score 90%? You genuinely learned it. Drop to 60%? You were
> memorizing. **That performance gap is the real measure of generalization** — something the
> same-source split can never reveal.

| | In-distribution (7/2/1) | Cross-domain (DIFRAUD) |
|---|---|---|
| Question answered | "Learned this data's patterns?" | "Works on unfamiliar scenarios?" |
| Failure mode | Score can be inflated (overfitting) | More honest, closer to production |
| Industry term | in-distribution | out-of-distribution / cross-domain |

**Not either/or — they're complementary.** Serious ML work (including the TeleAntiFraud paper)
reports cross-domain results precisely because an in-distribution score alone invites the
question "are you just overfitting?"

*Implementation detail worth mentioning in an interview:* splits are **stratified** on the
binary fraud label, so the fraud ratio is preserved across train/val/test (verified: ~0.50 in
all three BothBosu splits). This prevents class imbalance from being introduced by the split
itself.

---

## Part 3 — Why benchmark against XGBoost

The real "comparison" in this project is **not between datasets** — it's between **two models**:
our fine-tuned LLM vs. a classical XGBoost classifier.

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
  XGBoost (not worse).
- **LLM-only metrics** — JSON-validity rate and explanation quality — to show the LLM **delivers
  something XGBoost structurally cannot**: a defensible rationale.

### The actual thesis

> The pitch is **not** "the LLM crushes XGBoost on accuracy." It's: **at comparable accuracy,
> the LLM also explains its decision — and explainability is exactly what a fraud-triage
> workflow needs.** The XGBoost baseline exists to make that trade-off concrete and honest,
> rather than asserting the LLM is better by fiat.

---

## TL;DR for an interview

- **Three datasets, three jobs:** BothBosu (prototype the pipeline) → TeleAntiFraud (train +
  in-distribution test) → DIFRAUD (out-of-distribution generalization test).
- **Standard 7/2/1 is used** on the primary set; the cross-domain test is an **additional**,
  more honest check against overfitting — not a replacement.
- **XGBoost is the control group**, not a rival to beat on accuracy. It proves the LLM matches
  classical accuracy *while adding* the explainability the use case actually requires.
