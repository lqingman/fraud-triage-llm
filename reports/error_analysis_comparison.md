# In-domain vs. CLAIR error comparison

The same parser, scoring rules, and error-analysis script were used for both
saved prediction files. Invalid structured outputs count as non-fraud, exactly
as in the main evaluator.

| metric | in-domain calls | CLAIR emails |
|---|---:|---:|
| examples | 1,350 | 1,926 |
| fraud prevalence | 50.0% | 44.4% |
| precision | 0.985 | 0.750 |
| recall | 0.901 | 0.864 |
| F1 | 0.941 | 0.803 |
| JSON validity | 94.7% | 93.3% |
| false-positive rate | 1.3% (9/675) | 23.1% (247/1,070) |
| false-negative rate | 9.9% (67/675) | 13.6% (116/856) |
| invalid-output rate | 5.3% (71/1,350) | 6.7% (129/1,926) |
| false negatives caused by invalid output | 82.1% (55/67) | 84.5% (98/116) |
| dominant template share of false positives | 33.3% (3/9) | 99.6% (246/247) |
| diagnostic risk-only fallback F1 | 0.973 | 0.853 |

## Interpretation

1. **Schema adherence is a general model weakness, not primarily a domain-shift
   failure.** Invalid-output rates and the share of false negatives caused by
   invalid output are similar in both domains. Constrained decoding or bounded
   schema repair should therefore improve both evaluations.
2. **The large CLAIR degradation is overwhelmingly a false-positive problem.**
   The false-positive rate grows from 1.3% to 23.1%, while the false-negative
   rate grows only from 9.9% to 13.6%.
3. **The cross-domain false positives are highly systematic.** A single generic
   high-risk verdict accounts for 246 of 247 CLAIR false positives, compared
   with only 3 of 9 in-domain false positives.
4. **The model carries call-domain language into email evaluation.** The
   dominant CLAIR output says `Caller exhibits a scam pattern`, even for short
   or ordinary legitimate emails. This is evidence of prompt/training-template
   overfitting rather than random classification noise.

## Decision

Do not retrain blindly. First implement constrained structured output (or a
strictly measured repair path) and re-score both saved prediction sets. Then
address the separate CLAIR default-high behaviour with channel-neutral prompt
wording and a targeted set of legitimate, ambiguous, and short email examples.
