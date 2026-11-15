# Retrieval accuracy metrics

How well PLaNETS picks the right catalog variable for a plain-language question when no LLM is used.
Measured on 14 Nov 2026 with `python eval/eval_retrieval.py` (main set) and `python eval/eval_retrieval.py --heldout`.

## What was measured

Each case is a question plus groups of acceptable catalog ids. One group is one variable the question asks about,
so a two-variable question has two groups. Search returns the top 5 variables.

- **Top-1**: the best hit is acceptable for some group.
- **Cov@3 / Cov@5**: every group has an acceptable id in the top 3 / top 5. For one-variable questions this is
  simply "the right variable is in the top 3 / 5".
- **Off-topic cases** (no right answer) pass only when nothing is returned.

For two-variable questions, top-1 counts a hit on any one group, so the cov columns are the meaningful ones.

### Data sets
| Set | Cases | Purpose |
|---|---|---|
| Main (`eval/retrieval_cases.json`) | 84 | Used while building and tuning. Optimistic. |
| Held-out (`eval/retrieval_cases_heldout.json`) | 42 | Written before the phrase lexicon and never tuned against. The wording was seen while the lexicon was written, so it is only semi-independent. |

### Engines compared
All three engines use the same index, phrase lexicon, keyword-phrase boost and typo correction.

| Engine | What it does |
|---|---|
| **tfidf** | Keyword (TF-IDF) search on the whole question |
| **tfidf-np** | Same, after removing place names and years from the question |
| **hybrid** | Place names removed, keyword and embedding (bge-small) rankings merged by reciprocal rank fusion, plus an off-topic check. **This is what the app uses.** |

## Results: hybrid (the app's engine)

| Case type | Main n | Top-1 | Cov@3 | Cov@5 | Held-out n | Top-1 | Cov@3 | Cov@5 |
|---|---|---|---|---|---|---|---|---|
| **All cases** | 84 | 89% | 95% | 96% | 42 | 88% | 98% | 98% |
| Basic | 8 | 100% | 100% | 100% | 2 | 100% | 100% | 100% |
| Indirect | 15 | 100% | 100% | 100% | 6 | 100% | 100% | 100% |
| Slang | 7 | 86% | 100% | 100% | 3 | 100% | 100% | 100% |
| Typos | 5 | 100% | 100% | 100% | 3 | 100% | 100% | 100% |
| Hard | 31 | 81% | 90% | 94% | 18 | 78% | 100% | 100% |
| Two-variable | 13 | 100%* | 85% | 92% | 6 | 100%* | 83% | 83% |
| Compare (two places) | 3 | 100% | 100% | 100% | 2 | 100% | 100% | 100% |
| Place-name noise | 5 | 100% | 100% | 100% | 1 | 100% | 100% | 100% |
| Off-topic (should return nothing) | 3 | 100% | 100% | 100% | 3 | 100% | 100% | 100% |
| Giovanni (needs token) | 10 | 60% | 80% | 80% | 6 | 67% | 100% | 100% |
| PO.DAAC ocean (needs token) | 5 | 100% | 100% | 100% | 3 | 33% | 100% | 100% |
| Ambiguous | 1 | 100% | 100% | 100% | - | - | - | - |

\* Any-group hit; see "What was measured".

### Splitting a question into its variables
`nl.question_variables` cuts a question at joining words and searches each piece.

| Set | Two-variable questions where every variable was found | Single-variable questions wrongly split |
|---|---|---|
| Main | 13 of 13 | 0 of 68 |
| Held-out | 6 of 6 | 1 of 33 ("longer and brighter" read as two variables) |

## Results: all three engines

### Main set (84 cases)

| Case type | n | tfidf top-1 / cov@3 / cov@5 | tfidf-np top-1 / cov@3 / cov@5 | hybrid top-1 / cov@3 / cov@5 |
|---|---|---|---|---|
| All | 84 | 93 / 90 / 94 | 95 / 92 / 95 | 89 / 95 / 96 |
| Ambiguous | 1 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Basic | 8 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Compare | 3 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Giovanni | 10 | 70 / 80 / 80 | 80 / 80 / 80 | 60 / 80 / 80 |
| Hard | 31 | 94 / 87 / 94 | 94 / 87 / 94 | 81 / 90 / 94 |
| Indirect | 15 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Two-variable | 13 | 100 / 69 / 92 | 100 / 69 / 92 | 100 / 85 / 92 |
| Off-topic | 3 | 67 / 67 / 67 | 67 / 67 / 67 | 100 / 100 / 100 |
| Place-name noise | 5 | 80 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| PO.DAAC | 5 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Slang | 7 | 100 / 100 / 100 | 100 / 100 / 100 | 86 / 100 / 100 |
| Typos | 5 | 80 / 80 / 80 | 100 / 100 / 100 | 100 / 100 / 100 |

### Held-out set (42 cases)

| Case type | n | tfidf top-1 / cov@3 / cov@5 | tfidf-np top-1 / cov@3 / cov@5 | hybrid top-1 / cov@3 / cov@5 |
|---|---|---|---|---|
| All | 42 | 90 / 90 / 93 | 93 / 90 / 93 | 88 / 98 / 98 |
| Basic | 2 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Compare | 2 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Giovanni | 6 | 83 / 83 / 100 | 83 / 83 / 100 | 67 / 100 / 100 |
| Hard | 18 | 89 / 94 / 100 | 94 / 94 / 100 | 78 / 100 / 100 |
| Indirect | 6 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Two-variable | 6 | 100 / 83 / 83 | 100 / 83 / 83 | 100 / 83 / 83 |
| Off-topic | 3 | 67 / 67 / 67 | 67 / 67 / 67 | 100 / 100 / 100 |
| Place-name noise | 1 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| PO.DAAC | 3 | 67 / 67 / 67 | 67 / 67 / 67 | 33 / 100 / 100 |
| Slang | 3 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Typos | 3 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |

## How the system improved (main set, all cases)

The main set grew from 79 to 84 cases when PO.DAAC cases were added, so the last row is not strictly comparable.
"TF-IDF baseline" here is the original keyword search, before the lexicon, typo correction and phrase boost.

| Stage | Top-1 | Cov@3 | Cov@5 |
|---|---|---|---|
| TF-IDF baseline (original) | 66% | 59% | 65% |
| Place names and years removed | 68% | 62% | 66% |
| Hybrid with embeddings | 75% | 76% | 80% |
| Phrase lexicon, typo fixing, off-topic check | 90% | 90% | 92% |
| Keyword-phrase boost, question splitting | 91% | 95% | 96% |
| PO.DAAC variables added (84 cases) | 89% | 95% | 96% |

Held-out set: top-1 67%, cov@3 79%, cov@5 82% after the hybrid and before the lexicon (39 cases); 88%, 98%, 98% now
(42 cases).

## Caveats

- **Small samples.** Several case types have 1 to 6 cases, so one miss moves a percentage a lot.
- **Tuning.** Weights, thresholds and lexicon were tuned on the main set, and the held-out wording was visible while
  the lexicon was written. Both sets are somewhat optimistic; expect lower scores on genuinely new questions.
- **Keyword search wins on top-1.** On the main set tfidf-np scores 95% against 89% for the hybrid. The hybrid is
  better on cov@3 and is the only engine that rejects off-topic questions, so the app uses it.
- **Token-gated variables.** Giovanni and PO.DAAC questions sometimes rank a POWER variable first (for example an
  air-temperature or pressure variable for an ocean question). They remain in the top 3, but top-1 is weaker there.
- **Ground truth is judgement.** Acceptable ids for each case were chosen by hand; near-duplicate variables are
  listed as alternatives where either is reasonable.

## Reproduce

```
.\.venv\Scripts\python.exe eval_retrieval.py              # main set, all engines
.\.venv\Scripts\python.exe eval_retrieval.py --heldout    # held-out set
.\.venv\Scripts\python.exe eval_retrieval.py -v           # also list every miss
```

Run `python -m planets.query.retrieve --build-embeddings` first if the catalog or `planets/query/lexicon.py` changed.
