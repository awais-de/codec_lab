# codec-lab

Classical and neural quantization experiments probing one question: **when does
vector quantization (VQ) actually beat scalar quantization (SQ), and when does
it stop mattering?**

VQ's advantage over SQ factors into three independent mechanisms (Lookabaugh &
Gray, 1989): **memory gain** (exploiting correlation between dimensions),
**shape gain** (matching code density to the source's pdf), and
**space-filling gain** (better cell geometry). This repo builds up evidence for
each on synthetic sources with known answers, then asks what happens once a
neural encoder sits in front of the quantizer.

## Structure

- `theory/` — frozen classical-quantization learning log (entropy, source
  coding, uniform/Lloyd-Max SQ, LBG VQ). Not touched by the experiment suite.
- `src/codeclab/` — shared package: synthetic sources, scalar/vector
  quantizers, the PCA transform, a toy neural codec, rate-distortion helpers,
  plotting, run bookkeeping.
- `experiments/exp_NN_slug/` — one folder per experiment: `config.yaml`
  (parameters), `run.py` (the sweep), `README.md` (question, hypothesis, how
  to run, result).
- `results/exp_NN_slug/results_<timestamp>/` — one folder per run: plots,
  `metrics.csv`, a config snapshot, and `meta.json` (git commit, seed,
  duration).

## Setup

```bash
pip install -e .              # core (numpy/scipy/matplotlib toolchain)
pip install -e ".[neural]"    # + torch, for the neural experiments
```

## Experiment suite

Tracked on the [project board](https://github.com/users/awais-de/projects/6).
Each experiment is a rung in a ladder — one mechanism confirmed at a time
before moving to a learned encoder.

| # | question | result |
|---|---|---|
| EXP-01 | With no correlation, how much does VQ still beat SQ? | ~0.4 dB floor (space-filling + shape gain) |
| EXP-02 | Turn on source correlation — does VQ's gain match the classical memory-gain law? | Yes: `floor − 5·log₁₀(1−ρ²)` dB; 3.38 dB at ρ=0.85 |
| EXP-03 | Does a fixed linear transform (PCA) recover that gain? | Yes, collapses to the floor — but only with bit allocation, not rotation alone |
| EXP-04 | Does a trained neural encoder reproduce EXP-01/03's known answer? | Harness validated; the training proxy converges to a related-but-different optimum than classical bit allocation (documented caveat) |
| EXP-05 | On a source with no correlation to exploit at all (a ring), does VQ still win? | +3.29 dB — confirms the mechanism here is manifold/clustering, not memory gain |
| EXP-06+ | Does a neural encoder's capacity determine how much of that advantage survives? | in progress |

## References

- Fang, Mou, Yuan, Kong, Rudner — *A Unified Rate-Distortion Perspective on
  Vector, Product, and Scalar Quantization*, arXiv:2609.02107
