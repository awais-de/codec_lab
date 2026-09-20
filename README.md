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
  quantizers, the PCA transform, hand-built ring encoders, latent diagnostics,
  a toy neural codec (training, freeze-then-swap evaluation, fidelity and
  warm-start tooling), rate-distortion helpers, plotting, run bookkeeping.
- `experiments/exp_NN_slug/` — one folder per experiment: `config.yaml`
  (parameters), `run.py` (the sweep), `README.md` (question, hypothesis, how
  to run, result). One exception: EXP-04 lives in `smoke_gaussian_neural/`,
  which kept its original name so its committed results stay addressable.
- `experiments/audit_ring/aN_slug/` — the ring-claim audit, a separate
  initiative with the same layout one level deeper.
- `results/<experiment>/results_<timestamp>/` — one folder per run: plots,
  metrics (`metrics.csv` or `results.json`, depending on the experiment), a
  config snapshot, and `meta.json` (git commit, seed, duration).

## Setup

```bash
pip install -e .              # core (numpy / matplotlib / pyyaml)
pip install -e ".[neural]"    # + torch, for the neural experiments
pip install -e ".[theory]"    # + scipy, only to re-run the frozen theory/ scripts
```

## Experiment suite

Tracked on the [project board](https://github.com/users/awais-de/projects/6).
Each experiment is a rung in a ladder — one mechanism confirmed at a time
before moving to a learned encoder.

| # | question | result |
|---|---|---|
| EXP-01 | With no correlation, how much does VQ still beat SQ? | 0.394 dB floor at 2 bits/dim (0.018 space-filling + 0.376 shape) |
| EXP-02 | Turn on source correlation — does VQ's gain match the classical memory-gain law? | Yes: `floor − 5·log₁₀(1−ρ²)` dB; 3.38 dB at ρ=0.85 |
| EXP-03 | Does a fixed linear transform (PCA) recover that gain? | Yes, collapses to 0.30 dB — but only with bit allocation, not rotation alone |
| EXP-04 | Does a trained neural encoder reproduce EXP-01/03's known answer? | Harness validated; flat ~0.71 dB at every capacity (documented allocation caveat) |
| EXP-05 | On a source with nothing linear to exploit (a ring), does VQ still win? | +3.29 dB (SQ 13.16, VQ 16.45) — VQ's 16.45 dB is exactly the closed-form ceiling for 16 points on the circle |
| EXP-06 | Can a small neural encoder reshape the ring in SQ's favour? | No: +3.75 dB, the latent is still a ring |
| EXP-07 | Can a large one? | No: +4.78 dB at h=64 — but that rise is a training-proxy artifact, see the audit |
| EXP-04b | Does dropping the fairness conditions help SQ? | Abandoned — the training route is unreliable even on the Gaussian control |
| AUDIT-01/02/03 | Are the ring claims properties of SQ, or of our training setup? | Of the setup: a seamless fold lets SQ match VQ, and the capacity trend exists only at one proxy width |

Neural results depend on the width of the training rate proxy (`rate_noise`).
EXP-04/06/07 used 1.0, which injects roughly 5.7 dB more distortion than the
K=16 quantizer they are then evaluated with; `w ≈ 0.5` is the rate-matched
width. The audit measured 0.2 and 0.5 as well. New neural experiments report
the sweep, not a single width.

## References

- Fang, Mou, Yuan, Kong, Rudner — *A Unified Rate-Distortion Perspective on
  Vector, Product, and Scalar Quantization*, arXiv:2609.02107
