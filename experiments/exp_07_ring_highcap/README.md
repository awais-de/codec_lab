# EXP-07: High-capacity neural codec on the ring source

Tracks **[#6](https://github.com/awais-de/codec_lab/issues/6)** · Stage: Investigation

## Question
Can a genuinely expressive network unroll the ring — and if it does, does VQ's
advantage actually collapse?

## Hypothesis
EXP-06 (h=4) could not unroll the ring: gain stayed at the classical ceiling and
participation ratio stayed flat at ~2.0. If capacity matters, a much wider
network should push gain down and move participation ratio away from 2.0
toward 1 — the diagnostic that didn't budge at low capacity. If it stays flat
here too, that's the bigger finding: this style of encoder can't unroll the
ring at any capacity tested.

## Setup
- Source: 2D ring, radius 1.0, radial jitter 0.1 (`sources.ring_2d`, EXP-05/06's config)
- Codec: autoencoder, single hidden layer, h=64 (`codeclab.models`)
- Training: reconstruction MSE + additive-noise rate proxy, frozen encoder
  (fairness condition 1)
- Quantizer: SQ (per-axis Lloyd-Max + bit allocation) vs VQ (LBG), K=16
  (fairness condition 2)

## Run
```bash
python experiments/exp_07_ring_highcap/run.py
```
Output → `results/exp_07_ring_highcap/results_<timestamp>/`: `results.json`,
`ring_highcap_sq_vs_vq.png`.

## Definition of done
- [x] Ring-source run at high capacity using the existing harness
- [x] Both fairness conditions logged
- [x] Result compared against EXP-05 (3.29 dB) and EXP-06 (3.75 dB, PR 1.997)
- [x] One-paragraph interpretation, including whether participation ratio moved
- [x] Findings recorded for EXP-08

## Result
Run `results_20260916090854`. Latent gain **+4.781 dB** (SQ 13.101, VQ 17.882),
signal gain +2.416 dB, participation ratio 1.833. Gain went *up* with capacity
rather than down, and the latent scatter shows the ring stretched into an
ellipse, not unrolled. Cross-architecture check (h=64 and 2×64, tanh and ReLU,
3 seeds each — 12 runs): nothing approached the floor.

**Partly superseded — see the correction on issue #6.** The ring-claim audit
showed the rise is a `rate_noise = 1.0` artifact: at widths 0.2/0.5 both h=4 and
h=64 give 3.1–3.4 dB with the ring untouched (#16), and a network warm-started at
a folded latent is driven back to this same ellipse by the width-1.0 objective
(#18). A seamless fold under which SQ matches VQ does exist (#17), so the ring
limits unrolling, not SQ. What survives: from random init, no capacity tested
reduces the gap below the classical 3.29 dB.
