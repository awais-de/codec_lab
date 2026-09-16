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
- [ ] Ring-source run at high capacity using the existing harness
- [ ] Both fairness conditions logged
- [ ] Result compared against EXP-05 (3.29 dB) and EXP-06 (3.75 dB, PR 1.997)
- [ ] One-paragraph interpretation, including whether participation ratio moved
- [ ] Findings recorded for EXP-08

## Result
_(fill in after the run: did gain drop, did participation ratio move away from 2.0,
does the latent scatter show the ring visibly unrolled)_
