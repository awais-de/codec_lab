# EXP-06: Low-capacity neural codec on the ring source

Tracks **[#5](https://github.com/awais-de/codec_lab/issues/5)** · Stage: Investigation

## Question
Can a small neural encoder do anything a linear transform provably cannot — turn
the ring into a representation SQ can handle as well as VQ?

## Hypothesis
No linear map turns a circle into an axis-aligned-separable shape (a linear
transform of a circle is always an ellipse). A small (1 hidden layer, ~4 units)
network shouldn't manage it either, so VQ gain should stay close to EXP-05's
classical baseline (3.29 dB), not collapse toward the floor.

## Setup
- Source: 2D ring, radius 1.0, radial jitter 0.1 (`sources.ring_2d`, EXP-05's config)
- Codec: autoencoder, 1 hidden layer, 4 units (`codeclab.models`)
- Training: reconstruction MSE + additive-noise rate proxy, frozen encoder
  (fairness condition 1); no discrete quantizer in the training loop
- Quantizer: SQ (per-axis Lloyd-Max + bit allocation) vs VQ (LBG), K=16
  (fairness condition 2)

## Run
```bash
python experiments/exp_06_ring_lowcap/run.py
```
Output → `results/exp_06_ring_lowcap/results_<timestamp>/`: `results.json`,
`ring_lowcap_sq_vs_vq.png` (signal-space and latent-space reconstructions, SQ vs VQ).

## Definition of done
- [x] Ring-source run using the existing harness
- [x] Both fairness conditions logged
- [x] Result compared against EXP-05's 3.29 dB ceiling
- [x] One-paragraph interpretation
- [x] Latent statistics recorded for EXP-08

## Result
Run `results_20260916085107`. Latent gain **+3.749 dB** (SQ 13.503, VQ 17.252),
signal gain +2.011 dB, participation ratio 1.997, off-diagonal energy 7.8e-5.
The latent is still a ring — the encoder reshaped nothing — and the gain sits
just above EXP-05's classical 3.29 dB rather than collapsing toward the floor.
A small network cannot unroll the ring, as predicted.

Recorded at `rate_noise = 1.0`. AUDIT-01 (#16) later established that this
setting matters: at widths 0.2 and 0.5 the same configuration gives 3.33–3.38 dB
across 3 seeds. The conclusion — no collapse toward the floor — holds at every
width tested. Full write-up on issue #5.
