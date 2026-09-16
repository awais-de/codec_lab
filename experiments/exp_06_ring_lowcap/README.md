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
- [ ] Ring-source run using the existing harness
- [ ] Both fairness conditions logged
- [ ] Result compared against EXP-05's 3.29 dB ceiling
- [ ] One-paragraph interpretation
- [ ] Latent statistics recorded for EXP-08

## Result
_(fill in after the run: does gain stay near 3.29 dB, does the latent scatter show
the ring reshaped at all, is off-diagonal energy still ~0)_
