# AUDIT-01: Reconstruction fidelity of the ring encoders

Tracks **[#16](https://github.com/awais-de/codec_lab/issues/16)** · Stage: Diagnosis

## Question
Are the EXP-06/07 encoders continuous, injective maps on the ring — the premise behind
"their latent is still a closed loop" — as a logged fact rather than an assumption?

## Hypothesis
Reconstruction training should have made both networks near-invertible on the support:
high un-quantized reconstruction SQNR, an encoder Jacobian whose determinant keeps one
sign and stays away from zero along the ring, and no far-apart ring points collapsing
onto nearby latents.

## Setup
- Networks: EXP-06 (h=4) and EXP-07 (h=64), re-trained from their exact configs; seed 0
  must reproduce the committed `gain_latent_db` (3.749 / 4.781) to 1e-9 before the weights
  are trusted. Seeds 0, 1, 2 for each.
- Source: `ring_2d` (EXP-05/06/07's config). No quantizer involved.
- Metrics (`codeclab.models.fidelity`): `recon_sqnr_db` of `decoder(encoder(X))`;
  `det J` of the encoder on 2000 ring samples (scaled by the variance-conservation factor);
  far-pair ratio `‖E(x_i)−E(x_j)‖ / ‖x_i−x_j‖` for pairs > 30° apart.

## Pass bar (stated in advance)
recon ≥ 25 dB · zero sign flips in `det J` · min pair ratio > 0.1 — per seed.

## Run
```bash
python experiments/audit_ring/a1_recon_fidelity/run.py
```
Output → `results/audit_ring/a1_recon_fidelity/results_<timestamp>/`: `results.json`,
`det_jacobian_vs_angle.png`.

## Definition of done
- [x] EXP-06/07 networks reproduced to all printed digits
- [x] Three fidelity metrics logged, 3 seeds × 2 networks
- [x] Pass bar evaluated per seed — not met on reconstruction (cause identified), injectivity criteria pass
- [x] One-paragraph interpretation: "continuous + injective on the ring" is a logged fact

## Result
Run `results_20260918215452`. Seed 0 reproduces EXP-06/07 to 1e-9. Injectivity holds
directly for every network and seed: `det J` keeps one sign (all orientation-reversing),
min |det J| 0.08–0.5, far-pair ratio ≥ 0.63. The reconstruction criterion fails (17–20 dB
vs ≥ 25): at proxy width 1.0 the decoder is a denoiser, not an inverse — at width 0.2 it
reaches 32–33 dB. Two side results: (1) an additive error-floor model explains most of
the latent→signal gain compression on the ring; (2) EXP-07's "gain rises with capacity"
exists only at proxy width 1.0 — at 0.2 / 0.5 both capacities sit within ±0.2 dB of the
classical 3.29 with the ring untouched (PR 2.00), 3 seeds each. Full write-up on #16.
