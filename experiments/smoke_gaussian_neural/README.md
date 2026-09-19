# EXP-04: Neural harness validation (Gaussian smoke test)

Tracks **[#12](https://github.com/awais-de/codec_lab/issues/12)** · Stage: Reference

Formalised as EXP-04 after it ran; the folder and its results path keep the
original `smoke_gaussian_neural` name. Full reasoning:
`docs/design/neural-capacity-plan.md` (Phase A), a local design doc not committed
to git -- this README is the committed summary of what the script does and why.

## What it checks

Trains a linear, small (h=4), and large (h=64) autoencoder on the same
ρ=0.85 2D Gaussian source EXP-01/02/03 already have exact answers for, then runs
the freeze-then-swap SQ/VQ harness (`codeclab.models`) on each. Since the
rate-distortion-optimal transform for any Gaussian source is linear (a 2x2
matrix that any capacity, including the literal linear one, can represent
exactly), **every capacity is expected to land near the EXP-03 floor
(~0.3–0.4 dB), not EXP-02's no-transform number (3.38 dB)**. That's the correct
answer, not a null result — it confirms capacity isn't the bottleneck for a
linearly-structured source, and it confirms the harness (training protocol,
fairness-condition logging, freeze/evaluate machinery) works before it's trusted
on a real, undetermined-answer source.

## Run

```bash
pip install -e ".[neural]"     # once -- adds torch
python experiments/smoke_gaussian_neural/run.py
```

## Reading the output

- `gain(latent)` near 0.3–0.4 dB at every capacity → harness validated, move on
  to the real (non-Gaussian) capacity sweep.
- `gain(latent)` near 3+ dB, or inconsistent across capacities → a fairness
  condition is being violated somewhere, or training hasn't converged. Check
  `bits` (should mirror EXP-03's allocation pattern), `off-diag energy` (should
  be small if the encoder decorrelated), and `participation ratio` (expected to
  sit near 2 throughout — rotation alone doesn't move it for a full-rank Gaussian,
  see the design doc sec 2 / EXP-03's finding).
