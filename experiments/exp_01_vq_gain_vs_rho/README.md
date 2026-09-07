# EXP-01: VQ gain vs source correlation (ρ sweep)

Tracks **[#1](https://github.com/awais-de/codec_lab/issues/1)** · Stage: Reference · Priority: P0

## Question
How much of VQ's advantage over optimal scalar quantization depends on how
correlated the source is?

## Hypothesis
Classical prediction: VQ gain grows with ρ, because VQ exploits inter-dimension
correlation that SQ ignores. A monotonically rising VQ-gain-vs-ρ curve confirms
the mechanism the whole thesis rests on; a flat curve refutes it.

## Setup
- Source: 2D Gaussian, unit variances, correlation ρ swept over `config.yaml`
- SQ: optimal Lloyd-Max, `bits_per_dim` bits on each axis
- VQ: LBG, codebook rate-matched to the same total bits (`2**(bits_per_dim * 2)`)
- Metric: SQNR (dB) for each quantizer; VQ gain = SQNR_VQ − SQNR_SQ

## Run
```bash
pip install -e .          # once, from repo root
python experiments/exp_01_vq_gain_vs_rho/run.py
```
Output lands in `results/exp_01_vq_gain_vs_rho/results_<timestamp>/`:
`vq_gain_vs_rho.png`, `metrics.csv`, `config.snapshot.yaml`, `meta.json`.

## Definition of done
- [ ] `run.py` sweep implemented
- [ ] plot + `metrics.csv` generated and committed
- [ ] one-paragraph interpretation added below
- [ ] ρ=0.85 operating point confirmed for EXP-02 / EXP-05

## Result
_(fill in after the run: shape of the curve, does it match the classical prediction, the ρ=0.85 number)_
