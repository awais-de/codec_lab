# Ring-claim audit

A separate initiative from the main ladder. EXP-06/07 measured that no trained encoder
reduced VQ's advantage on the ring; the *explanation* recorded there (topology forbids any
seamless encoder from helping SQ) was found to overreach — a hand-built seamless fold of
the ring gives SQ ≈ VQ in latent space. These three experiments pin the cause down so the
claim can be stated in a form that does not depend on any architecture.

Design and reasoning: `docs/design/audit-ring-claims.md` (local).

## Preservation rules
- Nothing under `experiments/exp_*` or `results/exp_*` is modified.
- Package changes are additive only: `codeclab/folds.py`, `codeclab/models/fidelity.py`,
  `codeclab/models/warmstart.py`. No existing function changes.
- `check_regression.py` re-runs EXP-06's config and asserts its committed numbers.
  Run it before and after every package change here.

## Experiments
| ID | folder | question |
|---|---|---|
| AUDIT-01 | `a1_recon_fidelity/` | Are the EXP-06/07 encoders continuous and injective on the ring (measured, not assumed)? |
| AUDIT-02 | `a2_fold_classical/` | Does a seamless encoder exist under which SQ reaches VQ's ceiling, in signal space? |
| AUDIT-03 | `a3_fold_warmstart/` | Is the rate proxy what prevents trained encoders from folding? |

Order: `check_regression.py` → AUDIT-02 → AUDIT-01 → AUDIT-03.
