"""Regression guard for the ring-claim audit: EXP-06's committed numbers must reproduce
exactly through the current package before and after any additive change."""
from pathlib import Path
import json
import sys

import numpy as np

from codeclab.runctx import load_config
from codeclab import sources
from codeclab.models import train_autoencoder, evaluate_sq_vs_vq

ROOT = Path(__file__).resolve().parents[2]
EXP06 = ROOT / "experiments" / "exp_06_ring_lowcap"
REFERENCE = ROOT / "results" / "exp_06_ring_lowcap" / "results_20260916085107" / "results.json"
KEYS = ("gain_latent_db", "gain_signal_db", "participation_ratio", "off_diag_energy")


def main() -> int:
    cfg = load_config(EXP06 / "config.yaml")
    ref = json.load(open(REFERENCE))
    X = sources.ring_2d(cfg["source"]["n_samples"], radius=cfg["source"]["radius"],
                        radial_noise=cfg["source"]["radial_noise"], seed=cfg["source"]["seed"])
    ae = train_autoencoder(X, capacity=cfg["model"]["capacity"], rate_noise=cfg["train"]["rate_noise"],
                            epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"], seed=cfg["source"]["seed"])
    got = evaluate_sq_vs_vq(ae, X, bits_per_dim=cfg["quant"]["bits_per_dim"])
    ok = True
    for k in KEYS:
        match = np.isclose(got[k], ref[k], rtol=0, atol=1e-9)
        ok &= bool(match)
        print(f"{k:22s} ref={ref[k]:.12f}  got={got[k]:.12f}  {'ok' if match else 'MISMATCH'}")
    print("REGRESSION GUARD:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
