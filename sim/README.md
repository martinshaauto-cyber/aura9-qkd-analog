# sim — analog CV-QKD simulation

This folder contains a minimal continuous-variable (analog) QKD simulator.

Usage (from repository root):

```bash
python sim/analog_qkd.py --samples 100000 --transmittance 0.6 --excess-noise 0.01
```

Key points:
- Very small educational simulator using Gaussian assumptions.
- Estimates mutual informations from sample correlations and returns a
  simplified reverse-reconciliation key-rate estimate: `K = beta * I(A;B) - I(B;E)`.

See `sim/analog_qkd.py` for parameters and implementation notes.
