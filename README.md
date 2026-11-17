# HBO-GMR

Code and pretrained models for **HBO-GMR: Bayesian tuning and learned geological correction for ensemble-based reservoir history matching**.

## Setup

Tested with Python 3.12.

- `requirements-model.txt`: NumPy and PyTorch for model inference and training
- `requirements-science.txt`: full scientific stack (SciPy, scikit-learn, scikit-optimize, pandas, matplotlib, etc.)

```sh
pip install -r requirements-science.txt
```

## Quick start

All four offline examples run without an external simulator and write outputs to `outputs/` by default:

```sh
python examples/reproduce_metrics.py
python examples/geological_correction.py --members 1 --device cpu
python examples/schedule_and_acquisition.py
python examples/read_synthetic_rsm.py
```

## Structure

- `examples/`: standalone scripts covering benchmark metric reproduction, 3D Attention VAE + RealNVP geological correction inference, bounded-simplex / constrained EI acquisition, and RSM parsing.
- `simulation_snapshot/gmr_refactor/`: ES-MDA history matching, APBO schedule optimization, and GMR regularization pipeline (`main_gmr_v1.py`, `train_gmr_vae.py`, `train_gmr_flow.py`, `core/`, `src/`, `config/`, `checkpoints/`, `data/`).
- `inputs/`: observation arrays and prior ensemble numeric exports.
- `results/` & `figure_sources/`: Table 2 / confirmatory benchmark CSVs and Figure 6D source data.

> Note: End-to-end history matching via `simulation_snapshot/gmr_refactor/main_gmr_v1.py` requires an external licensed ECLIPSE simulator and reservoir simulation decks (not included).

## Citation

If you use this code in your research, please cite the metadata in [CITATION.cff](CITATION.cff).

## License

Source code is released under the [MIT License](LICENSE). Model weights, training patches, and numerical result exports are licensed under [CC BY 4.0](LICENSE-DATA.md).
