# HBO-GMR user guide

This guide covers the code, inputs, and model checkpoints at tag [v1.0.0](https://github.com/xiaoxiaomeng044-beep/HBO-GMR/tree/f8d6aab305e19a24b9236875a570d4799aa93d65), commit `f8d6aab305e19a24b9236875a570d4799aa93d65`. Start with the numerical-summary example to check the reported comparisons, then use the other offline examples to explore the algorithm components.

## Installation and computing requirements

Clone the repository and run the commands from its root:

```sh
git clone https://github.com/xiaoxiaomeng044-beep/HBO-GMR.git
cd HBO-GMR
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in a Bash-compatible shell. To run only the numerical-summary example, Python's standard library is sufficient. For the other offline examples and the configuration check, install the pinned scientific dependencies:

```sh
python -m pip install -r requirements-science.txt
```

For geological correction and the RSM reader alone, use `requirements-model.txt` instead. For a CPU build on a platform supported by PyTorch's CPU wheel index, install `torch==2.8.0` from that index before installing either requirements file:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

The four offline examples were checked on Windows x64 with Python 3.12.14 and the pinned dependencies, including NumPy 1.26.4, SciPy 1.13.0, and PyTorch 2.8.0+cpu. The default geological example processes one member on two CPU threads. No GPU, ECLIPSE installation, or network access is needed to run the examples after setup. A compatible CUDA-enabled PyTorch build is needed to use `--device cuda`. The checkout contains about 18 MiB of tracked files. Allow additional space for Python packages and any simulation outputs. Peak memory and full simulation hardware requirements were not measured in this check.

## Offline examples

Run these commands from the repository root:

```sh
python examples/reproduce_metrics.py
python examples/geological_correction.py --members 1 --device cpu
python examples/schedule_and_acquisition.py
python examples/read_synthetic_rsm.py
```

All four commands print a JSON summary with `"checks": "PASS"` on success. They read their inputs relative to the repository and write below `outputs/`. The first three accept `--output-dir PATH` for an alternative location. The RSM example writes to its fixed output directory. Runs with the same output path replace files of the same name.

### Recompute Table 2 and Figure 6D values

`reproduce_metrics.py` reads 20 saved terminal records in [confirmatory_metrics.csv](../results/confirmatory_metrics.csv) and checks its calculations against [Table 2](../results/table2_recomputed.csv) and the [Figure 6D source data](../figure_sources/figure6_panelD_HBO_GMR_vs_alphaBO.csv). The comparison pairs four methods with five evaluation seeds: `544826`, `119942`, `673063`, `251642`, and `129744`.

- The method labels `standard`, `apbo`, `gmr`, and `full` denote equal-inflation ES-MDA, inflation tuning alone, equal inflation with GMR, and HBO-GMR, respectively.
- `m_prod` measures ensemble-mean response misfit. The historical CSV label `m_prod_ensemble_mean` denotes the mean of the member-wise misfits. `geo_score` measures geostatistical deviation from the prior.
- `outputs/metrics/table2.csv` contains geometric means by method. The `full` row is approximately `0.162938`, `0.171783`, and `6.560548` in the order above.
- `outputs/metrics/relative_changes.csv` gives percentage changes relative to standard ES-MDA, with negative changes denoting reductions. The separate Figure 6D check compares `full` with `apbo` using positive values for improvements. `report.json` summarizes the checks and Table 2 results.

The example checks summaries from frozen results. It does not regenerate reservoir responses, bootstrap intervals, or the full figure.

### Apply the supplied geological correction

`geological_correction.py` reads horizontal permeability, porosity, and vertical permeability from [the numeric prior ensemble](../inputs/prior_ensemble_numeric.npz). Its arrays `mumyperm`, `mumyporo`, and `mumypermz` have shape `(2660, 200)`. Cells of the `19 × 28 × 5` grid are flattened in Fortran order. The supplied dataset metadata defines patch locations and normalization for log10 horizontal permeability and porosity. The example loads the published attention VAE and RealNVP weights and applies the first coefficient of the frozen four-round correction sequence to the selected prior members.

- `--members` selects the first 1 to 200 realizations. Start with `--members 1 --device cpu` before increasing the ensemble size.
- `outputs/geological_correction/corrected_fields.npz` contains finite arrays `permx` and `poro`, each with shape `(2660, members)`.
- `report.json` gives the blending coefficient, field changes, and diagnostics. The one-member CPU check returned shape `[2660, 1]` and blending coefficient approximately `0.388039`. `runtime_dataset_meta.json` records the resolved input path for that run.

This example demonstrates inference on prior fields. It uses the supplied weights and does not run production assimilation or training.

### Inspect the inflation schedule and acquisition function

`schedule_and_acquisition.py` reads the frozen equal-inflation and tuned schedule JSON files in [config](../simulation_snapshot/gmr_refactor/config). It checks that the inverse inflation factors sum to one, validates 32 feasible Sobol candidates, and evaluates constrained expected improvement for three synthetic Gaussian-process predictions. `outputs/schedule/report.json` records both schedules, the candidate count, and the acquisition scores. The synthetic example selects candidate index `0` using zero-based indexing. It does not repeat the manuscript's search or call ECLIPSE.

### Read a synthetic simulation response

`read_synthetic_rsm.py` creates a temporary two-well RSM file with three time steps and checks the values returned by [the response reader](../simulation_snapshot/gmr_refactor/core/getsimulated.py). It expects shape `[3, 7, 1]` for time steps, response channels, and members. The seven channels are field oil efficiency, two well bottom-hole pressures, two well gas–oil ratios, and two well water cuts. `outputs/rsm_reader/report.json` records the result. The temporary RSM is generated by the script, so no input file or simulator is needed.

## Check the frozen configuration for the full method

With the scientific dependencies installed, run this command from the repository root. It validates the released configuration for one evaluation seed and exits before reading simulator inputs.

```sh
python simulation_snapshot/gmr_refactor/main_gmr_v1.py --exp fullMethod --grid_config simulation_snapshot/gmr_refactor/config/grid_punq.json --logic paper --seed 119942 --production_metric m_prod --apbo_schedule_mode terminal_simplex --alpha_schedule_file simulation_snapshot/gmr_refactor/config/stage_c2_apbo_c010_frozen.json --ensemble_fidelity_manifest simulation_snapshot/gmr_refactor/config/stage_c2_fidelity_seed2026.json --ensemble_fidelity_level L3 --ae_constraint_form gmr_direction --gmr_global_guard 0 --gmr_forward_guard 1 --gmr_forward_guard_policy terminal_sequence --gmr_terminal_sequence_file simulation_snapshot/gmr_refactor/config/gmr_lambda_e011_frozen.json --gmr_eval_legacy_phi 0 --validate_stage_e2_wiring_only 1
```

Expect `STAGE_E2_TERMINAL_GMR_SEQUENCE_VALID` and a JSON configuration with four assimilation rounds, 200 selected members, and the frozen inflation and blending schedules. The manifest defines screening levels L1, L2, and L3 with 20, 50, and 200 members. Use `--help` to see the runner's other options and defaults. The command above checks the full method's settings and exits without running a reservoir simulation.

## Prepare a simulator-backed run

The full runner needs a separately installed, licensed ECLIPSE simulator and authorized PUNQ-S3 inputs. [Coventry University's PUNQ project page](https://www.coventry.ac.uk/research/research-directories/current-projects/2019/petroleum-reservoir-engineering-simulation-models/) describes the benchmark. The repository provides numerical observations and prior exports for inspecting the case and running offline examples. The original simulator decks, their include files, and reference model files must be obtained separately. The project page identifies the benchmark and does not guarantee access to the exact case files.

1. Copy `simulation_snapshot/gmr_refactor` into a separate working directory as `gmr_refactor`. Run the simulator workflow from that working directory. Ensure `eclrun` is on the Windows PATH and that ECLIPSE can launch your licensed case. The adapter calls `eclrun eclipse MASTER0.DATA` from each member directory.
2. Put the case inputs in the working directory. The default case expects `sgsim.out`, `sgsimporo.out`, and `sgz.out` as text arrays of the 200 prior fields in Fortran order, along with `effective.out` for the cell mask, `timestep.txt` for the time grid, and the observation summary `Real.RSM`. It also reads the numeric reference fields `rossmary.GRDECL`, `rossmaryporo.GRDECL`, and `rossmaryz2.GRDECL`. See the released [grid configuration](../simulation_snapshot/gmr_refactor/config/grid_punq.json) for dimensions and response channels. The runner reads these filenames directly, so a CSV or NPZ export is not a drop-in replacement.
3. Prepare the member directories `MASTER00001` through `MASTER00200` with `MASTER0.DATA` and the includes required by the authorized case. The runner writes `PERMX2.GRDECL`, `PORO2.GRDECL`, and `PERMZ2.GRDECL` there and reads each member's `MASTER0.RSM` after simulation. Check deck paths before starting the ensemble.
4. To use the full-method settings above, replace each `simulation_snapshot/gmr_refactor/` prefix with `gmr_refactor/` in the command to match the scratch layout. Remove `--validate_stage_e2_wiring_only 1` when ready to start simulation. Use a fresh output location and run ID, for example `--out_root results_new --run_id full_seed119942 --existing_run_policy error --compact_result_path 1`. Keep the frozen schedule, sequence, fidelity manifest, and seed explicit for comparisons. Inspect the optimization log CSV, experiment configuration and run-summary JSON, and saved iteration states in the path reported by the runner.

The configuration check and offline examples are the validation paths exercised for this release. A new simulator-backed run depends on access to the licensed software and original case files. The full simulation and retraining were not rerun during this check.

## Training and data reference

- [Model checkpoints and training metadata](../simulation_snapshot/gmr_refactor/checkpoints) include the two weight files and their recorded settings and histories. For retraining, use `train_gmr_vae.py --help` and `train_gmr_flow.py --help` under `simulation_snapshot/gmr_refactor/`. Pass the supplied dataset paths and an explicit `--out_dir` outside the distributed checkpoints. Train the VAE first, then pass its checkpoint with `--vae_ckpt` when training the flow. Use the recorded settings in the metadata to compare with the supplied weights, as the command defaults differ.
- [Inputs](../inputs) contains numerical observations and prior fields. [Training data](../simulation_snapshot/gmr_refactor/data) contains the normalized prior patches and metadata. [Results](../results) and [figure sources](../figure_sources) contain the saved terminal comparisons and Figure 6D values.
- [MIT](../LICENSE) covers the code and documentation. [CC BY 4.0](../LICENSE-DATA.md) covers the distributed weights and derived research materials. Cite the project using [CITATION.cff](../CITATION.cff) and the code version used.
