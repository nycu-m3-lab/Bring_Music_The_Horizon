# Conda environment setup

This project uses multiple Conda environments because different pipeline components have incompatible or heavy dependency groups.

Component mapping:

- `Component_0_args_saver.py`, `Component_1_allinone.py`, `Component_5_360DynamicAndTransitionGenerator.py`, `Component_5.5_NormalizeClips16x9.py`, `Component_6_Concatenator.py`, `Component_7_MusicAdder.py`: `final_01567`
- `Component_2_VAExtractor.py`: `final_2`
- `Component_3_Residual_Calculator.py`: `final_3`
- `Component_4_360ImageGenerator.py`: `final_4`
- `worker_allinone.py`: `final_allinone`

Install everything from the repository root:

```bash
bash scripts/setup_conda_envs.sh --yes
```

If you do not have sudo access or already installed system packages:

```bash
bash scripts/setup_conda_envs.sh --yes --skip-apt
```

The `final_allinone` env intentionally installs PyTorch, the checked-in local NATTEN wheel at `../big_files/natten-0.17.5+torch250cu121-cp310-cp310-linux_x86_64.whl`, `madmom`, and `all-in-one` with the project-specific pip commands. Do not replace that part with a Conda `allin1` install or an upstream NATTEN download URL.

Blackwell GPUs such as `sm_120` require PyTorch wheels built with CUDA 12.8 or newer. The `final_2`, `final_3`, and `final_4` env files use CUDA 12.8 wheels for that reason. Keep `final_allinone` on its pinned PyTorch 2.5.0 CUDA 12.1 stack unless you provide a matching NATTEN wheel.

Place the prepared SDXL base model directory at `../big_files/sdxl_model` by copying the shared `big_files/` folder. There is no SDXL download/install step in the environment setup script. The pipeline loads SDXL with `variant="fp16"`, so the shared folder should contain only the `*.fp16.safetensors` weights, not duplicate plain `.safetensors` files.

Remove the project environments:

```bash
bash scripts/remove_conda_envs.sh
```

Use `--yes` for non-interactive removal, and `--clean-cache` if you also want Conda package caches cleaned.
