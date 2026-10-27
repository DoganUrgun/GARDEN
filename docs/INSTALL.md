# Installation

Run the commands below from a Linux shell in the repository root. On Windows,
use Ubuntu under WSL2 for CUDA training; the native Windows Python environment
is not the GPU training target.

## Prerequisites

- Linux or WSL2 Ubuntu, with Python 3.11 or 3.12 and its `venv` module.
- An NVIDIA GPU visible to the Linux environment, with a compatible host driver.
- Network access for dependency installation and language-model API calls.
- An API key and account access to the configured model. Access to
  `gpt-6.1-sol` is not guaranteed by this repository.
- Disk capacity for multiple policy checkpoints, raw trajectories, intermediate
  snapshots, source copies, and model interaction records.

Check GPU visibility before installing Python packages:

```bash
nvidia-smi
```

If the command cannot see the GPU, resolve the host/WSL GPU setup first. On WSL2,
the GPU is provided through the Windows host; the Python environment belongs
inside Linux. Consult your platform and NVIDIA installation instructions for
driver setup rather than changing experiment settings to bypass the GPU check.

## Create a clean environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip check
```

The GPU requirements pin `jax[cuda12]==0.4.36` and the experiment's direct Python
dependencies. Use a fresh virtual environment; do not combine this installation
with an unrelated JAX stack. Do not upgrade packages during a registered run.
Manifests record resolved dependency versions, and changing them can invalidate
resume compatibility.

The required JaxMARL environment subset is bundled in `runtime/`; no separate
JaxMARL checkout is required. Its provenance and local adaptations are documented
in [runtime/UPSTREAM.md](../runtime/UPSTREAM.md).

## Configure the model and experiment

```bash
cp .env.example .env
```

Edit `.env` locally and put your key on the `OPENAI_API_KEY=` line. Never commit
or publish `.env`, and do not paste the key into logs or issue reports. Environment
variables already set in the shell take precedence over values loaded from
`.env`.

The configured model/settings are:

```text
OPENAI_REWARD_MODEL=gpt-6.1-sol
REWARD_ENGINEER_REASONING_MODE=pro
REWARD_ENGINEER_REASONING_EFFORT=max
OPENAI_MODEL_VERBOSITY=medium
```

Keep the same model and settings across all three conditions. There is no
automatic fallback. If your account cannot use the configured model, do not
silently substitute another identifier and describe it as the same experimental
configuration. A deliberate alternative must be registered and reported as a
different configuration.

The suite makes paid API calls and runs many full-budget policy-training jobs.
Choose seeds and search replicates before registering it. See
[Running experiments](EXPERIMENTS.md) for the budget and scientific controls.

## Validate the installation

First run non-training checks:

```bash
python main.py
python main.py doctor
python main.py plan
```

The first command only shows help. The doctor should report a GPU and `PASS`;
it checks CUDA availability, not learning quality or model authorization. The
plan explains both paper comparisons, their execution order, and the configured
budget. It does not call the model or train policies. A model identifier in the
plan is not an API-access test.

Run the unit tests on CPU:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```

## Start and return later

The autonomous designer executes generated shell commands and Python. Its
best-effort command and path restrictions are not an operating-system security
sandbox. Run in a dedicated environment without unrelated credentials or private
data. The execution checks do not certify security or reward alignment.

After reviewing the plan, start a fresh controlled suite:

```bash
python main.py --new
```

Run only one suite on the selected GPU at a time. Keep the machine awake and the
process alive; closing the terminal is not a background-execution mechanism.
For an interrupted registered suite, activate the same environment and resume:

```bash
source .venv/bin/activate
python main.py --resume latest
```

Do not use `--new` as a recovery command. Preserve the source tree, environment,
and run directory paths while a suite is active or resumable: manifests and
reference bindings identify the registered files. See
[Running experiments](EXPERIMENTS.md) for exact-suite resume and output locations.

To inspect progress from another terminal without launching work:

```bash
python main.py status
```

Here, `python` runs the Python interpreter, `main.py` is the study's entry point,
and the following word or option chooses its action. `--new` creates a new study;
`--resume latest` continues the most recently registered suite. These commands
must be run from the repository root with its virtual environment activated.
