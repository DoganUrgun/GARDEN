# Third-party notices

Bundled third-party materials retain their own licenses. This notice does not
relicense them. Preserve their license texts and applicable source notices when
redistributing the bundled runtime or substantial portions of it.

## JaxMARL

- Project: [FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL).
- Included material: the required Overcooked V1 and SMAX environment subset and
  shared environment/space definitions under `runtime/jaxmarl/`.
- License: Apache License 2.0, retained in
  [runtime/JAXMARL_LICENSE](runtime/JAXMARL_LICENSE).
- Local adaptations and available source provenance:
  [runtime/UPSTREAM.md](runtime/UPSTREAM.md).

## Overcooked-AI layout

- Project: [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai).
- Included material: adapted Bottleneck geometry in
  `runtime/jaxmarl/environments/overcooked/layouts.py`.
- Copyright notice: Copyright (c) 2019 Center for Human-Compatible AI.
- License: MIT, retained in
  [runtime/OVERCOOKED_AI_LICENSE](runtime/OVERCOOKED_AI_LICENSE).

The layout adaptation changes symbols to the JaxMARL representation and uses
JaxMARL's cooking dynamics. It is not a full distribution of Overcooked-AI.

## Installed dependencies

Packages installed from `requirements-gpu.txt` or `requirements.txt` are separate
dependencies with their own license terms. Their distributions carry those
notices. The dependency files identify the direct packages used by this project;
run manifests record resolved runtime dependency versions.
