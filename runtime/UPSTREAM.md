# Vendored environment runtime

`jaxmarl/` contains the source subset required by the experiment from a JaxMARL
snapshot reporting version `0.2.0`. The exact upstream commit is not recorded
here; registered runs identify their bundled source through file hashes.

Upstream project: [FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL).
The upstream Apache-2.0 license is retained in [JAXMARL_LICENSE](JAXMARL_LICENSE).

The subset includes classic Overcooked V1, SMAX, and shared environment/space
definitions. Package initializers expose only the required families. Overcooked
V2 and unrelated environment families are not included. The policy learner is
implemented in the repository's `common/mappo.py`; the vendored subset provides
environment dynamics rather than a separate training dependency.

## Local adaptations

In `jaxmarl/environments/smax/smax_env.py`, `get_obs_unit_list` constructs the
same ordered unit-list observation features with broadcasting instead of nested
`vmap` calls. This avoids the CUDA layout failure encountered with an outer
parallel-environment `vmap` on asymmetric teams. CPU checks do not establish GPU
execution. `python main.py doctor` verifies CUDA availability; actual environment
and learner execution must also be checked in the study's training logs.

`jaxmarl/environments/overcooked/layouts.py` includes the 7-by-5 Bottleneck
geometry adapted from
[Overcooked-AI's Bottleneck layout](https://github.com/HumanCompatibleAI/overcooked_ai/blob/master/src/overcooked_ai_py/data/layouts/bottleneck.layout).
Its symbols are translated from `X/D/S/1/2` to `W/B/X/A/A`. The geometry uses
JaxMARL's three-onion, automatic-cooking, +20-delivery dynamics; it is not an
exact reproduction of the complete Overcooked-AI environment. The source
project's MIT notice is retained in
[OVERCOOKED_AI_LICENSE](OVERCOOKED_AI_LICENSE).

Task-reward separation, standalone reward interfaces, and the SMAX episode
horizon correction are implemented in the family adapters. The Overcooked
adapter and CNN use the actual height-by-width-by-channel observation dimensions
despite the upstream width/height metadata mismatch. The runtime's internal
observation-shape field is unchanged. These adapter/network corrections do not
replace combat or cooking transition dynamics.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the bundled component
notices and [PROTOCOL.md](../PROTOCOL.md) for the common experimental contract.
