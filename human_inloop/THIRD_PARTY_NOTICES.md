# Third-Party Notices

This repository contains an interactive correction / DAgger data-collection workflow adapted from local robotics code derived from the open-source Evo-RL project:

- Upstream: https://github.com/MINT-SJTU/Evo-RL
- License: Apache License 2.0

The bundled `vendor/lerobot_piper/lerobot` runtime package and the Piper/OpenPI integration code are included to make the collection workflow easier to reproduce. Keep the Apache-2.0 license and this notice when redistributing modified versions.

OpenPI is treated as an external dependency and is not vendored here. The local workflow was tested against `git@github.com:1939645507/openpi.git` at commit `edfcb37eb2eaa0472627899edefae6af63850fe3`.

Large artifacts are intentionally not included:

- OpenPI source checkout (`OPENPI_ROOT`)
- policy checkpoints (`CHECKPOINT_DIR`)
- recorded datasets (`COLLECTION_OUTPUT_ROOT`)
