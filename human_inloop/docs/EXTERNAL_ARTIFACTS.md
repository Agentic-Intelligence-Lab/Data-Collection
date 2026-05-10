# External Artifacts

Large or third-party artifacts are intentionally not committed to this repository.

## OpenPI Source Checkout

Reference used by the local workflow:

- Repository: `git@github.com:1939645507/openpi.git`
- Commit: `edfcb37eb2eaa0472627899edefae6af63850fe3`

Set `OPENPI_ROOT` to that checkout. It must contain:

```text
OPENPI_ROOT/
  src/openpi/
  packages/openpi-client/src/
```

Do not copy OpenPI into this repository unless you have reviewed the license and have a clear reason to vendor it.

## Policy Checkpoint

Set `CHECKPOINT_DIR` to an inference checkpoint directory. Expected layout:

```text
CHECKPOINT_DIR/
  model.safetensors
  metadata.pt
  assets/
    <asset_id>/
      norm_stats.json
```

`NORM_STATS_PATH` is optional when `norm_stats.json` can be resolved from the checkpoint assets directory. Set it explicitly if the stats live outside the release package. Absolute asset paths embedded in checkpoint metadata are not treated as portable defaults.

## Required Files

- `model.safetensors`: model weights used for inference.
- `metadata.pt`: OpenPI training/runtime metadata.
- `assets/.../norm_stats.json`: normalization statistics for the policy.

`optimizer.pt` is usually not needed for inference or data collection. Do not require users to download it unless you also document a training-resume workflow.

## Distribution Recommendations

Do not upload multi-GB checkpoints to a normal GitHub repository. Prefer one of:

- Hugging Face Hub model repository.
- Git LFS with clear bandwidth/storage expectations.
- Institutional artifact storage.
- Private object storage or shared drive for lab-only releases.

## Release Checklist

Fill these fields before a public release:

- Checkpoint URL: TODO: fill in artifact URL before release.
- `model.safetensors` SHA256: TODO: fill in checksum before release.
- `metadata.pt` SHA256: TODO: fill in checksum before release.
- `norm_stats.json` SHA256: TODO: fill in checksum before release.
- OpenPI commit: `edfcb37eb2eaa0472627899edefae6af63850fe3`.
