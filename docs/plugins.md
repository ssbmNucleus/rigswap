# Melee plugin

`rigswap/plugins/melee/` supplies the current pipeline's game data, preparation,
export, and read-back checks. The registry supports `melee`.

- `targets/` prepares fighter skeletons and meshes, bone descriptions, weapon slots,
  camera records, memory estimates, and game animation frames from vanilla files.
- `facts/`, `dat.py`, `dae.py`, and `library.py` hold game facts and file readers.
- `prep/` applies sidedness, textures, triangle and memory limits, and normal repair.
- `export/` writes the costume through HSDRaw, preserves the game's object layout,
  adds blinking eyes and loose-chain physics, and checks the file after reading it back.

The core hands `prep` and `export` a build folder containing `build.blend`,
`target.json`, and optional records for eyes, loose chains, layers, and replaced
weapons. Prep writes `prep.json` and `textures.json`; export writes the costume and
`export/readback.json`. Read-back failures stop export.

Runtime paths are `RIGSWAP_MELEE_FILES` (vanilla fighter files),
`RIGSWAP_MELEE_TARGETS` (prepared targets), `RIGSWAP_HSDRAW` (HSDRawViewer), and
`RIGSWAP_BLENDER4` (Collada conversion). The queue configures these for builds.
See [the workflow](WORKFLOW.md) and [queue setup](../tools/loop/LAPTOP.md).
