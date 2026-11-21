# rigswap

rigswap puts a downloaded 3D model onto a game fighter's rig, so the fighter's own animations drive it. The method is
general; the one game it supports today is Super Smash Bros. Melee, through a plugin that turns the result into a
costume `.dat`. [SSBM Nucleus](https://github.com/ssbmNucleus) uses it for its model import.

## The method

[docs/DESIGN.md](docs/DESIGN.md) has the design in full. In one sentence: put the fighter's skeleton inside the model,
in the model's own pose and proportions, let the model's own structure decide which part of the mesh follows which
fighter bone, then move the whole mesh into the fighter's rest pose with one continuous skinning deformation.

- A rigged model brings its own skeleton and skin weights; they are used as they are, renamed through the match.
- A model built from rigid pieces uses its pieces and where they touch as its skeleton.
- The fighter's skeleton is never changed: hitboxes and hurtboxes hang off its bones, so proportions come from
  re-posing the mesh, not from moving joints.
- Questions of meaning (which way is the front, which chain is an arm, a tail or a wing) are asked of a vision model
  on rendered pictures; everything structural is code. Answers are cached.

## Layout

- `rigswap/load/` opens the download (an archive or a model file) in Blender, every model file in it, with its
  textures.
- `rigswap/core/` reads the model's rig or pieces, chooses up, front and the parts that show, matches the body to the
  fighter, deforms and skins the surface, readies the eyes, and settles loose chains. `rigswap/core/run.py` runs one
  swap.
- `rigswap/ai.py` and `rigswap/library/asks.py` send the visual questions (models via OpenRouter) and cache the
  answers.
- `rigswap/plugins/melee/` ([docs/plugins.md](docs/plugins.md)) supplies the fighter targets and game facts, prepares
  the build for the game's limits, exports through HSDRawViewer, and reads the costume back to check its skeleton,
  geometry, materials and layout.
- `tests/` ([tests/README.md](tests/README.md)) holds the tests.

## What it needs

- Python 3.13 with `bpy` 5.2.2 (Blender as a Python module), `numpy`, `pillow` and `requests`:
  `python -m pip install -e .[dev]`.
- A Blender 4.x install for Collada (`.dae`) files, since Blender 5 has no Collada importer: `RIGSWAP_BLENDER4`.
- HSDRawViewer from [SSBM Nucleus's HSDLib fork](https://github.com/ssbmNucleus/HSDLib) (it needs `--model export`
  with `--dobjs` and `--model import`): `RIGSWAP_HSDRAW`.
- The vanilla v1.02 fighter files (`Pl<code>Nr.dat` costumes and `Pl<code>.dat`) from your own Melee disc:
  `RIGSWAP_MELEE_FILES`. This repository contains no game files.
- A folder for the prepared fighter targets: `RIGSWAP_MELEE_TARGETS`. Targets missing there are made from the game
  files on first use.
- An OpenRouter API key in `OPENROUTER_API_KEY` for the visual questions. Alternatively, `RIGSWAP_ASK_RELAY` names a
  folder where each question is written for another program to answer (below).

## Running it

One download into one Melee costume:

```
python -m rigswap.plugins.melee.costume <download> <Fighter|auto> <out> [--cache DIR]
```

`<Fighter>` is the fighter's CamelCase name (`CaptainFalcon`), or `auto` to let rigswap pick the fighter the model
suits. `<out>/result.json` says how it ended and where the `.dat` is. Two follow-up forms change one entry of the
costume's record and build it again:

```
python -m rigswap.plugins.melee.costume --fix <out> "<what looks wrong>"
python -m rigswap.plugins.melee.costume --set <out> head=0.8 [limbs=1.2 ...]
```

The module docstrings (`rigswap/plugins/melee/costume.py`, `rigswap/core/run.py`) describe the steps and outputs.

## How SSBM Nucleus uses it

SSBM Nucleus ships a pinned commit of rigswap with its app and runs `rigswap.plugins.melee.costume` as a subprocess
when a user imports a model. When the import is driven through the app's MCP server, the visual questions go through
`RIGSWAP_ASK_RELAY` to the user's own AI instead of OpenRouter.

## Development

`tests/README.md` covers the test markers and environments; `pytest -m fast` needs no Blender or game files.
`tools/loop/` is the development harness that builds a set of swaps and captures them in Dolphin; it is set up for
the maintainers' machines.

## Licence

GNU General Public License v3.0; see `LICENSE`.
