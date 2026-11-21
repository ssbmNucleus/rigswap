"""The Melee plugin: the fighters as targets, with their test poses and the game camera, and the costume .dat export.

The core calls these entry points, which hand on to the folders that do the work:

    targets       ``target``, the fighter the user names, and ``targets``, every prepared fighter, for choosing among
                  them; made once from the vanilla costumes by ``prepare`` and ``describe_targets``
                  (``describe_from_tables`` for one fighter): targets/
    test poses and camera
                  each fighter's ``<Name>.pose-*.json`` and the game camera, ``<Name>.camera.json`` (targets/camera.py),
                  which ``target`` reads with it
    prep          the game's limits on one build: prep/
    export        the costume .dat written into the vanilla file and read back: export/. It preserves the game's
                  skeleton, object layout, materials and textures, adds eyes and loose-chain accessories, and
                  records every read-back check in readback.json.
    the details it supports
                  DETAILS: the fighters blink; export/ makes each detail the game's (export/eyes.py)

The folders:

    facts/        the tables, each with where it was read or measured: the fighters, the parts a costume keeps, the
                  chains, the materials' alpha
    dat.py        the .dat file read and written: Dat, the joint tree, the rest skeleton
    dae.py        Collada read the same way whoever wrote it (read_dae)
    targets/      making a target from a vanilla costume: prepare.py, camera.py, bones.py, weapons.py
    prep/         the limits applied inside a build, in order (``run``)
    export/       writing the costume and reading it back (``run``): write.py, dobjs.py, eyes.py, checks.py

Where things are (environment variables, or pass the folder as an argument):

    RIGSWAP_MELEE_FILES     the folder holding the game's vanilla fighter files (Pl<code>Nr.dat costumes and Pl<code>.dat
                            fighter data) as a vanilla v1.02 disc has them, not a modded project's; there is no
                            default
    RIGSWAP_MELEE_TARGETS   where ``prepare`` writes the targets and ``target`` reads them
    RIGSWAP_HSDRAW          HSDRawViewer.exe, the HSDLib fork with ``--model export`` and ``--model import``
    RIGSWAP_BLENDER4        a Blender 4.x, for Collada both ways (see rigswap.load.convert)

What ``prepare`` writes for each fighter, under the targets folder:

    <Name>.glb              the vanilla costume, every joint (named JOBJ_<n>, the joint's place in the costume's
                            joint tree, in that order, each with its own axes) and the meshes the game draws in play
                            when the fighter spawns (the high-detail DObjs, see export/write.py: ``export_dae``), in the
                            game's model space: y up, the fighter facing +z. Each joint is where the mesh was bound to
                            it, which for a few joints of most fighters is up to 0.08 units off the skeleton's rest
                            (dat.py: ``dat_joints``); the costume export binds the new mesh at the rest.
    <Name>.camera.json      the game camera for this fighter (see targets/camera.py)
    <Name>.memory.json      importer round-trip storage rates and provenance (facts/memory.py); required by prep
    vanilla/<Name>/         those DObjs as HSDRawViewer exports them: <Name>.dae, its textures and texture map
    frames/<Name>/          the fighter's own animation frames,
                            <action>-<frame>.json (targets/frames.py), each joint's world matrix as the game has it

``target`` also reads ``<Name>.description.json`` and ``<Name>.weapons.json``, written by
``describe_from_tables`` from the game's tables and vanilla costume, and any ``<Name>.pose-*.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .. import Failed, description_path, read_description

FILES_VAR = 'RIGSWAP_MELEE_FILES'
TARGETS_VAR = 'RIGSWAP_MELEE_TARGETS'
# The details the fighters have (docs/pipeline.html, Details): they blink (export/eyes.py), and one is flat
# (export/flat.py). A detail's record is <name>.json beside build.blend, from the core (the eyes) or prep (flat); the
# module of that name in export/ makes it the game's (its apply).
DETAILS = ('eyes', 'flat')


def _folder(given, var, what):
    path = given or os.environ.get(var)
    if not path:
        raise Failed(f'where are the {what}? pass the folder or set {var}')
    path = Path(path)
    if not path.is_dir():
        raise Failed(f'the {what} folder does not exist: {path}')
    return path


# The plugin's folders are imported here, after everything they read of this module is defined, and before the
# functions below: three of those functions, ``targets``, ``prep`` and ``export`` (plugins/__init__.py), have the
# names of folders, and a folder imported after its function would rebind the package's name for it. Reach a folder's
# modules by their whole names (``from rigswap.plugins.melee.prep import textures``), never as attributes of this
# package (``melee.prep`` is the function).
from .facts.fighters import _build_fighter, fighter, in_scope                        # noqa: E402
from .facts.memory import MeleeTarget, read_record                                  # noqa: E402
from .targets import bones as _bones, prepare as _prepare                          # noqa: E402
from .targets.weapons import read_weapons                                            # noqa: E402
from .prep import run as _prep                                                       # noqa: E402
from .export import run as _export                       # noqa: E402
from .export.dobjs import HIGH_DETAIL, fighter_tables                               # noqa: E402


# ---- the targets --------------------------------------------------------------------------------------------------

def _targets_dir(targets_dir):
    return _folder(targets_dir, TARGETS_VAR, 'Melee targets')


def target(name, targets_dir=None):
    """The fighter called ``name`` as a Target: its .glb as skeleton and original, its description, its test poses,
    the game camera, its weapon slots (none when describe-targets measured none) and its item joints
    (bones.item_joints). Failed when it has not been prepared or described yet, or its importer memory record is
    missing (rerun ``prepare``)."""
    f = fighter(name)
    folder = _targets_dir(targets_dir)
    glb, cam = folder / f'{f.name}.glb', folder / f'{f.name}.camera.json'
    if not glb.is_file() or not cam.is_file():
        raise Failed(f'{f.name} has not been prepared: no {glb.name} or {cam.name} in {folder}; run melee.prepare')
    physics_path = folder / f'{f.name}.physics.json'
    physics = json.loads(physics_path.read_text(encoding='utf-8')) if physics_path.is_file() else {}
    return MeleeTarget(name=f.name, skeleton=glb, original=glb, description=read_description(glb),
                  poses=sorted(folder.glob(f'{f.name}.pose-*.json')),
                  camera=json.loads(cam.read_text(encoding='utf-8')), max_influences=f.max_influences,
                  weapons=read_weapons(glb), item_joints=_bones.item_joints(f.name), memory=read_record(f, folder),
                  physics=physics)


def drawn(name, meshes, files_dir=None):
    """Of a flat fighter's target ``meshes`` (objects named Joint_<joint>_Object_<place>), the ones the game draws of
    him at spawn, option 0 of each model in Fighter.shown: a target's file holds his outline hull and all his forms too,
    which show nowhere. None for a fighter who isn't flat (facts/flat.py)."""
    f = fighter(name)
    if not f.flat:
        return None
    tables = fighter_tables(_folder(files_dir, FILES_VAR, 'vanilla game files') / f'Pl{f.code}.dat')
    places = {p for m in f.shown for p in tables.options[HIGH_DETAIL, m, 0]}
    return [o for o in meshes if int(o.name.split('_')[3]) in places]


def targets(targets_dir=None):
    """Every prepared fighter as a Target, for choosing among them. Failed naming every prepared fighter that has no
    description yet."""
    folder = _targets_dir(targets_dir)
    prepared = [f for f in in_scope() if (folder / f'{f.name}.glb').is_file()]
    undescribed = [f.name for f in prepared if not description_path(folder / f'{f.name}.glb').is_file()]
    if undescribed:
        raise Failed(f'{len(undescribed)} of {len(prepared)} prepared fighters must be described first: '
                     + ', '.join(undescribed))
    return [target(f.name, folder) for f in prepared]


def prepare(files_dir=None, out_dir=None, names=None, refresh_memory=False):
    """Make every in-scope fighter's target, or the fighters ``names``, from the vanilla costume, into ``out_dir``
    (else RIGSWAP_MELEE_TARGETS): targets/prepare.py. ``refresh_memory`` regenerates memory records for existing
    targets. Returns {name: 'made' | 'already there' | why it failed}."""
    return _prepare.prepare(files_dir, out_dir, names, refresh_memory=refresh_memory)


def describe_from_tables(name, targets_dir=None, files_dir=None):
    """Write <Name>.description.json, which ``target`` reads, from the game's own bone tables instead of the AI; the
    AI's is kept as <Name>.description.ai.json (targets/bones.py). Returns the Description."""
    return _bones.describe_from_tables(name, targets_dir, files_dir)


def describe_targets(targets_dir=None, files_dir=None):
    """describe_from_tables for every prepared fighter (``rigswap describe-targets <folder> --plugin melee``)."""
    return _bones.describe_targets(targets_dir, files_dir)


# ---- prep and export ----------------------------------------------------------------------------------------------

def prep(build_dir, fighter_name=None):
    """Fit one build to the game, in place: ``<build_dir>/build.blend`` gets the game's texture sizes, triangle budget,
    back faces for thin layers, memory budget and no normal without a direction; ``textures.json`` beside it has each
    material's texture format for export, ``prep.json`` what was done, with the numbers (see prep/). Failed when
    the build cannot fit. The fighter is ``fighter_name``, else the one ``<build_dir>/target.json`` names."""
    build_dir = Path(build_dir).resolve()
    return _prep(build_dir, _build_fighter(build_dir, fighter_name))


def export(build_dir, out, fighter_name=None, files_dir=None):
    """Write the costume for ``<build_dir>/build.blend`` into the fighter's vanilla Pl<code>Nr.dat as ``out`` and read
    it back: export/ (``run``). Failed when the read-back differs. The fighter is ``fighter_name``, else the one
    ``<build_dir>/target.json`` names; the game files are in ``files_dir``, else RIGSWAP_MELEE_FILES."""
    return _export(build_dir, out, fighter_name, files_dir)


def owns(build_dir):
    """Whether the records in ``build_dir`` are this plugin's: its prep ran there, so ``prep.json`` names the fighter
    that ``target.json`` does. rigswap check asks it of an older run that recorded no plugin."""
    build_dir = Path(build_dir)
    prepped, named = build_dir / 'prep.json', build_dir / 'target.json'
    if not prepped.is_file() or not named.is_file():
        return False
    prep_record = json.loads(prepped.read_text(encoding='utf-8'))
    return prep_record.get('fighter') == json.loads(named.read_text(encoding='utf-8'))['name']
