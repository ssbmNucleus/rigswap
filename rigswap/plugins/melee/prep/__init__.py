"""prep for Melee: one build made into what the game can load and draw, in place, before anything is compared.

``run`` (the plugin's ``prep``) opens ``<build_dir>/build.blend`` (skinned meshes on a fighter's skeleton, with image
textures), applies the game's limits in this order and saves it. First the copies of a surface the game cannot draw
are dropped: the rendering layers the description names (layers.py), and of one surface drawn twice the copies that
would hide the one drawing it (twins.py: a grey reflection pass over the body, an opaque lining behind glass). Then:

    1. sidedness   apply shape keys; pieces with boundary edges draw both sides, closed pieces cull, and a detail's
                   object takes one mode. Material variants separate the modes for export's PObj flags; no reversed
                   back triangles are added
    2. textures    links from an image the download does not ship are taken out, and what the game never draws of
                   a material (a normal map; an alpha read again from the colour's own picture); a material that
                   draws one flat colour gets it as a one-texel image (the import draws a material's texture only);
                   every image a material uses is made a power of two on each side, at most TEXTURE_MAX_SIDE, and
                   each image the game draws gets the GX format HSDRawViewer will write it in
    3. triangles   a build over the fighter's triangle budget is decimated to it, every mesh by the same share,
                   and the decimated meshes' weights held to the fighter's influence cap again (a collapse mixes them)
    4. memory      the costume file, estimated, must fit the fighter's share of the game's preload memory; the
                   costliest textures are halved until it does
    5. normals     remove collapsed faces and loose vertices; repair invalid normals from faces, neighbours or centre

It writes ``<build_dir>/textures.json``, {material: "FORMAT[:PALETTE]"}, the map HSDRawViewer's ``--model import
--texture-formats`` reads (export passes it), and ``<build_dir>/prep.json``, what each step did, with the numbers.

Each step is a module of its own: textures.py, triangles.py, sidedness.py, layers.py (the rendering layers
dropped before anything is counted), twins.py (a surface's copies dropped, as well), memory.py (with the triangle
palettes) and normals.py. join.py joins meshes by material when the vanilla costume runs out of DObj places for them:
export runs it on its copy of the build (export/write.py: blend_to_glb), when the places are known.

Where the numbers come from: the old importer (Nucleus's app/backend before rigswap; the file is named on each
constant), HSDRawViewer's source (the ssbmNucleus HSDLib fork), and the 25 vanilla round trips under
the prepared targets' roundtrip folder (each fighter's own .glb through ``export``, 2026-10-10).
"""
from __future__ import annotations

import json
from pathlib import Path

from ... import Failed
from .. import DETAILS
from ..facts import parts as melee_parts
from ..facts.memory import read_record
from . import flat
from .layers import drop_rendering_layers
from .sidedness import apply as apply_sidedness
from .memory import budget, check_mesh_palettes, costume_budget, hold_memory_budget
from .normals import repair_normals
from .textures import colour_textures, drawn_only, drop_missing_images, limit_textures, texture_formats
from .twins import drop_twins
from .triangles import TRIANGLE_BUDGET, apply_shape_keys, count_triangles, limit_triangles


# ---- prep ------------------------------------------------------------------------------------------------------------

def run(build_dir, f):
    """Apply the game's limits to ``<build_dir>/build.blend`` for fighter ``f`` (a melee.Fighter), save it, and write
    textures.json and prep.json beside it. Failed, with the numbers, when the build cannot fit; then build.blend is left
    as it was, with no textures.json, and prep.json says why."""
    import bpy
    build_dir = Path(build_dir)
    blend = build_dir / 'build.blend'
    if not blend.is_file():
        raise Failed(f'no build to prep: {blend}')
    (build_dir / 'textures.json').unlink(missing_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    report = {'fighter': f.name}
    try:
        measurement = read_record(f)
        meshes = drawn_meshes()
        if not meshes:
            raise Failed(f'{blend.name} has no mesh with faces')
        report['shape_keys'] = apply_shape_keys(meshes)
        if f.flat:
            report['flat'] = flat.ready(meshes, build_dir)
            meshes = drawn_meshes()
        desc = build_dir.parent / '2-describe' / 'description.json'
        pieces = json.loads(desc.read_text(encoding='utf-8')).get('pieces', []) if desc.is_file() else []
        drawn_by_details = detail_objects(build_dir)
        report['dropped_pieces'], kept_layers = drop_rendering_layers(meshes, pieces, keep=drawn_by_details)
        if kept_layers:         # rendering-layer faces a detail draws (the eyes), left in its objects
            report['rendering_layers_kept'] = kept_layers
        meshes = drawn_meshes()
        if not meshes:
            raise Failed(f'{blend.name} has no mesh left after dropping rendering layers')
        twins = drop_twins(meshes, keep=drawn_by_details)
        if twins['twins'] or twins['linings']:
            report['twins'] = twins
            meshes = drawn_meshes()
        _, replaced = melee_parts.kept_places(f, build_dir)
        if replaced:
            report['weapons_replaced'] = replaced
            report['weapon_forms_hidden'] = list(melee_parts.hidden_forms(f.name, replaced))
        kept = report['kept_parts'] = budget(f, replaced=replaced)
        report['sidedness'] = apply_sidedness(meshes, whole=drawn_by_details)
        missing = drop_missing_images(meshes)
        if missing:
            report['missing_images'] = missing
        drawn = drawn_only(meshes)
        if drawn:
            report['undrawn_taken_out'] = drawn
        report['colours'] = colour_textures(meshes)
        textures, report['textures'] = limit_textures(meshes)
        report['triangles'] = limit_triangles(meshes, max(TRIANGLE_BUDGET, f.triangles) - kept['triangles'],
                                              max_influences=f.max_influences)
        report['palettes'] = check_mesh_palettes(meshes)
        report['memory'] = hold_memory_budget(textures, count_triangles(meshes), costume_budget(f.preloaded),
                                              measurement['kept'] + kept['bytes'], measurement)
        report['textures']['images'] = [t.record() for t in textures]
        report['normals'] = repair_normals(meshes)
    except Failed as e:
        report['failed'] = str(e)
        _write_json(build_dir / 'prep.json', report)
        bpy.ops.wm.read_homefile(use_empty=True)
        raise
    _write_json(build_dir / 'textures.json', texture_formats(textures, drawn_meshes()))
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.read_homefile(use_empty=True)
    _write_json(build_dir / 'prep.json', report)
    return report


def drawn_meshes():
    """The meshes with faces in the open scene: what export writes."""
    import bpy
    return [o for o in bpy.context.scene.objects if o.type == 'MESH' and len(o.data.polygons)]


def detail_objects(build_dir):
    """The objects each detail's build record in ``build_dir`` lists, by name (rigswap.core.eyes; the eyes' are in
    eyes.json). The game draws them as the detail (export/eyes.py: the eyes in the fighter's animated DObjs), so prep
    drops none of their faces as a rendering layer (layers.py: drop_rendering_layers), gives each one sidedness mode
    (sidedness.py) and export puts none of their images in an atlas (join.py: join_for_places)."""
    names = []
    for name in DETAILS:
        made = Path(build_dir) / f'{name}.json'
        if made.is_file():
            names += [o['name'] for o in json.loads(made.read_text(encoding='utf-8')).get('objects', [])]
    return names


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1), encoding='utf-8')
