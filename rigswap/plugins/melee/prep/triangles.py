"""2. triangles: every mesh's shape keys applied as they are mixed and removed (Decimate can't be applied over them),
then a build over the fighter's triangle budget decimated to it, every mesh by the same share (limit_triangles). A
face that will get a back (layers.py: a thin layer's, or one the model draws from behind) counts twice."""
from __future__ import annotations

from .weights import cap_groups

import numpy as np

from ... import Failed
from .layers import _as_layers, back_faces


# The triangles a costume may draw: TRIANGLE_BUDGET, or the fighter's own count when it draws more (the table's
# ``triangles`` column; none does: Game & Watch draws the most, 8,286). 9000 is the old importer's budget (randall
# model_import/reduction.py TRIANGLES, the rig tool's default since June 2026), not a limit measured in the game;
# vanilla fighters draw 2,622 (Kirby) to 8,286 in play, most about 5,000 (the vanilla v1.02 files, 2026-10-10). The
# low-detail DObjs draw the same triangles again for shadows and the magnifier, from the same data (export/dobjs.py).
TRIANGLE_BUDGET = 9000
# A triangle with twice its area this or less draws nothing, and HSDRawViewer's import drops the run of triangles
# around it as well, so the ones decimation leaves are removed. (shared/modellab/native_import.py DEGENERATE_AREA)
DEGENERATE_AREA = 1e-12


# ---- 2. triangles ----------------------------------------------------------------------------------------------------

def apply_shape_keys(meshes):
    """Each mesh's shape keys applied as they are mixed now and then removed, so the mesh is the shape the build shows
    (keys at rest: the basis's). The build has no use for them: nothing after describe moves a shape key, and the
    costume the import writes has no morphs. Blender's Decimate modifier (the triangle budget, next) cannot be applied
    to a mesh that has them. Every prep that removes shape keys does it here: Melee's, Brawl's, and VRM's when it
    decimates.

    Through the data, whatever the object's state, not bpy.ops.object.shape_key_remove: that operator's poll needs the
    object to have an active shape key, and Blender's glTF importer leaves an object with none ("poll() failed,
    context is incorrect"). The mix is Blender's own
    (Object.shape_key_add from_mix: the evaluation the operator's apply_mix writes into the mesh).
    -> {mesh name: the keys it had}"""
    record = {}
    for o in meshes:
        if o.data.shape_keys is None:
            continue
        if o.data.users > 1:
            o.data = o.data.copy()
        record[o.name] = [k.name for k in o.data.shape_keys.key_blocks]
        mix = np.empty(len(o.data.vertices) * 3, np.float32)
        o.shape_key_add(name='prep: the mix', from_mix=True).data.foreach_get('co', mix)
        o.shape_key_clear()
        o.data.vertices.foreach_set('co', mix)
        o.data.update()
        if o.data.shape_keys is not None:
            raise Failed(f'{o.name}: its shape keys could not be applied')
    return record


def count_triangles(meshes):
    return sum(_triangles(o) for o in meshes)


def _triangles(o):
    """The triangles a mesh is written as (each n-sided face as n - 2)."""
    totals = np.empty(len(o.data.polygons), np.int64)
    o.data.polygons.foreach_get('loop_total', totals)
    return int((totals - 2).sum())


def _drawn(o, layers):
    """The triangles mesh ``o`` is drawn with once the faces that get a back (layers.back_faces: its layer faces,
    ``layers`` as read_layers gives them, and the faces marked as drawn from behind) have it."""
    totals = np.empty(len(o.data.polygons), np.int64)
    o.data.polygons.foreach_get('loop_total', totals)
    return int((totals - 2).sum() + (totals - 2)[back_faces(o, layers)].sum())


def limit_triangles(meshes, budget, doubled=(), max_influences=None):
    """Decimate ``meshes`` (Blender's Decimate modifier, collapse: UVs, vertex weights and face attributes are carried
    over) so the triangles drawn are at most ``budget``, every mesh by the same share. The faces that will get a
    back layer (step 3: layers.back_faces, the layers in ``doubled`` as read_layers gives them or a list of mesh
    names, and the faces marked as drawn from behind) count twice. Zero-area triangles the decimation leaves are
    removed. A collapse mixes the weights of the vertices it merges, so a decimated mesh's vertex can end up on more
    bones than the fighter's cap (``max_influences``, the cap weights.solve held them to; HSDRawViewer refuses a
    vertex on more than 6) or not summing to 1: the decimated meshes are held to the solve's rule again
    (weights.cap_groups), here, where their weights last change. -> record of the counts before and after, and of
    the weights held (influences)."""
    doubled = _as_layers(doubled)
    counts = {o.name: _triangles(o) for o in meshes}
    drawn = sum(_drawn(o, doubled) for o in meshes)
    record = {'budget': budget, 'before': drawn, 'after': drawn, 'decimated': False, 'counts_twice': sorted(doubled)}
    if drawn <= budget:
        return record
    share = budget / drawn
    reducible = [o for o in meshes if len(o.data.polygons) > 3]     # Blender's Decimate leaves 3 faces or fewer as is
    for o in reducible:
        _decimate(o, share)
    after = sum(_drawn(o, doubled) for o in meshes)
    passes = 1
    # The modifier lands within a few triangles of its share, over as often as under (Spider-Man on Zelda: 9,004 of
    # 9,000 after a second pass aimed at the budget), so while it is over, the next pass aims as far under the budget
    # as the last one landed over it, for as long as each pass still takes some.
    aim = budget
    while after > budget:
        for o in reducible:
            _decimate(o, max(aim, 1) / after)
        passes += 1
        now = sum(_drawn(o, doubled) for o in meshes)
        if now >= after:
            break
        aim, after = budget - (now - budget), now
    removed = sum(_remove_zero_area(o) for o in meshes)
    after = sum(_drawn(o, doubled) for o in meshes)
    record.update(after=after, decimated=True, share=round(share, 4), passes=passes, zero_area_removed=removed,
                  meshes={o.name: [counts[o.name], _triangles(o)] for o in meshes},
                  influences=cap_groups(reducible, max_influences))
    if after > budget:
        raise Failed(f'decimation left {after:,} triangles drawn; the budget is {budget:,} (from {drawn:,})')
    return record


def _decimate(o, ratio):
    import bpy
    if o.data.users > 1:
        o.data = o.data.copy()
    m = o.modifiers.new('prep: triangle budget', 'DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = ratio
    m.use_collapse_triangulate = True
    try:
        with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o]):
            bpy.ops.object.modifier_move_to_index(modifier=m.name, index=0)
            bpy.ops.object.modifier_apply(modifier=m.name)
    except RuntimeError as e:
        o.modifiers.remove(m)
        raise Failed(f'{o.name}: the triangle budget needs it decimated and Blender would not: {e}')


def _remove_zero_area(o):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(o.data)
    flat = [f for f in bm.faces if 2 * f.calc_area() <= DEGENERATE_AREA]
    if flat:
        bmesh.ops.delete(bm, geom=flat, context='FACES_ONLY')
        bm.to_mesh(o.data)
    bm.free()
    return len(flat)
