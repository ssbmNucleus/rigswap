"""3. back faces: HSDRawViewer's import culls one side of every mesh it writes, so each face the game should see from
behind is given a back layer (two_sided_layers): the thin layers ``<build_dir>/layers.json`` names (a cape, a skirt),
each entry a mesh's name (the whole mesh), or {"object": a mesh's name, "piece": n} (only that mesh's faces of piece n,
by the face attribute weights.solve writes; build writes these from the description), and every face the model draws
from behind (mark_drawn_from_behind): a face of a material that does not cull its back (Blender's use_backface_culling
off: every material a .dae, .fbx or .obj brings in, and a glTF material marked doubleSided) whose back is seen from
outside, as on a hat brim, a skirt or a sheet of hair. Before anything is counted, the rendering layers, which the game
cannot draw, are dropped (drop_rendering_layers)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ... import Failed
from .weights import PIECE
from ....library import moves_as

# The face attribute marking the faces the model draws from behind (mark_drawn_from_behind), which get a back layer
# (two_sided_layers). An integer, like PIECE, so the decimation carries it over to the faces it keeps.
BACK = 'rigswap_back'
# The directions a face is looked at from: an icosahedron's vertices with each edge split once (42, about 30 degrees
# apart), so a face's back side is looked at from about 21 of them.
LOOK_SUBDIVISIONS = 1
# How far behind a face its ray starts, as a share of the model's size: off the face itself.
OFFSET = 1e-4


def drop_rendering_layers(meshes, pieces, keep=()):
    """Remove the faces of the rendering layers (library.moves_as: belongs layer_of) using the build's piece attribute;
    retain the thin layers' geometry (a sleeve that moves as a layer), and every face of the objects in ``keep``, which
    a detail draws (prep.detail_objects: the eyes). A piece a detail chose is drawn as that detail, whatever else the
    description says of it: pieces described as a layer over a mask
    that the eye round also chose as the eyes: dropping them would take both eye objects whole, so the export had no
    eyes to make blink. -> (dropped, kept): a record for each piece, its meshes, faces and
    triangles; kept, the rendering-layer faces left in the detail's objects."""
    import bpy
    import bmesh
    rendering = {p['id'] for p in pieces if moves_as.is_rendering_layer(p)}
    dropped, kept = {}, {}
    if not rendering:
        return [], []
    # Melee decision: the body underneath is complete, and the game has no way to draw these rendering passes as
    # intended. Remove them before counting triangles or DObj places; physical layers such as sleeves stay.
    for o in meshes:
        attr = o.data.attributes.get(PIECE)
        if attr is None or attr.domain != 'FACE':
            raise Failed(f'{o.name} has no {PIECE} face attribute to identify rendering layers')
        faces = {f.index: attr.data[f.index].value for f in o.data.polygons
                 if attr.data[f.index].value in rendering}
        if not faces:
            continue
        for index, piece in faces.items():
            entry = (kept if o.name in keep else dropped).setdefault(
                piece, {'piece': piece, 'meshes': [], 'faces': 0, 'triangles': 0})
            if o.name not in entry['meshes']:
                entry['meshes'].append(o.name)
            entry['faces'] += 1
            entry['triangles'] += len(o.data.polygons[index].vertices) - 2
        if o.name in keep:
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in faces], context='FACES')
        bm.to_mesh(o.data)
        bm.free()
        if not o.data.polygons:
            bpy.data.objects.remove(o, do_unlink=True)
    return [dropped[p] for p in sorted(dropped)], [kept[p] for p in sorted(kept)]


def read_layers(build_dir):
    """The thin layers ``<build_dir>/layers.json`` lists, as {mesh name: None (the whole mesh) or the set of piece
    numbers whose faces are the layer}; {} when the file is not there. An entry is a mesh's name, or {"object": a
    mesh's name, "piece": a piece number}."""
    path = Path(build_dir) / 'layers.json'
    if not path.is_file():
        return {}
    return layers_of(json.loads(path.read_text(encoding='utf-8')), path.name)


def layers_of(entries, where='layers.json'):
    """read_layers on a list already read: {mesh name: None or a set of piece numbers}. A whole mesh named anywhere
    wins over its pieces."""
    def piece_entry(e):
        return isinstance(e, dict) and isinstance(e.get('object'), str) and isinstance(e.get('piece'), int)
    if not isinstance(entries, list) or not all(isinstance(e, str) or piece_entry(e) for e in entries):
        raise Failed(f'{where} must be a list of mesh names or {{"object": name, "piece": number}}, not {entries!r}')
    out = {}
    for e in entries:
        if isinstance(e, str):
            out[e] = None
        elif e['object'] not in out or out[e['object']] is not None:
            out[e['object']] = (out.get(e['object']) or set()) | {e['piece']}
    return out


def directions(subdivisions=LOOK_SUBDIVISIONS):
    """Unit vectors spread evenly over the sphere: an icosahedron's vertices, each edge split ``subdivisions`` times."""
    t = (1 + 5 ** .5) / 2
    verts = [np.array(v, float) / np.linalg.norm(v) for v in
             [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
              (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        middles, split = {}, []

        def middle(a, b):
            if (min(a, b), max(a, b)) not in middles:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                middles[(min(a, b), max(a, b))] = len(verts) - 1
            return middles[(min(a, b), max(a, b))]
        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = split
    return np.array(verts)


def seen_from_behind(normals, centres, hits, size, looks=None):
    """(n,) True for each face (its unit ``normals`` and its ``centres``) whose back is seen from outside: from some
    direction of ``looks`` (directions()) on its back side, the ray from just behind its centre out along that
    direction hits nothing. ``hits(origin, direction)``: whether a ray hits the model; ``size``: the model's size. A
    closed surface never shows its back from outside; an open sheet shows it wherever nothing covers it."""
    looks = directions() if looks is None else np.asarray(looks, float)
    out = np.zeros(len(normals), bool)
    for i, (n, c) in enumerate(zip(np.asarray(normals, float), np.asarray(centres, float))):
        start = c - n * size * OFFSET
        out[i] = any(not hits(start, d) for d in looks[looks @ n < 0])
    return out


def mark_drawn_from_behind(meshes, keep=()):
    """Mark (the face attribute BACK) the faces of ``meshes`` the model draws from behind: a face of a material that
    draws both sides (use_backface_culling off) whose back is seen from outside (seen_from_behind), every face of every
    mesh in the way, in the shape the meshes have now (the fighter's rest). The objects in ``keep`` are drawn as a
    detail made them (prep.detail_objects: the eyes) and are not marked. -> {mesh name: faces marked}"""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    verts, polys = [], []
    for o in meshes:
        base = len(verts)
        verts += [o.matrix_world @ v.co for v in o.data.vertices]
        polys += [[base + i for i in p.vertices] for p in o.data.polygons]
    if not polys:
        return {}
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=False)
    corners = np.array([tuple(v) for v in verts])
    size = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))

    def hits(origin, direction):
        return tree.ray_cast(Vector(origin), Vector(direction))[0] is not None
    marked = {}
    for o in meshes:
        me = o.data
        both = np.array([m is not None and not m.use_backface_culling for m in me.materials] or [False])
        index = np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get('material_index', index)
        mask = np.zeros(len(me.polygons), bool)
        if o.name not in keep:
            mask = both[np.clip(index, 0, len(both) - 1)]
        if mask.any():
            turn = o.matrix_world.to_3x3()
            faces = [me.polygons[int(i)] for i in np.flatnonzero(mask)]
            normals = np.array([tuple((turn @ p.normal).normalized()) for p in faces])
            centres = np.array([tuple(o.matrix_world @ p.center) for p in faces])
            mask[np.flatnonzero(mask)] = seen_from_behind(normals, centres, hits, size)
        attr = me.attributes.get(BACK) or me.attributes.new(BACK, 'INT', 'FACE')
        attr.data.foreach_set('value', mask.astype(np.int32))
        if mask.any():
            marked[o.name] = int(mask.sum())
    return marked


def back_faces(o, layers=()):
    """(n faces,) True for the faces of mesh ``o`` that get a back layer: its thin layer's faces (``layers``, as
    read_layers gives them, or a list of mesh names) and the faces marked as drawn from behind (BACK)."""
    layers = _as_layers(layers)
    mask = layer_faces(o, layers[o.name]) if o.name in layers else np.zeros(len(o.data.polygons), bool)
    attr = o.data.attributes.get(BACK)
    if attr is not None and attr.domain == 'FACE' and len(attr.data) == len(mask):
        marked = np.empty(len(mask), np.int32)
        attr.data.foreach_get('value', marked)
        mask = mask | (marked != 0)
    return mask


def layer_faces(o, pieces):
    """(n faces,) True for the faces of mesh ``o`` that are its layer: every face when ``pieces`` is None, else the
    faces whose piece (the face attribute weights.solve writes) is in ``pieces``. Failed when the mesh has no such
    attribute to tell its pieces apart."""
    nf = len(o.data.polygons)
    if pieces is None:
        return np.ones(nf, bool)
    attr = o.data.attributes.get(PIECE)
    if attr is None or attr.domain != 'FACE' or len(attr.data) != nf:
        raise Failed(f'layers.json names pieces {sorted(pieces)} of {o.name}, which has no {PIECE} face attribute')
    fp = np.empty(nf, np.int64)
    attr.data.foreach_get('value', fp)
    return np.isin(fp, sorted(pieces))


def _as_layers(layers):
    """{mesh name: None or piece numbers} from read_layers' dict or a plain list of mesh names."""
    return dict(layers) if isinstance(layers, dict) else {n: None for n in layers}


# ---- 3. back faces ---------------------------------------------------------------------------------------------------

def two_sided_layers(meshes, layers):
    """Give the faces that need one (back_faces: each layer in ``layers``, as read_layers gives them or a list of mesh
    names, and the faces marked as drawn from behind) a back layer: a copy of them with the winding reversed and the
    normals turned round, the same UVs, materials, weights and piece. HSDRawViewer's import culls one side of every
    mesh it writes (HSDRaw/Tools/POBJ_Generator.cs CullMode), and where vanilla turns culling off GX lights the back
    with the front's normals, so a face seen from behind needs the geometry to be seen and lit from behind. (randall
    model_import/studio/sheets.py) The mark is taken off once used. -> record"""
    layers = _as_layers(layers)
    record = {'layers': {n: (sorted(p) if p is not None else 'the whole mesh') for n, p in sorted(layers.items())},
              'triangles_added': 0, 'meshes': {}}
    for o in meshes:
        faces = back_faces(o, layers)
        if BACK in o.data.attributes:
            o.data.attributes.remove(o.data.attributes[BACK])
        if o.name in layers or faces.any():
            added = _add_back_layer(o, faces)
            record['meshes'][o.name] = added
            record['triangles_added'] += added
    return record


def _add_back_layer(o, faces_of_layer=None):
    """A back layer for the faces of ``o`` that ``faces_of_layer`` (a mask over its faces; None: all) marks."""
    import bmesh
    from .normals import corner_normals
    from .triangles import _triangles
    me = o.data
    before = _triangles(o)
    normals = corner_normals(me)
    corner_vert = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get('vertex_index', corner_vert)
    faces = len(me.polygons)
    if faces_of_layer is None:
        faces_of_layer = np.ones(faces, bool)
    if not faces_of_layer.any():
        return 0
    # the normal at each corner, by its face and vertex, for the copy to turn round
    by_corner = {(p.index, int(corner_vert[i])): normals[i] for p in me.polygons for i in p.loop_indices}

    bm = bmesh.new()
    bm.from_mesh(me)
    face_of = bm.faces.layers.int.new('prep_back_face')
    vert_of = bm.verts.layers.int.new('prep_back_vert')
    for f in bm.faces:
        f[face_of] = f.index
    for v in bm.verts:
        v[vert_of] = v.index
    bm.faces.ensure_lookup_table()
    originals = [f for f in bm.faces if faces_of_layer[f.index]]
    used = {v for f in originals for v in f.verts} | {e for f in originals for e in f.edges}
    copy = bmesh.ops.duplicate(bm, geom=list(used) + originals)
    backs = [g for g in copy['geom'] if isinstance(g, bmesh.types.BMFace)]
    bmesh.ops.reverse_faces(bm, faces=backs)
    bm.to_mesh(me)
    bm.free()

    face_source = np.empty(len(me.polygons), np.int64)
    me.attributes['prep_back_face'].data.foreach_get('value', face_source)
    vert_source = np.empty(len(me.vertices), np.int64)
    me.attributes['prep_back_vert'].data.foreach_get('value', vert_source)
    corner_vert = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get('vertex_index', corner_vert)
    out = np.empty((len(me.loops), 3))
    for p in me.polygons:
        turn = -1. if p.index >= faces else 1.
        for i in p.loop_indices:
            out[i] = turn * by_corner[(int(face_source[p.index]), int(vert_source[corner_vert[i]]))]
    me.attributes.remove(me.attributes['prep_back_face'])
    me.attributes.remove(me.attributes['prep_back_vert'])
    me.normals_split_custom_set(out.tolist())
    me.update()
    return _triangles(o) - before
