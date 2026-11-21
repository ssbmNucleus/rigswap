"""5. normals: collapsed faces and loose vertices removed; invalid normals repaired from faces, neighbours or the centre
(repair_normals)."""
from __future__ import annotations

import numpy as np

from ... import Failed
from .triangles import DEGENERATE_AREA


# A normal this short or shorter, or NaN, has no direction, and HSDRawViewer's model import hangs on it.
# (randall model_import/studio/geometry.py ZERO_NORMAL; shared/modellab/native_import.py DegenerateNormals)
ZERO_NORMAL = 1e-6
# A face no wider than this share of the model's size across its longest edge is a sliver: it draws nothing, and the
# .dae's float text rounds its short edge to nothing (two corners at one point: a zero normal the import rejects).
SLIVER = 1e-6
# HSDRawViewer writes each vertex coordinate in the smallest integer format that keeps it within this many units of
# its value (HSDRaw/Tools/GX_AttributeCompressor.cs, GX_VA_POS; export/checks.py holds the read-back to it).
POSITION_ERROR = 0.005


# ---- 5. normals ------------------------------------------------------------------------------------------------------

def corner_normals(me):
    values = np.empty(len(me.loops) * 3)
    me.corner_normals.foreach_get('vector', values)
    return values.reshape(-1, 3)


def repair_normals(meshes):
    """Keep usable corner normals; repair the others from incident faces, neighbours, then the mesh centre.

    A face with a collapsed edge has no corner-normal basis in Blender: even setting a custom normal can leave
    that corner zero. Remove those zero-area faces first (and slivers: SLIVER, and specks: speck), and vertices
    with no remaining faces. -> counts
    """
    import bmesh
    record = {'repaired': 0, 'collapsed_faces_removed': 0, 'loose_vertices_removed': 0, 'meshes': {}}
    points = [np.array([v.co[:] for v in o.data.vertices]) for o in meshes if len(o.data.vertices)]
    size = float(np.ptp(np.vstack(points), axis=0).max()) if points else 0.0

    def collapsed_face(f, turn):
        lengths = [(e.verts[0].co - e.verts[1].co).length for e in f.edges]
        area = f.calc_area()
        return ((2 * area <= DEGENERATE_AREA and min(lengths) == 0) or 2 * area <= SLIVER * size * max(lengths)
                or speck(np.array([v.co[:] for v in f.verts]) @ turn.T))
    for o in meshes:
        me = o.data
        bm = bmesh.new()
        bm.from_mesh(me)
        turn = np.array(o.matrix_world)[:3, :3]     # the written units are the scene's
        collapsed = [f for f in bm.faces if collapsed_face(f, turn)]
        counts = {'repaired': 0, 'faces': 0, 'neighbours': 0, 'outward': 0,
                  'collapsed_faces_removed': len(collapsed), 'loose_vertices_removed': 0}
        if collapsed:
            bmesh.ops.delete(bm, geom=collapsed, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        counts['loose_vertices_removed'] = len(loose)
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        if collapsed or loose:
            bm.to_mesh(me)
            me.update()
        bm.free()
        normals = corner_normals(me)
        bad = ~usable_normals(normals)
        if bad.any():
            fixes, sources = _normal_fallbacks(me, normals)
            for i in np.flatnonzero(bad):
                vertex = me.loops[i].vertex_index
                normals[i] = fixes[vertex]
                counts[sources[vertex]] += 1
            me.normals_split_custom_set(normals.tolist())
            me.update()
        flat = _shade_flat_where_no_fan(me)
        if flat:
            counts['shaded_flat'] = flat
        still = int((~usable_normals(corner_normals(me))).sum())
        if still:
            raise Failed(f'{o.name}: {still} normals still have no direction after the repair')
        counts['repaired'] = int(bad.sum())
        if any(counts.values()):
            record['meshes'][o.name] = counts
        for key in ('repaired', 'collapsed_faces_removed', 'loose_vertices_removed'):
            record[key] += counts[key]
    return record


def speck(corners):
    """Whether a face with these corners (in the units the costume is written in) is a speck the written file can't
    hold: one no wider on any axis than twice POSITION_ERROR. Each corner may be written that far off its place, so
    the whole face may be written at one point, or turned over. It draws nothing at any camera, and the game's own
    read-back of it is a face with two corners at one point, or none.

    Taz (Looney Tunes Racing OBJ, 2026-11-19) carries 51 such faces in his source, at most a thousandth of a unit
    across on Donkey Kong, and the refine cut each in 16: the read-back lost two of them (``missing``) and turned
    others over, wherever the answers happened to put them, and the export failed on one answer set and not the
    other."""
    corners = np.asarray(corners, float)
    return bool((np.ptp(corners, axis=0) <= 2 * POSITION_ERROR).all())


def _shade_flat_where_no_fan(me):
    """The faces with a corner that still has no direction, shaded flat, each such corner given its face's normal.

    Two faces folded back to back on the same vertices (a card with no thickness) make a smooth fan whose normals sum
    to nothing: Blender has no space to hold a custom normal there, so setting one leaves the corner zero. Shaded
    flat, a face is a fan of its own, and its normal is the corner's. -> how many faces were shaded flat"""
    normals = corner_normals(me)
    bad = ~usable_normals(normals)
    if not bad.any():
        return 0
    face_of = np.empty(len(me.loops), dtype=np.int64)
    for p in me.polygons:
        face_of[p.loop_start:p.loop_start + p.loop_total] = p.index
    faces = sorted({int(f) for f in face_of[bad] if usable_normals(np.array(me.polygons[int(f)].normal))})
    for f in faces:
        me.polygons[f].use_smooth = False
    for i in np.flatnonzero(bad):
        normals[i] = me.polygons[int(face_of[i])].normal[:]
    me.normals_split_custom_set(normals.tolist())
    me.update()
    return len(faces)


def usable_normals(normals):
    return np.isfinite(normals).all(axis=-1) & (np.linalg.norm(normals, axis=-1) > ZERO_NORMAL)


def _normal_fallbacks(me, normals):
    """One unit fallback per vertex, using only finite directions and nonzero-area faces."""
    points = np.array([v.co[:] for v in me.vertices])
    faces = [[] for _ in points]
    corners = [[] for _ in points]
    neighbours = [set() for _ in points]
    for p in me.polygons:
        if 2 * p.area > DEGENERATE_AREA and usable_normals(np.array(p.normal)):
            for v in p.vertices:
                faces[v].append(p.normal[:])
    for i, loop in enumerate(me.loops):
        if usable_normals(normals[i]):
            corners[loop.vertex_index].append(normals[i])
    for edge in me.edges:
        a, b = edge.vertices
        neighbours[a].add(b)
        neighbours[b].add(a)

    def average(values):
        direction = np.mean(values, axis=0) if len(values) else np.zeros(3)
        return direction / np.linalg.norm(direction) if usable_normals(direction) else np.zeros(3)

    face_normals = np.array([average(values) for values in faces])
    vertex_normals = np.array([average(values) for values in corners])
    missing = ~usable_normals(vertex_normals)
    vertex_normals[missing] = face_normals[missing]
    centre = points.mean(axis=0)
    # At the centre itself there is no radial direction: use the farthest vertex's ray, or +z for a point mesh.
    ray = points[np.argmax(np.linalg.norm(points - centre, axis=1))] - centre
    axis = average([ray]) if usable_normals(ray) else np.array([0., 0., 1.])
    fixes, sources = [], []
    for v, point in enumerate(points):
        candidates = [('faces', face_normals[v]),
                      ('neighbours', average([vertex_normals[n] for n in sorted(neighbours[v])
                                              if usable_normals(vertex_normals[n])])),
                      ('outward', average([point - centre]))]
        source, direction = next(((s, n) for s, n in candidates if usable_normals(n)), ('outward', axis))
        fixes.append(direction)
        sources.append(source)
    return fixes, sources
