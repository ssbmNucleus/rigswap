"""Open pieces draw both sides; closed pieces cull. Source material flags do not decide this. A face with its own
reversed twin just behind it (a gown's lining, a cape's inside: the surface draws its own back) culls even in an open
piece: drawn both sides, the two would lie at one depth and the game would show whichever wins, face by face. So does
a face whose back another piece hides (covered: a shield's dome and its flat back plate, two open meshes a hand's
breadth apart): drawn both sides, the plate's back showed round the dome's rim, a second shield behind the first.

Decide before reduction or the accessory partition cuts the surface. Material variants carry the result through
Blender's joins, GLB and the accessory SMD; export writes it into each draw unit's PObj flags. No back geometry.
An object a detail draws as one DObj (the eyes) takes one mode: both sides when any of its pieces is open.
"""
from collections import Counter
from itertools import product

import numpy as np

from .weights import PIECE

TWIN = 0.01         # a reversed face whose centre lies within this share of the model's size backs a face


def boundary_counts(points, faces, pieces):
    """Boundary edges per piece, across meshes/material seams. Coincident vertices share an exact position label.

    Only edges incident to exactly one face are boundaries. Collapsed edges do not count. This is a measurement;
    vertices, UVs, normals and skin weights are never welded or changed.
    """
    _, labels = np.unique(np.asarray(points), axis=0, return_inverse=True)
    edges = Counter()
    counts = dict.fromkeys(pieces, 0)
    for face, piece in zip(faces, pieces):
        ids = labels[face]
        for a, b in zip(ids, np.roll(ids, -1)):
            if a != b:
                edges[piece, min(a, b), max(a, b)] += 1
    for (piece, _, _), uses in edges.items():
        counts[piece] += uses == 1
    return counts


def backed(points, faces):
    """Per face: whether a reversed copy of it lies within TWIN of the model's size (its centre that near, facing the
    other way)."""
    pts = np.asarray(points, float)
    if not len(faces):
        return np.zeros(0, bool)
    tol = TWIN * (np.ptp(pts, axis=0).max() or 1.0)
    centres = np.array([pts[f].mean(axis=0) for f in faces])
    normals = np.zeros((len(faces), 3))
    for i, f in enumerate(faces):
        q = pts[f]
        n = np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)      # Newell: any polygon, any winding
        normals[i] = n / (np.linalg.norm(n) or 1.0)
    keys = [tuple(k) for k in np.floor(centres / tol).astype(np.int64)]
    cells = {}
    for i, k in enumerate(keys):
        cells.setdefault(k, []).append(i)
    out = np.zeros(len(faces), bool)
    for i, k in enumerate(keys):
        near = [j for d in product((-1, 0, 1), repeat=3) for j in cells.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]), ())]
        out[i] = any(j != i and normals[i] @ normals[j] < -0.5 and np.linalg.norm(centres[i] - centres[j]) < tol
                     for j in near)
    return out


def covered(points, faces, pieces):
    """Per face: whether the first surface straight behind it (from its centre, along its back) is another piece's,
    facing the other way. Seen from behind, that piece's front is what shows; the face's back never does (tlink's
    shield: its dome and its plate, each the other's). A piece's own far side (a sleeve, a skirt seen into) and a
    surface facing the face's way (the body under a shirt or a cape) leave it as it is."""
    import bpy  # noqa: F401  (mathutils comes with it)
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    pts = np.asarray(points, float)
    out = np.zeros(len(faces), bool)
    if not len(faces):
        return out
    centres = np.array([pts[f].mean(axis=0) for f in faces])
    normals = np.zeros((len(faces), 3))
    for i, f in enumerate(faces):
        q = pts[f]
        n = np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)
        normals[i] = n / (np.linalg.norm(n) or 1.0)
    tree = BVHTree.FromPolygons([tuple(p) for p in pts], [[int(v) for v in f] for f in faces])
    step = 1e-6 * (np.ptp(pts, axis=0).max() or 1.0)
    for i in range(len(faces)):
        if not normals[i].any():
            continue
        _, _, j, _ = tree.ray_cast(Vector(centres[i] - step * normals[i]), Vector(-normals[i]))
        out[i] = j is not None and j != i and pieces[j] != pieces[i] and normals[i] @ normals[j] < -0.5
    return out


def _surfaces(meshes):
    """Every face of ``meshes`` in world space, with its piece: (points, faces, pieces, [(object, face)])."""
    points, faces, pieces, owners = [], [], [], []
    for ob in meshes:
        base = len(points)
        points.extend(tuple(ob.matrix_world @ v.co) for v in ob.data.vertices)
        attr = ob.data.attributes.get(PIECE)
        for face in ob.data.polygons:
            faces.append(np.array(face.vertices) + base)
            pieces.append(('piece', attr.data[face.index].value) if attr is not None else ('object', ob.name))
            owners.append((ob, face))
    return points, faces, pieces, owners


def apply(meshes, whole=()):
    """Set material culling per piece, copying a shared material only when its faces need both modes. The objects
    named in ``whole`` draw as one group: every face of one takes its mode, open when any of its pieces is."""
    for ob in meshes:
        if ob.data.users > 1:
            ob.data = ob.data.copy()
    points, faces, pieces, owners = _surfaces(meshes)
    owners = [(ob, face.index, ob.material_slots[face.material_index].material if ob.material_slots else None)
              for ob, face in owners]
    counts = boundary_counts(points, faces, pieces)
    twin, hidden = backed(points, faces), covered(points, faces, pieces)
    groups = {ob.name: False for ob, _, _ in owners if ob.name in set(whole)}
    for (ob, _, _), piece in zip(owners, pieces):
        if ob.name in groups:
            groups[ob.name] |= counts[piece] > 0
    variants, first, splits = {}, {}, []
    for (ob, index, material), piece, own_back in zip(owners, pieces, twin | hidden):
        both = groups[ob.name] if ob.name in groups else counts[piece] > 0 and not own_back
        key = (material, both)
        if key not in variants:
            if material is None:
                import bpy
                variant = bpy.data.materials.new('surface')
            elif material not in first:
                variant = material
            else:
                variant = material.copy()
                splits.append({'source': material.name, 'variant': variant.name})
            first[material] = both
            variant.use_backface_culling = not both
            variants[key] = variant
        variant = variants[key]
        slots = list(ob.data.materials)
        if variant not in slots:
            ob.data.materials.append(variant)
            slots.append(variant)
        ob.data.polygons[index].material_index = slots.index(variant)
    return {'pieces': [{'kind': kind, 'id': pid, 'boundary_edges': n, 'two_sided': n > 0}
                       for (kind, pid), n in counts.items()], 'material_splits': splits,
            'faces_with_their_own_back': int(twin.sum()), 'faces_hidden_behind': int((hidden & ~twin).sum()),
            **({'whole_objects': groups} if groups else {})}
