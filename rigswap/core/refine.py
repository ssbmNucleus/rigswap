"""Give a sparse surface the fighter's resolution while it is still in its own pose, and shade the surface that results.

Four-way triangle cuts interpolate the author's skin and corner data, and say where they crossed a hard seam (``_across``).
PN edge midpoints use the author's normals; an edge whose two sides have different normals stays straight. Copies of a place share everything done there, across mesh
and UV boundaries. A crease is what the file kept apart, not a chosen angle.

A cut alone adds no shading: a new corner's interpolated normal is the old triangle's own, so a game that lights its
corners draws the old triangles again. New corners are only worth having once shading reads the surface (``shade``).
"""
import numpy as np

from . import segments

SEAM = 'rigswap_seam'       # INT point attribute: 1 where a cut crossed a hard seam


def cuts(source, fighter):
    """The whole number of four-way cuts whose triangle count is nearest the fighter's, ties taking fewer."""
    if source <= 0 or fighter <= source:
        return 0
    below = int(np.floor(np.log(fighter / source) / np.log(4)))
    return min((below, below + 1), key=lambda n: (abs(source * 4 ** n - fighter), n))


def count(meshes):
    """The drawn surface's triangle count, including polygons triangulated as Blender reads them."""
    total = 0
    for ob in meshes:
        ob.data.calc_loop_triangles()
        total += len(ob.data.loop_triangles)
    return total


def stored(ob):
    """(the corner normals the file stores, in world space; how far apart two readings of one normal can lie).

    Blender stores custom normals as two signed-short angles, each at most 2*pi; rounding costs half a step in each. Its
    encoder also snaps a normal within 1e-4 of the fan to that fan. Two corners further apart than that are normals
    the file kept apart. This is an encoding-error bound, not a crease angle. See mesh_normals.cc,
    corner_space_custom_normal_to_data.
    """
    me = ob.data
    world = np.array(ob.matrix_world)[:3, :3]
    normals = np.empty(len(me.loops) * 3)
    me.corner_normals.foreach_get('vector', normals)
    custom = me.attributes.get('custom_normal')
    noise = 2 * np.pi / np.iinfo(np.int16).max + 1e-4 if custom and custom.data_type == 'INT16_2D' else 1e-6
    return _unit(normals.reshape(-1, 3) @ np.linalg.inv(world)), noise * np.linalg.cond(world)


def places(points):
    """The place each point is at, numbered: the split copies a file makes along seams are one (segments.WELD)."""
    cell = segments.WELD * np.ptp(points, axis=0).max()
    return np.unique(np.round(points / cell).astype(np.int64), axis=0, return_inverse=True)[1].ravel()


def resolution(meshes, fighter_meshes):
    """Refine only when the fighter's triangle count calls for it; dense files keep their original data.

    BMesh's topological edge split identifies each midpoint directly and interpolates UVs, deform weights, shape
    keys and other attributes. No coordinate search or source-weight truncation is needed: the game skin is capped
    after correspondence, by repose. Original vertices stay in place; only new points follow the PN edge curve.
    """
    import bmesh
    source, target = count(meshes), count(fighter_meshes)
    rounds = cuts(source, target)
    report = dict(source_triangles=source, fighter_triangles=target, cuts=rounds, triangles=source)
    if not rounds:
        return report
    opened = []
    try:
        for ob in meshes:
            if ob.data.users > 1:
                ob.data = ob.data.copy()
            me = ob.data
            normals, noise = stored(ob)
            custom = me.attributes.get('custom_normal')
            if custom:
                me.attributes.remove(custom)
            bm = bmesh.new()
            bm.from_mesh(me)
            tag = bm.loops.layers.float_vector.new('rigswap_source_normal')
            for face, polygon in zip(bm.faces, me.polygons):
                for loop, index in zip(face.loops, polygon.loop_indices):
                    loop[tag] = normals[index]
            bm.verts.layers.int.new(SEAM)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            opened.append((ob, bm, tag, noise))
        for _ in range(rounds):
            points = _midpoints(opened)
            for ob, bm, _, _ in opened:
                _cut(bm, points[ob.name])
        for ob, bm, tag, _ in opened:
            # The author's normal field, interpolated on new corners, says what the file kept apart (``shade``).
            normals = np.array([loop[tag] for f in bm.faces for loop in f.loops]) @ np.array(ob.matrix_world)[:3, :3]
            bm.loops.layers.float_vector.remove(tag)
            bm.to_mesh(ob.data)
            ob.data.update()
            ob.data.polygons.foreach_set('use_smooth', [True] * len(ob.data.polygons))
            ob.data.normals_split_custom_set(_unit(normals).tolist())
            ob.data.update()
        report['triangles'] = count(meshes)
        return report
    finally:
        for _, bm, _, _ in opened:
            bm.free()


def shade(meshes, files):
    """Shading is what the surface does at each place: the faces meeting there, each by its angle at that place, so
    the reading doesn't turn on how the faces were cut. Every corner at a place takes it, across meshes, so no join
    between pieces shows. Only corners the file kept apart (``files``: ``stored``, per mesh) stay apart.

    The file's normals are true of the file's surface in the file's pose. A rigid piece could carry them, but a
    blended skin's surface also turns with its weights across a joint, and new corners have none of their own.
    """
    corners = [_corners(ob) for ob in meshes]
    points, faces, angles = [np.concatenate(values) for values in zip(*corners)]
    tags = np.concatenate([normals for normals, _ in files])
    noise = np.repeat([e for _, e in files], [len(normals) for normals, _ in files])
    at = places(points)
    order = np.argsort(at, kind='stable')
    normals = tags.copy()
    for group in np.split(order, np.flatnonzero(np.diff(at[order])) + 1):
        together = np.linalg.norm(tags[group, None] - tags[None, group], axis=2) <= noise[group, None] + noise[None, group]
        total = together @ (faces[group] * angles[group, None])
        length = np.linalg.norm(total, axis=1)
        met = length > 1e-9                     # faces that cancel exactly say nothing: the file's stands
        normals[group[met]] = total[met] / length[met, None]
    for ob, part in zip(meshes, np.split(normals, np.cumsum([len(c[0]) for c in corners])[:-1])):
        ob.data.normals_split_custom_set(part.tolist())
        ob.data.update()


def _corners(ob):
    """Per corner of a mesh: its place in world space, its face's normal there, and the angle its face has there."""
    me = ob.data
    world = np.array(ob.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    index = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get('vertex_index', index)
    points = co.reshape(-1, 3)[index] @ world[:3, :3].T + world[:3, 3]
    starts, totals = np.empty(len(me.polygons), np.int64), np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get('loop_start', starts)
    me.polygons.foreach_get('loop_total', totals)
    after, before = np.arange(1, len(index) + 1), np.arange(-1, len(index) - 1)
    after[starts + totals - 1] = starts
    before[starts] = starts + totals - 1
    a, b = points[after] - points, points[before] - points
    lengths = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    angles = np.arccos(np.clip((a * b).sum(1) / np.maximum(lengths, 1e-12), -1, 1))
    # A polygon's normal is the sum of its corners' crossings (Newell), whatever its side count.
    crossing = np.cross(a, b)
    faces = np.add.reduceat(crossing, starts)
    return points, _unit(faces[np.repeat(np.arange(len(starts)), totals)]), angles


def _unit(values):
    """Normalize normals, keeping an absent normal at zero."""
    values = np.asarray(values, float)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, length, out=np.zeros_like(values), where=length > 0)


def _midpoints(opened):
    """Each mesh's new points {edge: place in its own coordinates}, one PN midpoint shared by every copy of an edge,
    with straight creases from split corner normals.

    For a cubic PN edge with unit endpoint normals na, nb, its midpoint leaves the straight midpoint by
    -(dot(edge, na)*na - dot(edge, nb)*nb)/8. This polynomial needs no bulge cap. Every side of an edge must
    agree within its normal encoding's precision for that edge to curve; differing sides keep one straight edge.
    Every copy takes one worked-out point, so a seam that was one surface stays exactly one (repose.Surface).
    """
    worlds = {ob.name: np.array(ob.matrix_world) for ob, _, _, _ in opened}
    points = {ob.name: np.array([v.co[:] for v in bm.verts]) @ worlds[ob.name][:3, :3].T + worlds[ob.name][:3, 3]
              for ob, bm, _, _ in opened}
    at = places(np.vstack(list(points.values())))
    edges, output, start = {}, {}, 0
    for ob, bm, tag, noise in opened:
        output[ob.name] = {}
        bm.verts.index_update()
        for edge in bm.edges:
            a, b = sorted(edge.verts, key=lambda v: at[start + v.index])
            sides = []
            for loop in edge.link_loops:
                other = loop.link_loop_next
                left, right = (loop, other) if loop.vert is a else (other, loop)
                sides.append(([np.asarray(l[tag]) for l in (left, right)], noise))
            ends = points[ob.name][a.index], points[ob.name][b.index]
            edges.setdefault((at[start + a.index], at[start + b.index]), []).append((ob.name, edge, sides, ends))
        start += len(bm.verts)
    for copies in edges.values():
        sides = [side for _, _, each, _ in copies for side in each]
        a, b = copies[0][3]
        point = (a + b) / 2
        if sides and all(np.max(np.linalg.norm(np.subtract(n, m), axis=1)) <= e + f
                         for i, (n, e) in enumerate(sides) for m, f in sides[i + 1:]):
            na, nb = _unit(np.mean([n for n, _ in sides], axis=0))
            edge = b - a
            point = point - ((edge @ na) * na - (edge @ nb) * nb) / 8
        for name, edge, _, _ in copies:
            world = worlds[name]
            output[name][edge] = np.linalg.solve(world[:3, :3], point - world[:3, 3])
    return output


def seam(meshes):
    """The stacked points (rig.Rig.points) where a cut crossed a hard seam: repose.Surface.blend's seams on a cut surface."""
    rows, start = [], 0
    for ob in meshes:
        marks = ob.data.attributes.get(SEAM)
        if marks:
            values = np.empty(len(marks.data), np.int64)
            marks.data.foreach_get('value', values)
            rows.append(start + np.flatnonzero(values))
        start += len(ob.data.vertices)
    return np.concatenate(rows) if rows else np.empty(0, np.int64)


def _across(a, b, deform, marks):
    """(the end whose skin a new point on edge ``a``-``b`` takes, None to interpolate; whether the point is on a seam).

    Ends that share no joint are a hard seam: the file states no blend between them. The new point still interpolates,
    which keeps it on the surface the edge drew however far the fighter's joints part the two ends. But one level of
    skin would read as an authored blend, which repose.Surface.blend leaves alone, so the point is marked and blend
    starts its band there. Further cuts keep that line: along it skin interpolates, and off it a new point is its own
    end's, for the band to reach.
    """
    if a[marks] and b[marks]:
        return None, True
    if a[marks] or b[marks]:
        return b if a[marks] else a, False
    ends = [{group for group, weight in v[deform].items() if weight > 0} if deform else set() for v in (a, b)]
    return None, bool(ends[0] and ends[1] and not ends[0] & ends[1])


def _cut(bm, points):
    """Split each original edge once, at its new point, then join its face's three midpoints into four triangles.

    Topological references distinguish unrelated edges crossing at a midpoint. Shape-key offsets follow the same
    new-point displacement, so interpolating a key does not undo the refined basis surface when the rig is read.
    """
    import bmesh
    from mathutils import Vector
    faces = [list(f.edges) for f in bm.faces]
    deform = bm.verts.layers.deform.active
    marks = bm.verts.layers.int[SEAM]
    midpoints = {}
    for edge in list(bm.edges):
        point = Vector(points[edge])
        side, crossed = _across(*edge.verts, deform, marks)
        _, vertex = bmesh.utils.edge_split(edge, edge.verts[0], .5)
        move = point - vertex.co
        vertex.co = point
        for layer in bm.verts.layers.shape.values():
            vertex[layer] += move
        vertex[marks] = crossed
        if side:
            vertex[deform].clear()
            for group, weight in side[deform].items():
                vertex[deform][group] = weight
        midpoints[edge] = vertex
    for edges in faces:
        bmesh.ops.connect_verts(bm, verts=[midpoints[e] for e in edges], check_degenerate=True)
    bm.normal_update()
