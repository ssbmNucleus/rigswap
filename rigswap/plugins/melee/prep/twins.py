"""One surface drawn twice: a mesh lying on another face for face, where the game can draw only one of them.

A rip often splits a material the source game drew in passes into copies of the surface, one per pass, each with
one image: Mario + Rabbids Sparks of Hope's Mario has each of his meshes again as ``<mesh>_UV1``, the same vertices
drawing a grey reflection image (gen_reflection) through the second UV set. The source game blended the passes; the
costume's DObjs are each drawn whole, with one image, and of two at one depth the game shows the one drawn last, face
by face: Mario came out in his reflection's black and white stripes, his own colours under them. A pass that leaves
texels out (a cutout or a blended decal: Brawl's iris, highlight and shadow over the eye white, an eye shadow over
the eye) is drawn over the copy under it in the game too, and stays. A surface also has its own back as a second
mesh just behind it, facing the other way: a gown's lining, the inside of a cape, the inner wall of Olimar's glass
helmet (Pikmin 2: ``naka`` inside ``soto``).

``drop_twins`` reads the pairs (``pairs``: the same vertices, every vertex of each within sidedness.TWIN of the model's
size of one of the other's, both ways, and every face on the other's surface) and decides by what each copy draws
(``Drawn``):

    twins     the two face the same way: the same side drawn twice. Of the copies that draw all of it (opaque
              wherever they draw: no texel the game leaves out, not blended), the game would show one and lose the
              others whole. The one that draws the surface (``rank``) is in colour (a light pass, a reflection or a
              light or shadow map, is grey: it scales the light under it and has no colour of its own), else the copy
              the others are named after (``msh_body``, not ``msh_body_UV1``, ``Body``, not ``Body.UV1``). An opaque
              copy that is its pass by either (grey where it is in colour, or named after it: ``passes``) is
              dropped; copies nothing tells apart (the same image again, a form of the surface the variants left)
              are left as they are, and so are the decals.
    lining    the two face away from each other: the surface's own back. Both are kept (sidedness.py culls each, so
              each side is seen from its own side), unless one is translucent (facts/alpha.py: the mode its image's
              alpha gives it is Blend) and the other is not: an opaque lining is then seen through the glass as a
              solid wall, and it is dropped; the translucent surface, open with no back left, draws both its sides.

The objects a detail draws (the eyes) are never read or dropped. Run before sidedness and the triangle budget, so
neither counts a copy the game will not draw."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from ..facts.alpha import choose as alpha_mode
from .sidedness import TWIN
from .textures import _base_color_image, read_pixels

# A pair faces one way when its faces' normals agree (dot above this) on average, away from each other below its
# negative; between, it is not one surface.
FACING = 0.5
# A copy is grey when its drawn texels' channels differ by less than this on average, out of 255 (a light pass written
# as RGB: gen_reflection_02 is exactly grey; a JPEG of one is within a few levels).
GREY = 4.0

_TAKEN = re.compile(r'\.\d{3}$')       # Blender's suffix for a name already taken


@dataclass(eq=False)
class Copy:
    """One surface of the build: an object's faces of one material (a key by identity)."""
    obj: object
    slot: int
    faces: np.ndarray                   # face indices in the object
    centres: np.ndarray                 # (n, 3) world face centres
    normals: np.ndarray                 # (n, 3) world face normals
    verts: list                         # world vertices of the object, for its BVH
    polys: list                         # its faces as vertex index lists
    lo: np.ndarray = field(default=None)
    hi: np.ndarray = field(default=None)
    points: np.ndarray = field(default=None)       # (k, 3) the world vertices its faces use
    _tree: object = None
    _kd: object = None

    @property
    def material(self):
        slots = self.obj.material_slots
        return slots[self.slot].material if self.slot < len(slots) else None

    @property
    def name(self):
        m = self.material
        return f'{self.obj.name}/{m.name if m is not None else "-"}'

    def kd(self):
        if self._kd is None:
            from mathutils.kdtree import KDTree
            self._kd = KDTree(len(self.points))
            for i, p in enumerate(self.points):
                self._kd.insert(tuple(p), i)
            self._kd.balance()
        return self._kd

    def tree(self):
        if self._tree is None:
            from mathutils.bvhtree import BVHTree
            self._tree = BVHTree.FromPolygons(self.verts, self.polys, all_triangles=False)
        return self._tree


def copies(meshes, keep=()):
    """Every surface of ``meshes`` (an object's faces of one material slot) but those of the objects in ``keep``."""
    out = []
    for o in meshes:
        if o.name in keep:
            continue
        me = o.data
        world = np.array(o.matrix_world)
        co = np.empty(3 * len(me.vertices))
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
        turn = np.linalg.inv(world[:3, :3]).T
        index = np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get('material_index', index)
        centre = np.empty(3 * len(me.polygons))
        me.polygons.foreach_get('center', centre)
        normal = np.empty(3 * len(me.polygons))
        me.polygons.foreach_get('normal', normal)
        centre = centre.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
        normal = normal.reshape(-1, 3) @ turn.T
        normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(1e-12)
        verts = [tuple(v) for v in co]          # one list for the object's copies, read only
        for slot in np.unique(index):
            faces = np.flatnonzero(index == slot)
            polys = [list(me.polygons[int(f)].vertices) for f in faces]
            used = co[sorted({v for p in polys for v in p})]
            out.append(Copy(o, int(slot), faces, centre[faces], normal[faces], verts, polys,
                            used.min(axis=0), used.max(axis=0), used))
    return out


def facing(a, b, tol):
    """How ``a``'s faces face ``b``'s where they lie on it: the mean dot of each face's normal with the normal of the
    face of ``b`` nearest its centre; None when some face of ``a`` lies farther than ``tol`` from ``b``."""
    tree = b.tree()
    dots = []
    for c, n in zip(a.centres, a.normals):
        hit, hit_normal, _, distance = tree.find_nearest(tuple(c), tol)
        if hit is None:
            return None
        dots.append(float(np.dot(n, tuple(hit_normal))))
    return float(np.mean(dots)) if dots else None


def on_vertices(a, b, tol):
    """Whether every vertex of ``a`` lies within ``tol`` of a vertex of ``b``."""
    kd = b.kd()
    return all(kd.find(tuple(p))[2] <= tol for p in a.points)


def pairs(found, size):
    """[(a, b, 'twins' or 'lining')] for each two surfaces of ``found`` (copies) that are one surface drawn twice:
    the same vertices (every vertex of each within TWIN of ``size``, the model's, of one of the other's, both ways: a
    copy of the mesh, not another mesh that happens to lie there), every face on the other's surface, facing the same
    way on average (twins) or away from each other (lining)."""
    tol = TWIN * size
    out = []
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            if np.any(np.abs(a.lo - b.lo) > tol) or np.any(np.abs(a.hi - b.hi) > tol):
                continue
            if not (on_vertices(a, b, tol) and on_vertices(b, a, tol)):
                continue
            ab = facing(a, b, tol)
            ba = facing(b, a, tol) if ab is not None else None
            if ab is None or ba is None:
                continue
            if ab > FACING and ba > FACING:
                out.append((a, b, 'twins'))
            elif ab < -FACING and ba < -FACING:
                out.append((a, b, 'lining'))
    return out


@dataclass
class Drawn:
    """What a copy draws, at the texels its faces sample (``drawn``)."""
    image: str | None
    covers: float               # share of the drawn texels shown (alpha 128 or more)
    grey: bool                  # its drawn texels have no colour of their own (a light pass)
    mode: str                   # what its alpha makes it in the game (facts/alpha.py: choose): Blend, Test

    @property
    def opaque(self):
        """It draws every texel where it draws, unblended: drawn over another copy, it hides it whole."""
        return self.covers >= 1. and self.mode != 'Blend'

    def record(self):
        return {'image': self.image, 'covers': round(self.covers, 3), 'grey': self.grey, 'mode': self.mode,
                'opaque': self.opaque}


def drawn(copy, images=None):
    """What ``copy`` draws (Drawn): its material's base colour image (textures._base_color_image) sampled at each
    face's centre and its corners pulled a third of the way in, through the UV layer the image node reads (else the
    one the mesh renders with); an untextured material, its Base Color and Alpha. ``images``: a cache of read pixels by
    image name."""
    images = {} if images is None else images
    m = copy.material
    shader = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if m and m.node_tree else None
    meaning = {'alpha_mode': _alpha_mode(shader), 'kind': _kind(shader)}
    image = _base_color_image(m.node_tree) if m is not None and m.node_tree is not None else None
    if image is None or not (image.size[0] and image.size[1]):
        colour = np.asarray(shader.inputs['Base Color'].default_value[:3] if shader else (1., 1., 1.)) * 255
        alpha = shader.inputs['Alpha'].default_value if shader else 1.
        rgba = np.array([[*colour, alpha * 255]])
    else:
        if image.name not in images:
            images[image.name] = read_pixels(image)
        rgba = _sample(images[image.name], _uvs(copy, _image_node(m.node_tree, image)))
    a = rgba[:, 3].astype(np.uint8)
    chroma = rgba[:, :3].max(axis=1) - rgba[:, :3].min(axis=1)
    return Drawn(image.name if image is not None else None, float((a >= 128).mean()), float(chroma.mean()) < GREY,
                 alpha_mode(meaning, a)['mode'])


def _alpha_mode(shader):
    """The glTF alpha mode Blender's exporter writes for a Principled BSDF: MASK for an alpha cut by a Math node
    (round, greater or less than), BLEND for any other alpha that is not 1, else OPAQUE."""
    if shader is None:
        return 'OPAQUE'
    socket = shader.inputs['Alpha']
    if socket.is_linked:
        node = socket.links[0].from_node
        cut = node.type == 'MATH' and node.operation in ('ROUND', 'GREATER_THAN', 'LESS_THAN')
        return 'MASK' if cut else 'BLEND'
    return 'BLEND' if socket.default_value < 1 else 'OPAQUE'


def _kind(shader):
    """'glow' for a shader drawing an emissive image (export/write.py: plan_alpha_modes' meaning), else None."""
    if shader is None or not shader.inputs['Emission Color'].is_linked:
        return None
    strength = shader.inputs.get('Emission Strength')
    return 'glow' if strength is None or strength.default_value > 0 else None


def _image_node(tree, image):
    return next((n for n in tree.nodes if n.type == 'TEX_IMAGE' and n.image == image), None)


def _uvs(copy, node):
    """(k, 2) UV points of ``copy``'s faces: each face's centre and its corners a third of the way in, on the UV layer
    ``node`` reads (a UV Map node on its Vector input), else the one the mesh renders with. None with no UV layer."""
    me = copy.obj.data
    layer = None
    if node is not None and node.inputs['Vector'].is_linked:
        source = node.inputs['Vector'].links[0].from_node
        if source.type == 'UVMAP' and source.uv_map:
            layer = me.uv_layers.get(source.uv_map)
    layer = layer or next((l for l in me.uv_layers if l.active_render), None) or me.uv_layers.active
    if layer is None:
        return None
    uv = np.empty(2 * len(me.loops))
    layer.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    out = []
    for f in copy.faces:
        p = me.polygons[int(f)]
        corners = uv[p.loop_start:p.loop_start + p.loop_total]
        centre = corners.mean(axis=0)
        out.append(centre)
        out.extend(corners + (centre - corners) / 3)
    return np.array(out)


def _sample(pixels, uv):
    """The texels of ``pixels`` ((h, w, 4), top row first) at ``uv`` (repeating); every texel when ``uv`` is None."""
    h, w = pixels.shape[:2]
    if uv is None or not len(uv):
        return pixels.reshape(-1, 4)
    x = (np.mod(uv[:, 0], 1.) * w).astype(int) % w
    y = ((1. - np.mod(uv[:, 1], 1.)) * h).astype(int) % h
    return pixels[y, x]


def extends(name, base):
    """Whether object name ``name`` is ``base``'s with more after it, from a separator on (``msh_body_UV1`` of
    ``msh_body``, ``Body.UV1`` of ``Body``; not ``meshnode_12`` of ``meshnode_1``), Blender's .001 left out."""
    name, base = _TAKEN.sub('', name), _TAKEN.sub('', base)
    return len(name) > len(base) and name.startswith(base) and not name[len(base)].isalnum()


def rank(group, seen):
    """``group`` (opaque copies of one surface) best first, as ``drop_twins`` keeps them (see the module's
    docstring): in colour, then named after by the others, then by name. ``seen``: {copy: Drawn}."""
    def named_after(c):
        return any(extends(o.obj.name, c.obj.name) for o in group)
    return sorted(group, key=lambda c: (seen[c].grey, not named_after(c), c.name))


def passes(base, group, seen):
    """The copies of ``group`` that are ``base``'s passes: grey where it is in colour, or named after it."""
    return [c for c in group if c is not base and ((seen[c].grey and not seen[base].grey)
                                                    or extends(c.obj.name, base.obj.name))]


def drop_twins(meshes, keep=()):
    """Drop the copies of a surface the game cannot draw (see the module's docstring): of twins, the opaque copies
    but the one that draws the surface; an opaque lining behind a translucent surface. ``keep``: the objects a detail
    draws, never read nor dropped. -> record: {'twins': [...], 'linings': [...], 'triangles_dropped': {copy:
    triangles}}, a row for each set of twins and each lining, naming its copies, what each draws, the ones kept and
    the ones dropped."""
    import bpy
    import bmesh
    found = copies(meshes, keep)
    record = {'twins': [], 'linings': [], 'triangles_dropped': {}}
    if len(found) < 2:
        return record
    points = np.concatenate([np.asarray(c.verts) for c in found])
    size = float(np.ptp(points, axis=0).max()) or 1.0
    found_pairs = pairs(found, size)
    if not found_pairs:
        return record
    images, seen = {}, {}

    def what(c):
        if c not in seen:
            seen[c] = drawn(c, images)
        return seen[c]
    parent = {c: c for c in found}

    def find(c):
        while parent[c] is not c:
            c = parent[c]
        return c
    for a, b, kind in found_pairs:
        if kind == 'twins':
            parent[find(a)] = find(b)
    groups = {}
    for c in found:
        groups.setdefault(find(c), []).append(c)
    dropped = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        opaque = rank([c for c in group if what(c).opaque], seen)
        gone = passes(opaque[0], opaque, seen) if opaque else []
        dropped.update(gone)
        record['twins'].append({'kept': [c.name for c in group if c not in gone], 'dropped': [c.name for c in gone],
                                'copies': {c.name: seen[c].record() for c in group}})
    for a, b, kind in found_pairs:
        if kind != 'lining' or a in dropped or b in dropped:
            continue
        clear = {c: what(c).mode == 'Blend' for c in (a, b)}
        row = {'copies': {c.name: seen[c].record() for c in (a, b)}}
        if clear[a] != clear[b]:
            glass, lining = (a, b) if clear[a] else (b, a)
            dropped.add(lining)
            row.update(kept=glass.name, dropped=[lining.name],
                       why='an opaque lining behind a translucent surface is seen through it as a solid wall')
        else:
            row.update(kept=[a.name, b.name], dropped=[],
                       why='both translucent' if clear[a] else 'the surface and its own back, both opaque')
        record['linings'].append(row)
    triangles = {}
    for o in {c.obj for c in dropped}:
        gone = sorted(int(f) for c in dropped if c.obj is o for f in c.faces)
        triangles.update({c.name: int(sum(len(o.data.polygons[int(f)].vertices) - 2 for f in c.faces))
                          for c in dropped if c.obj is o})
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in gone], context='FACES')
        bm.to_mesh(o.data)
        bm.free()
        if not o.data.polygons:
            bpy.data.objects.remove(o, do_unlink=True)
    record['triangles_dropped'] = triangles
    return record
