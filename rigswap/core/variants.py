"""The model's pieces, when its file holds every look the game switches between: what they are, and drawn.

A game's model file often carries each form a part can take: 15-60 face expressions on one head bone, a sword in the
hand and the same sword on the back, a hand open and in a fist, a cap on and off, a fine and a coarse copy. Drawn all
at once they stack. What the files say, by source (each an export format's own convention, the same for every model
from it; looked at on 19 Ultimate, 4 Smash 4 and 3 Brawl rips and Mario 64, 2026-10-25):

- Smash Ultimate and Smash 4 rips (FBX and DAE from the community MaxScripts): a mesh the game switches on and off by
  animation has ``_VIS`` in its name, and the text before it names the switch (``Mario_Blink_VIS_O_OBJ``,
  ``link_kenb_VIS_O_OBJ``, ``RacketMFLIP_VIS_O_OBJ``); its ``.001`` copies and Smash 4's ``__Layer_2`` are the same
  switch's parts per material. A mesh without it is always drawn. No default state survives: every node is visible
  (the DAE's ``primary_visibility`` is 1 throughout, the FBX has none) and the motions that set it aren't in the rip.
  Bodies are often switched too (``samus_body_VIS``, ``koopa_body_VIS``, ``Ness_Head_VIS``), so which switched meshes
  make the everyday look is a semantic call. ``_O_OBJ``, ``_O_NOSORT_FAR``, ``_O_SORTEACHNODE``... are draw flags.
- A level of detail named by word: ``body_c00_high`` and ``body_c00_low``, ``hair_high`` and ``hair_low`` (Ultimate
  Link, Roy), ``koopa_Shell_low`` beside the switched ``koopa_Shell_VIS_O_OBJ`` (Bowser). The coarser copy fills the
  same box as the finer. (Load drops the LOD0, LOD1, ... spelling already: load.open.finest_levels.)
- Brawl rips (BrawlBox DAE, saved again by Blender): mesh names are lost (``mesh0``, ``mesh1``...); the switch was each
  mesh's visibility bone, and props ride their own bones (``KenM`` in the hand, ``KenMb`` on the back), so nothing in
  the file says which are switched.
- Mario 64 and other N64 / OBJ rips: rigid pieces on one bone each, every hand pose and cap state its own piece
  (``mario_left_hand_fist`` and ``_open``, ``mario_cap_on`` and ``_off``); no hidden objects or visibility flags.

What is here is structure: the pieces (units: a switch the file names, else a mesh of one name on bones in common),
their boxes and shapes, the coarser levels of detail the file names (coarser), the words for the weapons a fighter
holds (holding), the drop of meshes from the scene, and the grouped tiles assemble.py asks with (draw, schema).

Which pieces the everyday look shows is decided in look.py, by one look at all of them, and nothing here overrides
it. Until 2026-11-18 it was decided here: sets of "forms of one part" read from overlapping boxes, three voted asks a
set, and rules after the answers (a blink never, a neutral face by name, one form per surface place by its open
edges, a hand holds one form, copies one stays, pieces hosted on dropped ones go). Each was a prediction about the
next file, and between them they dropped what the answers kept: Kamek's hands, Diddy's eyes, Frieza's head shell and
forearms. The file conventions above are now told to the look as a glossary (look.GLOSSARY), not applied.
"""
import re
from pathlib import Path

import numpy as np
from .render import still

_TAKEN = re.compile(r'\.\d{3}$')                              # Blender's suffix for a name already taken
_SHAPE = re.compile(r'Shape\d*$')                             # a Maya-style shape node's name, as the FBX keeps it
_LAYER = re.compile(r'_+Layer_\d+$')                          # Smash 4 DAE: one switch's second layer
_FLAGS = re.compile(r'(?:_VIS)?(?:_O_[A-Z0-9]+(?:_[A-Z0-9]+)*)*_*$')    # Ultimate / Smash 4: _VIS, then _O_OBJ ...
_SWITCHED = re.compile(r'_VIS(?:_|$)')
_LEVEL_WORD = re.compile(r'(?:(?<![A-Za-z])(?i:high|mid|low)|(?<=[a-z])(?:High|Mid|Low))(?![a-z])')
LEVELS = {'high': 0, 'mid': 1, 'low': 2}
COPY = 0.05             # the same shape: principal extents within this share of the larger, triangles within 2%
TILE = 384
THROUGH = 0.6           # draw(through=True): the members alone laid over again at this opacity


def plain(name):
    """The name with Blender's, the exporter's and the game's additions taken off: the switch or the part it names."""
    n = _SHAPE.sub('', _TAKEN.sub('', name))
    return _FLAGS.sub('', _LAYER.sub('', n)) or n


def level(name):
    """(base, level) a plain name declares: load's LOD0, LOD1 ... or the words high, mid, low; None when it names
    none."""
    from ..load.open import level_of_detail
    found = level_of_detail(name)
    if found is not None:
        return found
    words = list(_LEVEL_WORD.finditer(name))
    if not words:
        return None
    m = words[-1]
    return _base(m.string[:m.start()] + ' ' + m.string[m.end():]), LEVELS[m.group(0).lower()]


def _base(text):
    return re.sub(r'[^a-z]+', ' ', re.sub(r'\d+', ' ', text.lower())).strip()


class Unit:
    """One switch (or one piece, where the file names none): its meshes, its rows in the rig, its box and shape."""

    def __init__(self, key, rig, meshes, rows):
        self.key, self.meshes, self.rows = key, meshes, rows
        pts = rig.points[rows]
        self.lo, self.hi = pts.min(0), pts.max(0)
        self.centre = (self.lo + self.hi) / 2
        w = rig.weights[rows].sum(0)
        self.bones = set(np.flatnonzero(w > 1e-6 * len(rows)))
        self.bone = int(np.argmax(w)) if w.sum() > 0 else -1
        self.extents = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(pts.T)), 0)) if len(pts) > 3 else np.zeros(3)
        self.triangles = 0


def units(rig):
    """The rig's meshes grouped into units: one per switch the file names (the text before _VIS), else one per mesh
    name without Blender's suffix, of the meshes riding bones in common (a piece cut by material). A mesh of that name
    riding none of their bones is the same piece in another place, a unit of its own, keyed by its own name: Frieza's
    ``gu`` and ``gu.001``, a fist in each hand, made one unit over both hands that rode no bone of its own, so no hand
    form was ever a candidate and every form showed in both hands. -> (units, named): named when the file names its
    switches."""
    import bpy
    names = list(rig.meshes)
    named = any(_SWITCHED.search(n) for n in names)
    groups, start = {}, 0                         # key -> its places, each [meshes, rows, bones]
    for n in names:
        k = len(bpy.data.objects[n].data.vertices)
        rows = range(start, start + k)
        bones = set(np.flatnonzero(rig.weights[start:start + k].sum(0) > 1e-6 * k)) if k else set()
        start += k
        if named and _SWITCHED.search(n):         # a switch is one unit, wherever its meshes ride
            g = groups.setdefault(plain(n), [[[], [], set()]])[0]
        else:
            places = groups.setdefault(_TAKEN.sub('', n), [])
            g = next((g for g in places if g[2] & bones or not (g[2] or bones)), None)
            if g is None:
                g = [[], [], set()]
                places.append(g)
        g[0].append(n)
        g[1].extend(rows)
        g[2].update(bones)
    pieces = []
    for key, places in groups.items():            # the place of the bare name keeps it, the others their own names
        bare = next((g for g in places if key in g[0]), places[0])
        pieces += [(key if g is bare else g[0][0], g[0], g[1]) for g in places]
    out = []
    for key, meshes, rows in pieces:
        u = Unit(key, rig, meshes, np.array(rows))
        u.switched = named and any(_SWITCHED.search(m) for m in meshes)
        u.triangles = sum(_triangles(bpy.data.objects[m].data) for m in meshes)
        out.append(u)
    return out, named


def _triangles(me):
    corners = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get('loop_total', corners)
    return int((corners - 2).sum())


def coarser(us, size):
    """The units the file names a coarser level of detail of another unit that fills the same box: {key: finer key}."""
    by_base = {}
    for u in us:
        found = level(plain(u.key))
        if found is not None:
            by_base.setdefault(found[0], []).append((u, found[1]))
    for u in us:                                  # unmarked, beside marked levels with no finest: it is the finest
        if level(plain(u.key)) is None:
            base = _base(plain(u.key))
            if base in by_base and min(lv for _, lv in by_base[base]) > 0:
                by_base[base].append((u, -1))
    out = {}
    for members in by_base.values():
        top = min(lv for _, lv in members)
        finest = [u for u, lv in members if lv == top]
        for u, lv in members:
            if lv == top:
                continue
            match = next((f for f in finest if np.abs(f.lo - u.lo).max() <= 0.05 * size
                          and np.abs(f.hi - u.hi).max() <= 0.05 * size), None)
            if match is not None:
                out[u.key] = match.key
    return out


def _boxes(a, b, pad):
    """(intersection over union, share of the smaller inside the other) of the two padded boxes."""
    lo = np.maximum(a.lo, b.lo) - pad
    hi = np.minimum(a.hi, b.hi) + pad
    inter = np.prod(np.clip(hi - lo, 0, None))
    va, vb = np.prod(a.hi - a.lo + 2 * pad), np.prod(b.hi - b.lo + 2 * pad)
    return inter / (va + vb - inter), inter / min(va, vb)


def _copies(a, b):
    if abs(a.triangles - b.triangles) > max(1, 0.02 * max(a.triangles, b.triangles)):
        return False
    return bool(np.abs(a.extents - b.extents).max() <= COPY * max(a.extents.max(), b.extents.max(), 1e-12))


# what the fighter holds through play, said to a set with a piece in a hand: the everyday look is a game's idle, and
# the game the model is from may stow its weapons there (Ocarina of Time's Link idles with his sword and shield on his
# back, tlink 2026-10-28: the ask kept them there); in this one they are in its hands. A weapon the file has only
# stowed then goes, and the fighter keeps the game's own (tlink's Hylian Shield, only on his back); telling the ask to
# keep such a one was tried the same day and turned its answers to his Mirror Shield, hammer and bow
HELD = (' This character fights holding {weapons} through play, so its everyday look holds {them} out, {each}whole '
        'and once, and leaves out {their} copies stowed on its back or in a sheath (an empty sheath stays).')


def holding(weapons):
    """The question's words for the weapons (kinds) a fighter holds through play: '' for none."""
    if not weapons:
        return ''
    if len(weapons) == 1:
        return HELD.format(weapons=f'its {weapons[0]} in its hand', them='it', each='', their='its')
    return HELD.format(weapons=' and '.join(f'its {w}' for w in weapons) + ', one in each hand', them='them',
                       each='each ', their='their')


def schema(rows):
    """An answer per group (its letter): the numbers of its members kept; and cant_tell."""
    props, n = {}, 0
    for g, row in enumerate(rows):
        numbers = list(range(n + 1, n + len(row) + 1))
        props[_letter(g)] = {'type': 'array', 'items': {'type': 'integer', 'enum': numbers}}
        n += len(row)
    props['cant_tell'] = {'type': 'boolean'}
    return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}


def surfaces(rig):
    """A file of one mesh names none of its pieces, but its maker built it of them: the surfaces welded apart from the
    rest (segments._pieces), those painted with the same materials together (a kart's four wheels; a head of hair
    cards is one piece, not a tile for each card), are its pieces, each made a mesh of its own, so the look sees each
    on a tile of its own and leaves out what the character is only shown with in its game, and a complaint can name
    one. The largest keeps the mesh; the others are ``<name> 2``, ``<name> 3``... largest first, so a record names the
    same piece again. A surface no wider than a pixel of the look's picture of the whole has no tile to show on: it
    is a detail of the piece it sits on, and goes with the seen surface nearest it (Taz's kart has crumbs for
    headlights and a cluster under its seat; left with him, the largest, they floated beside him once the kart was
    gone, and were offered as his eyes).

    Taz (Looney Tunes Racing OBJ, 2026-11-19) is one mesh with his go-kart, its four wheels and a floor shadow: one
    piece, never offered, so they rode along and were sized as him, the kart five sixths of the room he was given.
    -> (the rig read again, what whole() needs to make the pieces kept one mesh again); (the same rig, None) for a
    file of more meshes, or of one piece."""
    import bpy
    from . import look, rig as rigs, segments
    if len(rig.meshes) != 1:
        return rig, None
    ob = bpy.data.objects[rig.meshes[0]]
    me = ob.data
    points, tris, _ = segments._surface([ob])
    if not len(tris):
        return rig, None
    on = segments._pieces(points, tris)
    size = float(np.ptp(points, axis=0).max())
    welded = {}
    for p in me.polygons:
        welded.setdefault(int(on[p.vertices[0]]), []).append(p.index)

    def seen(faces):
        corners = sorted({v for f in faces for v in me.polygons[f].vertices})
        return float(np.ptp(points[corners], axis=0).max()) > size / look.VIEW
    painted, crumbs = {}, []        # the surfaces seen, by the materials they are painted with
    for faces in welded.values():
        if seen(faces):
            painted.setdefault(frozenset(me.polygons[f].material_index for f in faces), []).extend(faces)
        else:
            crumbs.append(faces)
    if crumbs and len(painted) > 1:
        from mathutils.bvhtree import BVHTree
        of = [(f, faces) for faces in painted.values() for f in faces]
        tree = BVHTree.FromPolygons(points.tolist(), [list(me.polygons[f].vertices) for f, _ in of])
        for faces in crumbs:
            corners = sorted({v for f in faces for v in me.polygons[f].vertices})
            of[tree.find_nearest(points[corners].mean(0).tolist())[2]][1].extend(faces)
    parts = sorted((sorted(faces) for faces in painted.values()), key=lambda faces: (-len(faces), faces[0]))[1:]
    if not parts:
        return rig, None
    made = {'name': ob.name, 'mesh': me.name, 'whole': me.copy(), 'faces': {}}
    for n, faces in enumerate(parts, 2):
        part = ob.copy()
        part.data = me.copy()
        part.name = part.data.name = f'{ob.name} {n}'
        for collection in ob.users_collection:
            collection.objects.link(part)
        _keep_faces(part.data, set(faces))
        made['faces'][part.name] = faces
    made['faces'][ob.name] = sorted(set(range(len(me.polygons))) - {f for faces in parts for f in faces})
    _keep_faces(me, set(made['faces'][ob.name]))
    return rigs.scene(rig.blend), made


def whole(rig, made):
    """The pieces surfaces() made, those the look kept, one mesh again: the file's own mesh without the faces of the
    pieces left out, as it would be had its maker left them out. The pieces are how the look is asked and how a
    record names what it leaves out; every later step reads the model as its file has it, one surface with one skin
    (Goku's one mesh, nothing of it left out, read in six pieces had its chains pictured and named anew, and came
    out with a patch of his chest at his shoulder: r2-pieces-2, 2026-11-19). ``made``: surfaces()'s; None, the rig
    as it is. -> the rig read again."""
    import bpy
    from . import rig as rigs
    if made is None:
        return rig
    kept = [n for n in made['faces'] if n in bpy.data.objects]
    if not kept:
        return rig
    ob = bpy.data.objects.get(made['name']) or bpy.data.objects[kept[0]]
    for name in made['faces']:
        other = bpy.data.objects.get(name)
        if other is not None and other is not ob:
            data = other.data
            bpy.data.objects.remove(other, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
    was, ob.data = ob.data, made['whole']
    if was.users == 0:
        bpy.data.meshes.remove(was)
    if len(kept) < len(made['faces']):
        _keep_faces(ob.data, {f for n in kept for f in made['faces'][n]})
    ob.name, ob.data.name = made['name'], made['mesh']
    return rigs.scene(rig.blend)




def _keep_faces(me, keep):
    """The mesh ``me`` with only the faces ``keep`` (their indices), and what they use."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context='FACES')
    bm.to_mesh(me)
    me.update()
    bm.free()


def drop(rig, names):
    """Remove the meshes ``names`` from the open scene and read the rig again without them (the same rig if none)."""
    import bpy
    from . import rig as rigs
    for name in names:
        ob = bpy.data.objects[name]
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data.users == 0:
            bpy.data.meshes.remove(data)
    return rigs.scene(rig.blend) if names else rig


def _letter(g):
    return chr(65 + g) if g < 26 else chr(65 + g // 26 - 1) + chr(65 + g % 26)


def draw(us, rows, pooled, up, front, context, size, path, through=False):
    """Tile 0: the set's members together; then the groups (``rows``), tile j member j alone. Each in its own colours,
    the rest of the body grey, the other candidates (``pooled``, mesh names) left out, seen from the side of the body
    the member sits on (against ``context``, the points always drawn, near it), or from its flank when it is far wider
    seen so; close on it. With ``through``, the members alone are laid over again at THROUGH, so one inside or behind
    what covers it (an ink in its tank's glass) is seen. Returns the pictures: tile 0, then one per group."""
    import bpy
    from mathutils import Matrix
    from PIL import Image, ImageDraw
    from .render import _base_color_image, _font, _prepare_scene
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE
    up = np.asarray(up, float) / np.linalg.norm(up)
    front = np.asarray(front, float) - (np.asarray(front, float) @ up) * up
    front /= np.linalg.norm(front)
    members = [u for row in rows for u in row]
    letter = {u.key: _letter(g) for g, row in enumerate(rows) for u in row}
    candidates = set(pooled) | {m for u in members for m in u.meshes}
    grey = bpy.data.materials.new('core_variants_grey')
    grey.diffuse_color = (0.4, 0.42, 0.45, 1.0)
    saved = []
    for u in us:
        for m in u.meshes:
            ob = bpy.data.objects[m]
            ob.hide_render = m in candidates
            if m not in candidates:
                for slot in ob.material_slots:
                    saved.append((slot, slot.link))
                    slot.link = 'OBJECT'
                    slot.material = grey
    # a member's material drawing no colour image shows its colour: Workbench would draw the first image it has (the
    # Inkling's ink drew its normal map, purple)
    blanked = [(node, node.image) for mat in {s.material for u in members for m in u.meshes
                                              for s in bpy.data.objects[m].material_slots}
               if mat is not None and mat.node_tree is not None and _base_color_image(mat) is None
               for node in mat.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image is not None]
    for node, _ in blanked:
        node.image = None
    cam = bpy.data.objects.new('core_variants', bpy.data.cameras.new('core_variants'))
    cam.data.type = 'ORTHO'
    scene.collection.objects.link(cam)
    scene.camera = cam
    tiles = []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for j, shown in enumerate([members] + [[u] for u in members]):
            for u in members:
                for m in u.meshes:
                    bpy.data.objects[m].hide_render = u not in shown
            lo = np.min([u.lo for u in shown], axis=0)
            hi = np.max([u.hi for u in shown], axis=0)
            c = (lo + hi) / 2
            reach = (hi - lo) / 2 + 0.05 * size
            near = context[np.all(np.abs(context - c) <= reach, axis=1)]
            ref = np.median(near if len(near) >= 10 else context, axis=0)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            across = np.cross(up, front)
            facing = front if (c - ref) @ front >= -0.01 * size else -front
            flank = across if (c - ref) @ across >= 0 else -across

            def shown_area(v):
                return np.ptp(corners @ np.cross(up, v)) * np.ptp(corners @ up)
            # from the front or back, unless the part is far wider seen from its side (a sword held pointing ahead)
            side = flank if shown_area(flank) > 2 * shown_area(facing) else facing
            right = np.cross(up, side)
            span = max(np.ptp(corners @ right), np.ptp(corners @ up))
            cam.data.ortho_scale = max(1.5 * span, 0.22 * size)
            cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 10
            rot = np.column_stack([right, up, side])
            m4 = np.eye(4)
            m4[:3, :3], m4[:3, 3] = rot, c + side * size * 3
            cam.matrix_world = Matrix(m4.tolist())
            tile = path.with_name(f'{path.stem}-{j}.png')
            scene.render.filepath = str(tile)
            still()
            img = Image.new('RGB', (TILE, TILE), (205, 210, 218))
            with Image.open(tile) as rendered:
                rgba = rendered.convert('RGBA')
            tile.unlink()                         # only the group pictures are kept
            img.paste(rgba, (0, 0), rgba)
            if through:                           # the members alone again, laid over what covers them
                rest = [bpy.data.objects[m] for u in us if u not in shown for m in u.meshes
                        if not bpy.data.objects[m].hide_render]
                for ob in rest:
                    ob.hide_render = True
                still()
                for ob in rest:
                    ob.hide_render = False
                with Image.open(tile) as rendered:
                    alone = rendered.convert('RGBA')
                tile.unlink()
                alone.putalpha(alone.getchannel('A').point(lambda a: int(a * THROUGH)))
                img.paste(alone, (0, 0), alone)
            d = ImageDraw.Draw(img)
            d.text((10, 6), str(j), fill='black', font=_font(40, bold=True))
            if j:
                d.text((TILE - 10, 6), letter[shown[0].key], fill='black', font=_font(40, bold=True), anchor='ra')
            seen = 'back' if side @ front < -0.5 else 'side' if abs(side @ front) < 0.5 else ''
            if seen:
                d.text((TILE - 10, 52), seen, fill='black', font=_font(22), anchor='ra')
            tiles.append(img)
    finally:
        for slot, link in saved:
            slot.material = None
            slot.link = link
        for node, image in blanked:
            node.image = image
        bpy.data.objects.remove(cam)
        bpy.data.materials.remove(grey)
        for u in us:
            for m in u.meshes:
                bpy.data.objects[m].hide_render = False
    gap, out, start = 8, [path.with_name(f'{path.stem}-all.png')], 1
    tiles[0].save(out[0])
    for g, row in enumerate(rows):                # a picture per group, so that a big set's tiles stay big
        cols = min(4, len(row))
        lines = (len(row) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * TILE + (cols - 1) * gap, lines * TILE + (lines - 1) * gap), 'white')
        for n, t in enumerate(tiles[start:start + len(row)]):
            sheet.paste(t, ((n % cols) * (TILE + gap), (n // cols) * (TILE + gap)))
        out.append(path.with_name(f'{path.stem}-{_letter(g)}.png'))
        sheet.save(out[-1])
        start += len(row)
    return out
