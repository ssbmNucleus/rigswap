"""The Melee plugin's read-back checks: what a trip through the game's file kept, two skeletons and two models
compared. ``export`` reads the costume it wrote back and runs them (each a dict whose ``pass`` says whether it held;
``require_checks`` fails the export on any that did not); ``dae_to_glb`` checks a target's .glb against its .dae with
``compare_joints`` and ``tree_differs``.

    tolerances      how far a number may move and still be the same: the .dae's own precision (JOINT_TOLERANCE),
                    HSDRawViewer's vertex compression (VERTEX_TOLERANCE), conversion noise in weights and winding
    two skeletons   compare_joints (JointCheck), tree_differs
    two models      compare_models, triangle_matches (material_looks tells coincident layers apart), read_place (a
                    read-back mesh's DObj place), without_kept (the read-back without the kept vanilla parts, which
                    check_parts checks byte for byte)
    the checks      material_check, preservation_checks (materials, weights, winding), position_check,
                    image_count_check and material_count_check (the .glb against the .dae, before the import),
                    dobj_count_check (the DObjs the join counted against the ones the import made), forms_check (every
                    form of the build's object draws something),
                    dobj_image_check (real image buffers in the file: image_buffers, check_images),
                    texture_format_check (the formats prep asked for), alpha_mode_check (the alpha modes the import was
                    told, as the file has them behind the import's gate: import_transparent on the texels it wrote,
                    texel_alpha), check_palettes (the envelope palettes the file has), check_parts (the
                    kept vanilla parts), check_costume_size (the file against prep's share of memory), require_checks
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations, product
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

from ... import Failed
from ..dae import NS, PLACE
from ..dat import Dat, joints, joint_dobjs
from ..prep.memory import ARCHIVE_HEADER_BYTES, _in_units, costume_budget
from ..prep.normals import POSITION_ERROR
from .dobjs import HIGH_DETAIL, LOW_DETAIL, MAX_ENVELOPES, TOBJ_PASS


# A joint that moved less than this, in model units, did not move: the DAE text carries numbers to about six
# significant figures. Mario is about 14 units tall.
JOINT_TOLERANCE = 1e-4
# HSDRawViewer stores vertex positions in the smallest integer format that keeps each coordinate within 0.005 units
# (POSITION_ERROR), so a written vertex may sit 0.005 off on every axis at once: 0.005 times root 3 from the model's
# (s4-mario's small face overlays read back 0.00497 off on two axes, 0.0052 away).
VERTEX_TOLERANCE = POSITION_ERROR * 3 ** .5 + JOINT_TOLERANCE
# Read-back policies: at most 1% of a vertex's total influence may move to other joints, and at most 0.1% of
# matched triangles may run their corners the other way. These allow numeric conversion noise, not game limits.
WEIGHT_TOLERANCE = .01
WINDING_TOLERANCE = .001
# UVs closer than this, in UV units (a quarter of a texel of the largest texture prep writes), count as the same UVs
# when telling coincident triangles apart (triangle_matches). HSDRawViewer compresses UVs as it does positions, so a
# written UV moves a little; layers whose UVs differ at all differ by texels.
UV_TOLERANCE = 1e-3
# The side of the small picture two materials' images are compared by (material_looks): fine enough to tell two
# expressions' eyelids apart, coarse enough that the import's encoding (CMPR's 4x4 blocks) barely moves it.
LOOK_SIDE = 16


@dataclass
class JointCheck:
    worst: str | None       # the joint that moved most
    distance: float         # how far it moved
    missing: list           # joints in before, not in after
    extra: list             # joints in after, not in before
    matrix_error: float = 0.  # largest absolute element change of a full matrix

    def moved(self, tolerance=JOINT_TOLERANCE):
        return bool(self.missing or self.extra or not np.isfinite(self.distance + self.matrix_error)
                    or self.distance > tolerance or self.matrix_error > tolerance)

    def __str__(self):
        parts = [f'worst joint {self.worst} moved {self.distance:.6g}; matrix change {self.matrix_error:.6g}']
        if self.missing:
            parts.append(f'{len(self.missing)} joints missing ({", ".join(self.missing[:6])})')
        if self.extra:
            parts.append(f'{len(self.extra)} joints added ({", ".join(self.extra[:6])})')
        return '; '.join(parts)


def compare_joints(before, after, before_parents=None, after_parents=None):
    """Compare positions and full matrices; parent maps convert world matrices to each joint's local frame."""
    def locals_of(matrices, parents):
        if parents is None:
            return matrices
        return {n: np.linalg.solve(matrices[parents[n]], m) if parents.get(n) in matrices else m
                for n, m in matrices.items()}
    before, after = locals_of(before, before_parents), locals_of(after, after_parents)
    def pos(v):
        v = np.asarray(v, float)
        return v[:3, 3] if v.shape == (4, 4) else v
    shared = [n for n in before if n in after]
    worst, distance, matrix_error = None, 0., 0.
    for n in shared:
        d = float(np.linalg.norm(pos(before[n]) - pos(after[n])))
        if np.shape(before[n]) == np.shape(after[n]) == (4, 4):
            error = float(np.abs(np.asarray(before[n]) - np.asarray(after[n])).max())
            if not np.isfinite(error) or error > matrix_error:
                matrix_error, worst = error, n
        if worst is None or d > distance:
            worst, distance = n, d
    return JointCheck(worst, distance, [n for n in before if n not in after], [n for n in after if n not in before],
                      matrix_error)


def compare_models(before, after):
    """Numbers for a trip through the game's file: joints, triangles, materials, textures, where the vertices went,
    and how their weights changed. Each vertex after is matched to the nearest vertex before."""
    joints = compare_joints(before.joints, after.joints, before.parents, after.parents)
    out = {'joints': [len(before.joints), len(after.joints)], 'joint_check': str(joints),
           'joints_moved_max': joints.distance, 'triangles': [before.triangles, after.triangles],
           'joint_matrix_change_max': joints.matrix_error,
           'degenerate': [before.degenerate, after.degenerate], 'materials': [before.materials, after.materials],
           'textures': [len(set(before.textures.values()) - {None}), len(set(after.textures.values()) - {None})],
           'textures_missing': [sum(v is None for v in before.textures.values()),
                                sum(v is None for v in after.textures.values())],
           'textures_same_pixels_file': len((set(before.textures.values()) & set(after.textures.values())) - {None}),
           'vertices': [len(before.points), len(after.points)]}
    if len(before.points) and len(after.points):
        nearest, gap = _nearest(after.points, before.points)
        names = sorted({n for w in before.weights + after.weights for n in w})
        col = {n: i for i, n in enumerate(names)}

        def dense(ws):
            m = np.zeros((len(ws), len(names)))
            for i, w in enumerate(ws):
                for n, v in w.items():
                    m[i, col[n]] = v
            return m
        # A UV seam or a hard edge can put several influence sets at the same position. Compression can round
        # them to slightly different positions. Compare full sets within vertex export precision, both ways,
        # so a lost coincident influence set still fails.
        # The points of b within reach of p are looked for among those whose x is (a little more than) within reach,
        # b sorted by x once: the same points, each measured as before, so the same numbers.
        def changes(a, b, aw, bw, distances=None):
            if distances is None:
                _, distances = _nearest(a, b)
            by_x = np.argsort(b[:, 0], kind='stable')
            xs = b[by_x, 0]
            result = []
            for p, w, distance in zip(a, aw, distances):
                reach = max(VERTEX_TOLERANCE, distance + JOINT_TOLERANCE)
                margin = reach * 1.001 + 1e-9
                near = by_x[np.searchsorted(xs, p[0] - margin, 'left'):np.searchsorted(xs, p[0] + margin, 'right')]
                candidates = near[np.linalg.norm(b[near] - p, axis=1) <= reach]
                result.append(float((.5 * np.abs(bw[candidates] - w).sum(1)).min()))
            return result
        bw, aw = dense(before.weights), dense(after.weights)
        diff = np.array(changes(after.points, before.points, aw, bw, gap)     # gap is _nearest(after, before)'s
                        + changes(before.points, after.points, bw, aw))
        out.update(vertex_gap_max=float(gap.max()), vertex_gap_mean=float(gap.mean()),
                   weight_change_max=float(diff.max()), weight_change_mean=float(diff.mean()),
                   vertices_weight_changed_over_1pct=int((diff > .01).sum()))
    return out


def triangle_matches(before, after):
    """Match triangles one to one by all three corner positions; prefer the same way round at coincident faces.

    Source triangles on the same three corners (an eye layer drawn over the face; a rip's alternative expressions,
    each a mesh and material of its own on the same eyelid) are interchangeable by position, so which read-back
    triangle takes which would be arbitrary, and a swap puts a material of one image and format among the read-back
    materials of another: the texture-format and alpha checks then fail on a file the import wrote right. Among
    triangles at the same corners the read-back triangle takes, in turn, the one whose UVs at those corners are its
    own (UV_TOLERANCE), then the one whose material's image looks most like its read-back material's
    (material_looks), so a material is matched to the read-back material that carries it. When a read-back material
    still took triangles from more than one source material (layers with one UV layout and one look), the matching
    is run again preferring the source material that most of that read-back material's triangles come from."""
    materials = len(before.face_materials) == len(before.faces) and len(after.face_materials) == len(after.faces)
    looks = material_looks(before, after) if materials else None
    matched, flipped, missing = _match_triangles(before, after, None, looks)
    if not materials:
        return matched, flipped, missing     # no materials to tell the triangles apart by
    sources = {}
    for i, source in enumerate(matched):
        if source is not None:
            sources.setdefault(after.face_materials[i], {}).setdefault(before.face_materials[source], 0)
            sources[after.face_materials[i]][before.face_materials[source]] += 1
    if any(len(v) > 1 for v in sources.values()):
        return _match_triangles(before, after, {(s, r): n for r, v in sources.items() for s, n in v.items()}, looks)
    return matched, flipped, missing


def _match_triangles(before, after, affinity, looks=None):
    """The matching of triangle_matches. Between triangles at the same corners it prefers, in order: the same way round,
    the UVs (each corner's, as the corners matched), ``looks`` ({(source material, read-back material):
    material_looks' difference}), and ``affinity`` ({(source material, read-back material): triangles}), the pair
    most triangles make. A triangle's winding is the order its corners run in, read from which corner matched which:
    compression can carry a thin triangle's apex across its base and reverse its normal, with the order kept."""
    a = before.points[np.asarray(before.faces, int).reshape(-1, 3)]
    b = after.points[np.asarray(after.faces, int).reshape(-1, 3)]
    uvs = len(a) and len(b) and len(before.face_uvs) == len(a) and len(after.face_uvs) == len(b)
    a_uv = np.asarray(before.face_uvs, float).reshape(-1, 3, 2) if uvs else None
    b_uv = np.asarray(after.face_uvs, float).reshape(-1, 3, 2) if uvs else None
    cells = {}
    for i, centre in enumerate(a.mean(axis=1)):
        cells.setdefault(tuple(np.floor(centre / VERTEX_TOLERANCE).astype(int)), []).append(i)
    used, matched, flipped = set(), [], 0
    orders = np.array(list(permutations(range(3))))
    reversed_order = np.array([sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) % 2 for p in orders], bool)
    for k, tri in enumerate(b):
        cell = np.floor(tri.mean(axis=0) / VERTEX_TOLERANCE).astype(int)
        candidates = [i for delta in product((-1, 0, 1), repeat=3)
                      for i in cells.get(tuple(cell + delta), ()) if i not in used]
        choices = []
        gaps = uv_gaps = turns = ()
        if candidates:
            # each candidate's gap: over the orders of its corners, the least of the farthest corner (all at once);
            # its UV gap: the farthest corner's UV in that order
            spread = np.linalg.norm(tri - a[candidates][:, orders], axis=3).max(2)
            order = spread.argmin(1)
            gaps = spread[np.arange(len(candidates)), order]
            turns = reversed_order[order]
            uv_gaps = (np.linalg.norm(b_uv[k] - a_uv[candidates][np.arange(len(candidates))[:, None], orders[order]],
                                      axis=2).max(1) if uvs else np.zeros(len(candidates)))
        for i, gap, uv_gap, turned in zip(candidates, map(float, gaps), map(float, uv_gaps), turns):
            if gap <= VERTEX_TOLERANCE:
                pair = (before.face_materials[i], after.face_materials[k]) if looks or affinity else None
                look = looks.get(pair, 0.) if looks else 0.
                rank = -affinity.get(pair, 0) if affinity else 0
                choices.append((bool(turned), int(uv_gap // UV_TOLERANCE), look, rank, gap, i))
        if choices:
            turned, *_, i = min(choices)
            used.add(i)
            matched.append(i)
            flipped += turned
        else:
            matched.append(None)
    return matched, flipped, len(a) - len(used)


def material_looks(before, after):
    """{(source material, read-back material): how unlike their images look}, for each pair of materials the two
    models draw with: the mean difference of their first images' looks (_image_look), 0 for two materials with no
    image and 1 for one with an image and one without. The import's encoding moves a texture's look a little; another
    texture, even another expression's eyelid on the same UVs, moves it several times as much."""
    def look(model, material):
        images = model.material_images.get(material) or []
        return _image_look(images[0]) if images else None
    sources = {m: look(before, m) for m in set(before.face_materials)}
    reads = {r: look(after, r) for r in set(after.face_materials)}
    return {(m, r): (0. if s is None and t is None else 1. if s is None or t is None else float(np.abs(s - t).mean()))
            for m, s in sources.items() for r, t in reads.items()}


def _image_look(path):
    """An image's look (LOOK_SIDE a side, colour premultiplied by alpha, and alpha), or None when it is not a picture."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return _image_look_of(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=512)
def _image_look_of(path, _modified, _size):
    from PIL import Image
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGBA').resize((LOOK_SIDE, LOOK_SIDE), Image.Resampling.BOX),
                                float) / 255.
    except (OSError, ValueError):
        return None
    return np.concatenate([pixels[..., :3] * pixels[..., 3:], pixels[..., 3:]], axis=-1)


def read_place(name):
    """The DObj place in a read-back mesh's name (``--names place``: Joint_<joint>_Object_<place>), or None."""
    match = re.fullmatch(r'Joint_\d+_Object_(\d+)', name or '')
    return int(match[1]) if match else None


def material_check(before, after, eyes=(), alike=None, padded=(), matched=None):
    """The materials the source's triangles draw with against the read-back's. The read-back names a material by its
    bytes, not its place (HSDRawViewer ModelExporter.cs: Material_<FNV hash of the MObj and every structure it points
    at, each running to the next pointed-at offset in the file>), so it counts distinct materials as drawn, which the
    import keeps one to one with the .dae's, except that two the import writes alike are the same bytes.

    ``alike`` ({source material: what the import writes for it}, write.py: written_alike) counts the source's
    materials as the import writes them, and the record lists under ``written_alike`` the ones it writes alike:
    two materials that draw one image read back as one material, which a
    count of the .dae's materials called a loss. A material that is lost still fails: one the import writes
    differently from every other reads back as its own.

    ``eyes`` is what eyes.py moved ({'object', 'to_place'} each). It draws those objects at the animated places with
    materials of its own, a copy of the eye's for each place with that vanilla place's TObj slots, and how many
    materials those copies read back as is not something kept or lost: two copies (each with the
    target's second slot) are the same bytes and read back as one, not the eye's old one; or one copy reads back as the
    eye's old material and another as another, only because unreferenced bytes follow its TObj in the file. So the eye
    objects and their places are left out on both sides; the eye places' images and
    formats are checked by place (``dobj_image_check``, ``texture_format_check``). A material the eyes share with other
    triangles still counts where those are drawn.

    ``padded`` (pokecast, Blastoise) are the places whose material the export pads with blank TObjs up to the vanilla
    place's count (write.py material_at), so a material drawn there is other bytes than the same material drawn at a
    place that isn't: one written material read back as two is no loss when all but one of them are drawn only at
    padded places (``matched``: triangle_matches, the source face of each read-back face). Those count once, listed
    under ``padded_alike``; a read-back material drawing two written materials is still a loss."""
    written = (lambda m: alike.get(m, m)) if alike else (lambda m: m)
    check = {'before': len({written(m) for m in before.face_materials}) if alike else before.materials,
             'after': after.materials}
    if eyes:
        objects, places = sorted({m['object'] for m in eyes}), sorted({m['to_place'] for m in eyes})
        check = {'before': len({written(m) for m, o in zip(before.face_materials, before.face_objects)
                                if o not in objects}),
                 'after': len({m for m, o in zip(after.face_materials, after.face_objects)
                               if read_place(o) not in places}),
                 'not_counted': {'eye_objects': objects, 'eye_places': places}, 'with_eyes': check}
    groups = {}
    for m in sorted(set(before.face_materials)):
        groups.setdefault(written(m), []).append(m)
    if any(len(g) > 1 for g in groups.values()):
        check['written_alike'] = sorted(g for g in groups.values() if len(g) > 1)
    if padded and matched is not None:
        skip = set(check.get('not_counted', {}).get('eye_places', ()))
        keys, at = {}, {}
        for face, source in enumerate(matched):
            place = read_place(after.face_objects[face])
            if source is None or place in skip:
                continue
            keys.setdefault(after.face_materials[face], set()).add(written(before.face_materials[source]))
            at.setdefault(after.face_materials[face], set()).add(place)
        reads = {}
        for r, ks in keys.items():
            if len(ks) == 1:
                reads.setdefault(next(iter(ks)), []).append(r)
        twins = sorted(sorted(rs) for rs in reads.values()
                       if len(rs) > 1 and sum(not at[r] <= set(padded) for r in rs) <= 1)
        if twins:
            check['padded_alike'] = twins
            check['after'] -= sum(len(rs) - 1 for rs in twins)
    check['pass'] = check['before'] == check['after']
    return check


def preservation_checks(before, after, numbers, *, eyes=(), alike=None, padded=()):
    matched, flipped, missing = triangle_matches(before, after)
    unmatched = sum(i is None for i in matched)
    fraction = flipped / max(1, len(matched))
    weight = numbers.get('weight_change_max')
    return {'materials': material_check(before, after, eyes, alike, padded, matched),
            'weights': {'maximum_change': weight, 'tolerance': WEIGHT_TOLERANCE,
                        'pass': weight is not None and np.isfinite(weight) and weight <= WEIGHT_TOLERANCE},
            'winding': {'flipped': flipped, 'fraction': fraction, 'tolerance': WINDING_TOLERANCE,
                        'unmatched': unmatched, 'missing': missing,
                        'pass': not (unmatched or missing) and fraction <= WINDING_TOLERANCE}}


def image_count_check(glb, dae):
    """An independent GLB baseline, before the DAE becomes the DAT read-back reference."""
    raw = Path(glb).read_bytes()
    length, kind = struct.unpack_from('<2I', raw, 12)
    if kind != 0x4E4F534A:
        raise Failed(f'{Path(glb).name}: its first chunk is not JSON')
    gltf = json.loads(raw[20:20 + length])
    # Melee draws base colour only. Normal, metallic/roughness and unused images are not exported to Collada.
    textures = gltf.get('textures', [])
    before = len({textures[base['index']]['source'] for m in gltf.get('materials', [])
                  if (base := m.get('pbrMetallicRoughness', {}).get('baseColorTexture')) is not None})
    root = ET.parse(dae).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    colour_images = set()
    for material in root.iter(NS + 'material'):
        instance = material.find(NS + 'instance_effect')
        effect = ids.get(instance.get('url', '').lstrip('#')) if instance is not None else None
        if effect is None:
            continue
        params = {p.get('sid'): p for p in effect.iter(NS + 'newparam')}
        for diffuse in effect.iter(NS + 'diffuse'):
            for texture in diffuse.iter(NS + 'texture'):
                ref = texture.get('texture')
                seen = set()
                while ref in params and ref not in seen:
                    seen.add(ref)
                    param = params[ref]
                    source = param.find('.//' + NS + 'source')
                    if source is None:
                        source = param.find('.//' + NS + 'init_from')
                    ref = source.text.strip() if source is not None and source.text else None
                if ref in ids and ids[ref].tag == NS + 'image':
                    colour_images.add(ref)
    after = len(colour_images)
    return {'before': before, 'after': after, 'pass': before == after}


def require_checks(checks, where):
    failed = {name: value for name, value in checks.items() if not value['pass']}
    if failed:
        raise Failed(f'{where}: preservation checks failed: {json.dumps(failed)}')


def position_check(numbers):
    gap = numbers.get('vertex_gap_max')
    return {'maximum_gap': gap, 'tolerance': VERTEX_TOLERANCE,
            'pass': gap is not None and np.isfinite(gap) and gap <= VERTEX_TOLERANCE}


def material_count_check(glb, dae):
    """Materials that faces actually use, both sides: a material whose only faces were dropped (an outline pass) is
    rightly gone, so the material list alone would count it on one side and not the other."""
    raw = Path(glb).read_bytes()
    length = struct.unpack_from('<I', raw, 12)[0]
    gltf = json.loads(raw[20:20 + length])
    used = {prim.get('material') for mesh in gltf.get('meshes', []) for prim in mesh.get('primitives', [])
            if prim.get('material') is not None}
    before = len(used)
    root = ET.parse(dae).getroot()
    after = len({el.get('material') for tag in ('triangles', 'polylist', 'polygons') for el in root.iter(NS + tag)
                 if el.get('material')})
    return {'before': before, 'after': after, 'pass': before == after}


def _nearest(a, b, reaches=(VERTEX_TOLERANCE, 16 * VERTEX_TOLERANCE), chunk=512):
    """For each point of a, the index of the nearest point of b (the first, by index, of those as near) and the
    distance to it. Each point is first measured against the points of b whose x lies within a reach of its own (b
    sorted by x once): when the nearest of them is nearer than that reach, no point outside can be as near, so it is
    the answer; else the next reach, and at last every point of b. Each distance is measured as one measured against
    all of b, so the numbers are the same."""
    index, gap = np.empty(len(a), int), np.empty(len(a))
    by_x = np.argsort(b[:, 0], kind='stable')
    xs = b[by_x, 0]
    far = []
    for k, p in enumerate(a):
        for reach in reaches:
            near = np.sort(by_x[np.searchsorted(xs, p[0] - reach, 'left'):np.searchsorted(xs, p[0] + reach, 'right')])
            if len(near):
                d = np.linalg.norm(b[near] - p, axis=1)
                i = d.argmin()
                if d[i] < reach * 0.999:
                    index[k], gap[k] = near[i], d[i]
                    break
        else:
            far.append(k)
    for s in range(0, len(far), chunk):          # against every point of b, many points at once
        ks = far[s:s + chunk]
        d = np.linalg.norm(a[ks, None, :] - b[None, :, :], axis=2)
        index[ks], gap[ks] = d.argmin(1), d.min(1)
    return index, gap


def tree_differs(joints, parents, other_joints, other_parents):
    """Why two joint trees differ, or None when they have the same joints, parents and order. Order matters: the game
    knows a joint by its place in the tree, not by name."""
    if list(joints) != list(other_joints):
        at = next((i for i, (a, b) in enumerate(zip(joints, other_joints)) if a != b), min(len(joints),
                                                                                          len(other_joints)))
        return (f'joint {at} is {list(joints)[at] if at < len(joints) else "missing"} in one and '
                f'{list(other_joints)[at] if at < len(other_joints) else "missing"} in the other '
                f'({len(joints)} and {len(other_joints)} joints)')
    for n in joints:
        if parents.get(n) != other_parents.get(n):
            return f'{n} hangs from {parents.get(n)} in one and {other_parents.get(n)} in the other'
    return None


def dobj_count_check(planned, imported, symbol):
    """The DObjs the join planned vanilla places for (prep/join.py: join_for_places, its ``dobjs``) against the ones
    the import made in ``imported``, before the vanilla list is put back around them. Both count one DObj for each mesh
    and material, the detail objects (the eyes) among them: a plan that counted fewer would fit where the import does
    not, and the export would fail later at choosing places, with the plan's count nowhere."""
    d = Dat.read(imported)
    made = sum(len(joint_dobjs(d, j)) for j in joints(d, symbol))
    return {'planned': int(planned), 'imported': made, 'pass': made == int(planned)}


def forms_check(dat, symbol, tables):
    """Every form of the object the build is in draws something: the build is model 0's option 0, so the game hides it
    whenever it shows another option of that model, in play or in the shadow and the magnifier (dobjs.py: Objects). An
    option with nothing to draw is the fighter vanishing for as long as the game shows it."""
    d = Dat.read(dat)
    listed = [dobj for j in joints(d, symbol) for dobj in joint_dobjs(d, j)]
    shown = {key: places for key, places in tables.options.items() if key[0] in (HIGH_DETAIL, LOW_DETAIL) and key[1] == 0}
    empty = sorted(key for key, places in shown.items() if not any(d.ptr(listed[p] + 0x0C) is not None for p in places))
    return {'options': len(shown), 'empty': empty, 'pass': not empty}


def dobj_image_check(dat, symbol, source, *readbacks):
    """Require actual DAT image buffers where matched source triangles used a base-colour image.

    Expectations come from the source, not the read-back material: a lost TObj must not turn a textured DObj into
    an apparently intentional untextured one. Place names survive material renames, joins and split PObj chains.
    """
    expected, unmapped = {}, 0
    for back in readbacks:
        matched, _, _ = triangle_matches(source, back)
        for face, original in enumerate(matched):
            if original is None:
                unmapped += 1
                continue
            material = source.face_materials[original]
            if not source.material_images.get(material):
                continue
            place = read_place(back.face_objects[face] if face < len(back.face_objects) else '')
            if place is None:
                unmapped += 1
                continue
            expected.setdefault(place, set()).add(material)
    check = check_images(image_buffers(dat, symbol), expected)
    check['unmapped_triangles'] = unmapped
    check['pass'] = check['pass'] and not unmapped
    return check


def texture_format_check(asked, before, after):
    """Map each drawn material's requested encoding to encodings on its matched read-back triangles.

    Only images referenced by drawn materials are read; dummy or stale PNGs beside the DAE are irrelevant.
    HSDRawViewer names the actual encoding Texture_<n>_<format>[_<palette>].png (CMP is its name for CMPR). A
    pass-through TObj the export padded a material with (dobjs.padded_material: an 8x8 I4 image of zeros) is no
    texture of it: an untextured material padded at a place with one TObj reads back with it and is still untextured.
    """
    def canonical(code):
        parts = code.upper().replace('_', ':').split(':')
        parts[0] = 'CMPR' if parts[0] == 'CMP' else parts[0]
        return ':'.join(parts)
    wanted = json.loads(Path(asked).read_text(encoding='utf-8')) if asked else {}
    matched, _, _ = triangle_matches(before, after)
    read_materials = {m: set() for m in before.face_materials}
    for i, source in enumerate(matched):
        if source is not None:
            read_materials[before.face_materials[source]].add(after.face_materials[i])
    rows = {}
    for material, read_ids in sorted(read_materials.items()):
        formats = []
        for read_id in sorted(read_ids):
            images = [p for p in after.material_images.get(read_id, [])
                      if not (Path(p).name.endswith('_I4.png') and _blank_image(p))]
            if not images:
                formats.append(None)
            for path in images:
                match = re.fullmatch(r'Texture_\d+_(.+)\.png', Path(path).name)
                formats.append(canonical(match[1]) if match and Path(path).is_file() else None)
        request = canonical(wanted[material]) if wanted.get(material) else None
        untextured = not before.material_images.get(material) and request is None
        rows[material] = {'requested': request, 'read_materials': sorted(read_ids),
                          'read_back': sorted(set(formats), key=lambda v: v or ''),
                          'pass': bool(formats) and all(v == request for v in formats)
                                  and (request is not None or untextured)}
    return {'materials': rows, 'prep_not_drawn': sorted(set(wanted) - set(rows)),
            'unmatched_triangles': sum(i is None for i in matched),
            'pass': bool(rows) and all(row['pass'] for row in rows.values()) and None not in matched}


def _blank_image(path):
    """Whether the image at ``path`` is the blank TObj's: 8x8, every texel zero (dobjs: GX_I4, IMAGE_ALIGN)."""
    from PIL import Image
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('L'))
    except OSError:
        return False
    return pixels.shape == (8, 8) and not pixels.any()


def alpha_mode_check(dat, symbol, source, back, planned):
    """Check actual MObj/TObj/PE flags, so an importer ignoring options fails (write.py: plan_alpha_modes planned them).

    Offsets and values: HSDRaw/Common/HSD_MOBJ.cs and HSD_TOBJ.cs;
    Test's PE descriptor: HSDRawViewer/Converters/ModelImporter.cs:ConfigureAlpha.

    The import's gate (facts/alpha.py): ConfigureAlpha gives a material the alpha operation of its mode only when its
    texture, as the import wrote it, is transparent (IsTransparent); else no alpha operation, no translucent render
    flags and no pixel engine compare, which is what "Opaque" reads back as. The gate is read here from the texels in
    the file (import_transparent), so the check holds each material to the mode it was told where its texture is
    transparent and to Opaque where it is not, and the plan's own reading of the gate (``transparent``, from the PNG
    and the format the import writes) must agree with the file's where the plan could tell (``plan_agrees``).

    The import makes one TObj for a textured material (ModelImporter.cs GenerateMaterial), and its alpha operation is
    the material's. A TObj whose flags are dobjs.TOBJ_PASS (colour and alpha both PASS) changes neither: it is a
    vanilla texture slot the eye step kept on an eye material so the game's texture indices stay fixed (eyes.py
    _eye_material: Zelda's eye DObjs have a second TObj), not a texture the import configured, so it is left out of
    the material's alpha operations and counted under ``passthrough_slots``.
    """
    d = Dat.read(dat)
    dobjs = [obj for joint in joints(d, symbol) for obj in joint_dobjs(d, joint)]
    matches, _, _ = triangle_matches(source, back)
    places = {key: set() for key in planned}
    for face, original in enumerate(matches):
        if original is None:
            continue
        match = re.fullmatch(r'Joint_\d+_Object_(\d+)', back.face_objects[face])
        if match and source.face_materials[original] in places:
            places[source.face_materials[original]].add(int(match[1]))
    rows = {}
    for key, expected in planned.items():
        found = []
        for place in sorted(places[key]):
            mobj = d.ptr(dobjs[place] + 8)
            flags = d.u32(mobj + 4)
            pe = d.ptr(mobj + 0x14)
            tobjs = d.chain(d.ptr(mobj + 8), 4)
            imported = [t for t in tobjs if d.u32(t + 0x40) != TOBJ_PASS]
            alpha = [(d.u32(t + 0x40) >> 20) & 15 for t in imported]
            mode = 'Blend' if alpha and all(a == 2 for a in alpha) and flags & (3 << 29) == 3 << 29 else (
                'Test' if alpha and all(a == 4 for a in alpha) and not flags & (3 << 29)
                and pe is not None and d.data[pe + 1] == 0x80 and d.data[pe] & 0x38 == 0x30
                else 'Opaque' if all(a == 0 for a in alpha) and not flags & (3 << 29) and pe is None
                else 'other')
            gates = [import_transparent(d, t) for t in imported]
            read = None if any(g is None for g in gates) else any(transparent for transparent, _ in gates)
            transparent = read if read is not None else expected.get('transparent')
            wanted = None if transparent is None else expected['mode'] if transparent else 'Opaque'
            agrees = expected.get('transparent') is None or read is None or expected['transparent'] == read
            found.append({'place': place, 'mode': mode, 'wanted': wanted, 'transparent': read,
                          'transparent_texels': [share for _, share in (g for g in gates if g is not None)],
                          'plan_agrees': agrees, 'render_flags': flags, 'alpha_ops': alpha,
                          'passthrough_slots': len(tobjs) - len(imported),
                          'pass': mode == wanted and agrees})
        rows[key] = dict(expected, read_back=found, **{'pass': bool(found) and all(r['pass'] for r in found)})
    return {'default': 'Test', 'materials': rows, 'pass': all(r['pass'] for r in rows.values())}


# GX image formats as HSDRaw stores them: format -> (block width, block height, bytes a block). The CI formats (8, 9,
# 10) index a palette of 16-bit colours, the TObj's TLUT. (HSDRaw GX/GXTexFmt; Tools/GXImageConverter.cs)
GX_BLOCKS = {0: (8, 8, 32), 1: (8, 4, 32), 2: (8, 4, 32), 3: (4, 4, 32), 4: (4, 4, 32),
             5: (4, 4, 32), 6: (4, 4, 64), 8: (8, 8, 32), 9: (8, 4, 32), 10: (4, 4, 32), 14: (8, 8, 32)}
GX_RGB565, GX_CI4, GX_CI8, GX_CMP = 4, 8, 9, 14
TLUT_IA8, TLUT_RGB565 = 0, 1


def _rgb5a3_alpha(v):
    """DecodeRGBA3's alpha of 16-bit colours: opaque with the top bit, else three bits scaled to 255."""
    return np.where(v & 0x8000, 255, ((v >> 12) & 7) * 255 // 7)


def texel_alpha(d, tobj):
    """The alpha of each texel of ``tobj``'s image as HSDRaw decodes it (Tools/GXImageConverter.cs DecodeTPL, what the
    import's IsTransparent reads): (h, w) ints; None when the TObj has no image data or its format is not one of GX's.
    A palette index past the palette's colours decodes clear (fromCI8), as does every texel of an I4 image under 8 on
    a side (fromI4)."""
    image = d.ptr(tobj + 0x4C)
    data = d.ptr(image) if image is not None else None
    if data is None:
        return None
    w, h, fmt = d.u16(image + 4), d.u16(image + 6), d.u32(image + 8)
    if fmt not in GX_BLOCKS or not (w and h):
        return None
    bw, bh, size = GX_BLOCKS[fmt]
    across, down = -(-w // bw), -(-h // bh)
    raw = np.frombuffer(bytes(d.data[data:data + across * down * size]), np.uint8)
    if len(raw) < across * down * size:
        return np.zeros((h, w), int)
    raw = raw.reshape(across * down, size).astype(int)
    words = raw[:, 0::2] << 8 | raw[:, 1::2]
    nibbles = np.stack([raw >> 4, raw & 15], -1).reshape(len(raw), -1)
    if fmt == 0:                                                    # I4: the intensity is the alpha
        texels = nibbles * 255 // 15 if min(w, h) >= 8 else np.zeros_like(nibbles)
    elif fmt == 1:                                                  # I8
        texels = raw
    elif fmt == 2:                                                  # IA4
        texels = (raw >> 4) * 255 // 15
    elif fmt == 3:                                                  # IA8: alpha, intensity
        texels = raw[:, 0::2]
    elif fmt == GX_RGB565:
        texels = np.full((len(raw), bw * bh), 255)
    elif fmt == 5:                                                  # RGB5A3
        texels = _rgb5a3_alpha(words)
    elif fmt == 6:                                                  # RGBA8: AR pairs, then GB pairs
        texels = raw[:, 0:32:2]
    elif fmt == GX_CMP:
        texels = _cmp_texel_alpha(raw)
    else:                                                           # CI4, CI8, CI14X2
        index = {GX_CI4: nibbles, GX_CI8: raw}.get(fmt, words & 0x3FFF)
        palette = _palette_alpha(d, d.ptr(tobj + 0x50))
        texels = np.where(index < len(palette), palette[np.minimum(index, max(len(palette) - 1, 0))]
                          if len(palette) else 0, 0)
    tiles = texels.reshape(down, across, bh, bw).swapaxes(1, 2).reshape(down * bh, across * bw)
    return tiles[:h, :w]


def _cmp_texel_alpha(raw):
    """CMPR (fromCMP): each 8x8 tile four 4x4 blocks (top left, top right, bottom left, bottom right) of 8 bytes, two
    RGB565 colours and sixteen 2-bit indices; a block whose first colour is not above its second has a clear colour,
    index 3. -> (tiles, 64) alpha, each tile's texels row by row."""
    blocks = raw.reshape(len(raw), 4, 8)
    c0 = blocks[..., 0] << 8 | blocks[..., 1]
    c1 = blocks[..., 2] << 8 | blocks[..., 3]
    index = (blocks[..., 4:8, None] >> np.array([6, 4, 2, 0])) & 3          # (tiles, block, row, column)
    clear = (c0 <= c1)[..., None, None] & (index == 3)
    alpha = np.where(clear, 0, 255).reshape(len(raw), 2, 2, 4, 4)            # (tile, block row, block column, y, x)
    return alpha.transpose(0, 1, 3, 2, 4).reshape(len(raw), 64)


def _palette_alpha(d, tlut):
    """The alpha of each colour of a TLUT (+0x00 colours, +0x04 format, +0x0C u16 count; PaletteToRGBA): IA8's high
    byte, RGB565 opaque, RGB5A3's own. Empty without one."""
    if tlut is None or d.ptr(tlut) is None:
        return np.zeros(0, int)
    at, fmt, count = d.ptr(tlut), d.u32(tlut + 0x04), d.u16(tlut + 0x0C)
    raw = np.frombuffer(bytes(d.data[at:at + 2 * count]), np.uint8).astype(int)
    words = raw[0::2] << 8 | raw[1::2]
    if fmt == TLUT_IA8:
        return words >> 8
    if fmt == TLUT_RGB565:
        return np.full(len(words), 255)
    return _rgb5a3_alpha(words)


def import_transparent(d, tobj):
    """The import's gate on a TObj in the file (HSDRawViewer Extensions/TOBJExtensions.cs IsTransparent): never for
    RGB565, nor for CI4 or CI8 with an RGB565 palette; else whether a texel decodes with alpha below 255. ->
    (transparent, the share of such texels), or None when the TObj has no image to read."""
    image = d.ptr(tobj + 0x4C)
    tlut = d.ptr(tobj + 0x50)
    fmt = d.u32(image + 8) if image is not None else None
    if fmt == GX_RGB565 or (fmt in (GX_CI4, GX_CI8) and tlut is not None and d.u32(tlut + 0x04) == TLUT_RGB565):
        return False, 0.
    alpha = texel_alpha(d, tobj)
    if alpha is None:
        return None
    share = float((alpha < 255).mean())
    return share > 0, share


def image_buffers(path, symbol):
    """Each written DObj's image buffers, read from the DAT rather than its exported PNGs.

    HSDRaw Common/HSD_TObj.cs: image +0 is data, +4/+6 dimensions, +8 GX format; TObj +0x50 is the TLUT.
    Buffers end at the next referenced structure (or data-block end), just as HSDRawFile partitions the file.
    """
    from bisect import bisect_right
    d = path if isinstance(path, Dat) else Dat.read(path)
    boundaries = sorted({d.u32(at) for at in d.relocations} | {off for _, off in d.roots} | {len(d.data)})

    def length(at):
        if at is None or not 0 <= at < len(d.data):
            return 0
        return boundaries[bisect_right(boundaries, at)] - at

    blocks = GX_BLOCKS          # CI formats require a separate 16-bit palette
    rows = []
    for joint in joints(d, symbol):
        for dobj in joint_dobjs(d, joint):
            mobj = d.ptr(dobj + 0x08)
            textures = []
            for tobj in d.chain(d.ptr(mobj + 0x08), 0x04) if mobj is not None else []:
                image = d.ptr(tobj + 0x4C)
                row = {'image_bytes': 0, 'required_bytes': 0, 'size': [0, 0], 'pass': False}
                if image is not None and length(image) >= 0x18:
                    w, h, fmt = d.u16(image + 4), d.u16(image + 6), d.u32(image + 8)
                    bw, bh, size = blocks.get(fmt, (1, 1, 0))
                    required = ((w + bw - 1) // bw) * ((h + bh - 1) // bh) * size
                    actual = length(d.ptr(image))
                    row.update(size=[w, h], format=fmt, image_bytes=actual, required_bytes=required)
                    row['pass'] = bool(w and h and required and actual >= required)
                    if fmt in (8, 9, 10):
                        tlut = d.ptr(tobj + 0x50)
                        colors = d.u16(tlut + 0x0C) if tlut is not None and length(tlut) >= 0x0E else 0
                        palette = length(d.ptr(tlut)) if colors else 0
                        row.update(palette_bytes=palette, palette_colors=colors)
                        row['pass'] &= bool(colors and palette >= 2 * colors)
                textures.append(row)
            rows.append({'drawn': d.ptr(dobj + 0x0C) is not None, 'textures': textures})
    return rows


def check_images(dobjs, expected):
    """Check a DObj list against {place: source textured materials}; blank and untextured places need no image."""
    rows = {}
    for place, materials in sorted(expected.items()):
        dobj = dobjs[place] if 0 <= place < len(dobjs) else None
        textures = dobj['textures'] if dobj is not None else []
        rows[place] = {'source_materials': sorted(materials), 'textures': textures,
                       'pass': bool(dobj and dobj['drawn'] and any(t['pass'] for t in textures))}
    return {'dobjs': rows, 'checked': len(rows), 'pass': all(row['pass'] for row in rows.values())}


def check_palettes(path, symbol):
    """Check the actual serialized envelope-pointer arrays, including low-detail and hidden PObjs."""
    d = Dat.read(path)
    sizes, seen = [], set()
    for j in joints(d, symbol):
        for dobj in joint_dobjs(d, j):
            for p in d.chain(d.ptr(dobj + 0x0C), 0x04):
                if p in seen:
                    continue
                seen.add(p)
                # PObjDesc layout: src/sysdolphin/baselib/pobj.h:40; type mask: baselib/forward.h:124.
                if d.u16(p + 0x0C) & 0x3000 != 0x2000:
                    continue
                table, count = d.ptr(p + 0x14), 0
                if table is None:
                    raise Failed(f'PObj {p:#x}: envelope palette has 0 entries; expected 1 to {MAX_ENVELOPES}')
                # loadEnvelopeDesc reads a null-terminated pointer array (baselib/pobj.c:216).
                while table + 4 * count + 4 <= len(d.data) and d.ptr(table + 4 * count) is not None:
                    count += 1
                    if count > MAX_ENVELOPES:
                        raise Failed(f'PObj {p:#x}: palette has at least {count} envelopes; limit {MAX_ENVELOPES}')
                if not count or table + 4 * count + 4 > len(d.data):
                    raise Failed(f'PObj {p:#x}: invalid palette with {count} envelopes; limit {MAX_ENVELOPES}')
                sizes.append(count)
    return {'pobjs': len(seen), 'palettes': len(sizes), 'maximum': max(sizes, default=0),
            'limit': MAX_ENVELOPES, 'pass': True}


def check_parts(vanilla_path, written_path, symbol, record):
    """Read the written file again; compare all copied bytes and relocated pointers."""
    source, written = Dat.read(vanilla_path), Dat.read(written_path)
    originals = [o for j in joints(source, symbol) for o in joint_dobjs(source, j)]
    objects = [o for j in joints(written, symbol) for o in joint_dobjs(written, j)]
    mapped = record['pointers']
    bad = []
    for old, new, size in record['blocks']:
        expected = bytearray(source.data[old:old + size])
        for field in source.relocations:
            if old <= field < old + size:
                struct.pack_into('>I', expected, field - old, mapped[source.u32(field)])
                if new + field - old not in written.relocations:
                    bad.append(new + field - old)
        if written.data[new:new + size] != expected:
            bad.append(new)
    for place in record['places']:
        for field in (8, 0x0c):
            original = source.ptr(originals[place] + field)
            if written.ptr(objects[place] + field) != (mapped[original] if original is not None else None):
                bad.append(objects[place] + field)
    return {'places': record['places'], 'bytes': record['bytes'], 'blocks': len(record['blocks']),
            'differing_offsets': sorted(set(bad)), 'pass': not bad}


def without_kept(model, places):
    """The new model's triangles only; kept vanilla parts have their own byte check."""
    if not places:
        return model
    keep = set(places)
    faces = [i for i, name in enumerate(model.face_objects)
             if not (match := PLACE.fullmatch(name)) or int(match[1]) not in keep]
    vertices = sorted({v for i in faces for v in model.faces[i]})
    indices = {old: new for new, old in enumerate(vertices)}
    materials = {model.face_materials[i] for i in faces}
    images = {m: model.material_images.get(m, []) for m in materials}
    textures = {path: value for path, value in model.textures.items() if any(path in paths for paths in images.values())}
    return replace(model, points=np.asarray(model.points)[vertices], weights=[model.weights[i] for i in vertices],
                   faces=[[indices[v] for v in model.faces[i]] for i in faces], triangles=len(faces),
                   face_objects=[model.face_objects[i] for i in faces], materials=len(materials),
                   face_materials=[model.face_materials[i] for i in faces], material_images=images, textures=textures)


def check_costume_size(path, f):
    """Check the written file against prep's share; kept bytes are already inside the file, never added twice."""
    size = Path(path).stat().st_size
    used, budget = _in_units(size), costume_budget(f.preloaded)
    record = {'file_bytes': size, 'allocated_bytes': used, 'budget_bytes': budget,
              'preloaded_bytes': sum(_in_units(n) + ARCHIVE_HEADER_BYTES for n in f.preloaded if n),
              'kept_bytes_in_file': f.kept, 'pass': used <= budget}
    if not record['pass']:
        raise Failed(f'{Path(path).name}: {size:,} file bytes ({used:,} in allocation units, including kept bytes); '
                     f'budget {budget:,} after {record["preloaded_bytes"]:,} preloaded bytes: {used - budget:,} over')
    return record
