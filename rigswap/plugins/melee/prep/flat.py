"""A flat fighter's build made what the game draws of it (facts/flat.py), first thing in prep, so that its limits see
what will be drawn. The placed model is moved as one, so its pieces stay as they join, on the weights it has:

- The broad side shows. The game shows YZ, so a model that covers more from the front than from the side is turned a
  quarter about up, through his root: what faced his front faces -X, what he is seen from facing right.
- A body has depth. The game shows any turn of a section about up, so a section keeps its shape when it goes as far
  along X as along Y: model X is stretched by the one number that makes the model's own do so. YZ, all the game shows
  of a joint that doesn't turn, stays as it was.
- Black under any fill. Every material but the eyes' is one black (colour_textures makes it the black texel).
- The eyes show from either side. Each gets a mirror image through his root, for when he faces left.
- The outline is a hull: the whole surface pushed out along its normals by the line his own has, on the same weights,
  in the same black, as one object (export/flat.py puts it in the hull's places). The normals are taken across the
  seams of the model's pieces, where a vertex is doubled, so the hull doesn't open there.

flat.json beside build.blend records it: its ``objects`` are details of the build (prep: detail_objects).
"""
import json
from pathlib import Path

import numpy as np

from ....core import onepiece, segments
from ....core.eyes import _join
from ..facts.flat import LINE

HULL = 'hull'
BLACK = 'flat'
QUARTER = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])     # on row vectors: -Y to -X, about Z


def stood(points, tris):
    """``points`` of the surface ``tris`` with the broad side showing and a body's depth, as the module's docstring
    says. -> (those points, its outline from the side, from the front, what X was stretched by)"""
    side, front = (onepiece.outline(points, tris, along) for along in (0, 1))
    if front > side:
        points = points @ QUARTER
    depth = float(np.ptp(points[:, 1]) / np.ptp(points[:, 0]))
    return points * (depth, 1, 1), side, front, depth


def pushed(points, tris, line):
    """``points`` of the surface ``tris`` out along its normals by ``line``, a vertex doubled at a seam moving as one."""
    at = np.unique(points, axis=0, return_inverse=True)[1].ravel()
    corners = points[tris]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    out = np.zeros((at.max() + 1, 3))
    for k in range(3):
        np.add.at(out, at[tris[:, k]], faces)
    out = out[at]
    return points + line * out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def ready(meshes, build_dir):
    """Make the open scene's ``meshes`` what the game draws, as the module's docstring says, and write flat.json in
    ``build_dir``. A scene that has the hull already, from a prep before, stays as it is. -> what flat.json holds."""
    import bpy
    build_dir = Path(build_dir)
    record = build_dir / 'flat.json'
    if HULL in bpy.data.objects:
        return json.loads(record.read_text(encoding='utf-8'))
    seen = build_dir / 'eyes.json'
    eyes = {o['name'] for o in json.loads(seen.read_text(encoding='utf-8')).get('objects', [])} if seen.is_file() \
        else set()
    points, tris, counts = segments._surface(meshes)
    points, side, front, depth = stood(points, tris)
    for o, part in zip(meshes, np.split(points, np.cumsum(counts)[:-1])):
        _place(o, part)
    black = bpy.data.materials.new(BLACK)
    black.use_nodes = True
    next(n for n in black.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = \
        (0., 0., 0., 1.)
    far = [_far(o) for o in meshes if o.name in eyes]
    hull = _join(bpy, [_copy(o) for o in [*meshes, *far]], HULL)
    _place(hull, pushed(*segments._surface([hull])[:2], LINE))
    for o in [hull, *(o for o in meshes if o.name not in eyes)]:
        o.data.materials.clear()
        o.data.materials.append(black)
        o.data.polygons.foreach_set('material_index', np.zeros(len(o.data.polygons), np.int32))
    made = {'side': side, 'front': front, 'turned': bool(front > side), 'depth': depth, 'line': LINE, 'hull': hull.name,
            'eyes': sorted(eyes) + [o.name for o in far], 'objects': [{'name': o.name} for o in [hull, *far]]}
    record.write_text(json.dumps(made, indent=1), encoding='utf-8')
    return made


def _copy(o):
    twin = o.copy()
    twin.data = o.data.copy()
    o.users_collection[0].objects.link(twin)
    return twin


def _place(o, points):
    """Object ``o``'s vertices at ``points``. The normals stored with them no longer hold: Blender works them out."""
    me = o.data
    me.vertices.foreach_set('co', points.ravel())
    if 'custom_normal' in me.attributes:
        me.attributes.remove(me.attributes['custom_normal'])
    me.update()


def _far(o):
    """Object ``o`` mirrored through his root across X, its faces turned to look the other way."""
    far = _copy(o)
    far.name = far.data.name = f'{o.name}_far'
    points = segments._surface([far])[0]
    points[:, 0] *= -1
    _place(far, points)
    far.data.flip_normals()
    return far
