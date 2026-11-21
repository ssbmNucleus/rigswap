"""load.binds: a glTF joint that no vertex is bound to, bound at a scale Blender can't hold.

Blender's glTF importer reads each bone's bind from its parent's and its own inverse bind matrices, keeping the turn and
the place between them: the scale between them is lost. It then carries each vertex from the file's bind to its own by
the blend of its joints' moves, and posing brings it back only where those moves agree. Above every bound joint the
lost scale is the same for all of them: Melee's own rips scale the whole model at their ``Trans`` joint (0.025), and
the joints bound below it share that scale. Between bound joints it carries them apart, and the vertices they share
tear.

A file's skinning reads a joint's inverse bind matrix only for the vertices bound to that joint, so where none is, an
exporter writes what it likes. Sketchfab's FBX conversion writes the FBX's unit scale at the origin: Tung Tung
Sahur's ``spine.004_05``, a spine bone between his chest and his head that carries no skin, is bound at 0.01 with his
head bound at 1 below it. His head was bound 100 times too far up, and the 388 vertices his chest and head bones share
were smeared into a band up to a quarter of his height; his arms came apart from it.

So a joint no vertex is bound to takes the scale of the nearest joint bound below it, the one scale that loses nothing
between them; its turn and place stay the file's, which cost nothing once nothing is lost. The file is imported from a
copy beside it that says so, or as it is where every such joint already has that scale, to the file's precision. What
the file shows is the same either way: its skinning never reads those matrices."""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from ..core.body import EXACT

COMPONENTS = {5120: '<i1', 5121: '<u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
GLB, JSON_CHUNK, BIN_CHUNK = b'glTF', 0x4E4F534A, 0x004E4942


def _read(path):
    """-> (the file's JSON, its buffers' bytes or None where one can't be read, its chunks after the JSON if a .glb)."""
    path = Path(path)
    data = path.read_bytes()
    chunks = []
    if data[:4] == GLB:
        at = 12
        while at + 8 <= len(data):
            length, kind = struct.unpack_from('<II', data, at)
            chunks.append((kind, data[at + 8:at + 8 + length]))
            at += 8 + length
        gltf = json.loads(chunks.pop(0)[1])
    else:
        gltf = json.loads(data)
    buffers = []
    for k, buffer in enumerate(gltf.get('buffers', [])):
        uri = buffer.get('uri')
        if uri is None:
            buffers.append(chunks[0][1] if k == 0 and chunks and chunks[0][0] == BIN_CHUNK else None)
        elif uri.startswith('data:'):
            buffers.append(base64.b64decode(uri.split(',', 1)[1]) if ';base64,' in uri else None)
        else:
            file = path.parent / unquote(uri)
            buffers.append(file.read_bytes() if file.is_file() else None)
    return gltf, buffers, chunks


def _accessor(gltf, buffers, index):
    """An accessor's values, one row per element, or None where the bytes aren't the values (sparse, or compressed)."""
    a = gltf['accessors'][index]
    if 'sparse' in a or 'bufferView' not in a or a['type'] not in WIDTHS:
        return None
    view = gltf['bufferViews'][a['bufferView']]
    data = buffers[view['buffer']]
    if data is None or view.get('extensions'):
        return None
    dtype, width = np.dtype(COMPONENTS[a['componentType']]), WIDTHS[a['type']]
    stride = view.get('byteStride') or dtype.itemsize * width
    return np.ndarray((a['count'], width), dtype, data, view.get('byteOffset', 0) + a.get('byteOffset', 0),
                      (stride, dtype.itemsize))


def _bound_nodes(gltf, buffers):
    """The nodes some vertex is bound to, through any skin, or None if a skin's weights can't be read."""
    bound = set()
    for node in gltf.get('nodes', []):
        if 'skin' not in node or 'mesh' not in node:
            continue
        joints = gltf['skins'][node['skin']]['joints']
        for primitive in gltf['meshes'][node['mesh']].get('primitives', []):
            attributes, k = primitive.get('attributes', {}), 0
            while f'JOINTS_{k}' in attributes:
                which = _accessor(gltf, buffers, attributes[f'JOINTS_{k}'])
                weight = _accessor(gltf, buffers, attributes[f'WEIGHTS_{k}']) if f'WEIGHTS_{k}' in attributes else None
                if which is None or weight is None:
                    return None
                bound.update(joints[j] for j in np.unique(which[weight > 0]) if j < len(joints))
                k += 1
    return bound


def _binds(skin, nodes, matrices, bound):
    """The skin's inverse bind matrices, each joint no vertex is bound to given the scale of the nearest joint bound
    below it. One already at that scale, to the file's precision (``EXACT`` of each axis), stays as it is, and so does
    one with none bound below it: it carries nothing the importer reads."""
    index = {node: k for k, node in enumerate(skin['joints'])}
    out = matrices.copy()
    for k, node in enumerate(skin['joints']):
        if node in bound:
            continue
        below, frontier = None, list(nodes[node].get('children', []))
        while frontier and below is None:
            below = next((c for c in frontier if c in index and c in bound), None)
            frontier = [g for c in frontier for g in nodes[c].get('children', [])]
        if below is None:
            continue
        try:
            bind, under = np.linalg.inv(matrices[k]), np.linalg.inv(matrices[index[below]])
        except np.linalg.LinAlgError:
            continue
        size, wanted = np.linalg.norm(bind[:3, :3], axis=0), np.linalg.norm(under[:3, :3], axis=0)
        if np.any(size == 0) or np.all(np.abs(wanted - size) <= EXACT * size):
            continue
        bind[:3, :3] *= wanted / size
        out[k] = np.linalg.inv(bind)
    return out


def bound(path):
    """The file to import for ``path``: itself, or a copy beside it (``<name>.bound.glb`` or ``.gltf``) whose skins bind
    every joint no vertex is bound to at the scale of the joints bound below it (``_binds``). A file whose weights
    can't be read goes as it is."""
    path = Path(path)
    gltf, buffers, chunks = _read(path)
    if not gltf.get('skins'):
        return path
    bound_nodes = _bound_nodes(gltf, buffers)
    if bound_nodes is None:
        return path
    changed = False
    for skin in gltf['skins']:
        if 'inverseBindMatrices' not in skin:
            continue
        old = _accessor(gltf, buffers, skin['inverseBindMatrices'])
        if old is None:
            continue
        old = old.astype(float).reshape(-1, 4, 4).transpose(0, 2, 1)
        new = _binds(skin, gltf['nodes'], old, bound_nodes)
        if np.array_equal(new, old):
            continue
        data = new.transpose(0, 2, 1).astype('<f4').tobytes()
        gltf['buffers'].append({'byteLength': len(data),
                                'uri': 'data:application/octet-stream;base64,' + base64.b64encode(data).decode()})
        gltf.setdefault('bufferViews', []).append({'buffer': len(gltf['buffers']) - 1, 'byteLength': len(data)})
        gltf['accessors'].append({'bufferView': len(gltf['bufferViews']) - 1, 'componentType': 5126,
                                  'count': len(skin['joints']), 'type': 'MAT4'})
        skin['inverseBindMatrices'] = len(gltf['accessors']) - 1
        changed = True
    if not changed:
        return path
    out = path.with_name(path.stem + '.bound' + path.suffix)
    text = json.dumps(gltf).encode()
    if chunks or path.suffix.lower() == '.glb':
        text += b' ' * (-len(text) % 4)
        body = struct.pack('<II', len(text), JSON_CHUNK) + text
        body += b''.join(struct.pack('<II', len(c), kind) + c for kind, c in chunks)
        out.write_bytes(GLB + struct.pack('<II', 2, 12 + len(body)) + body)
    else:
        out.write_bytes(text)
    return out
