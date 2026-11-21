import json
import struct

import numpy as np
import pytest

from rigswap.load import binds

PLACEHOLDER = np.diag([100.0, 100, 100, 1])        # what Sketchfab's FBX conversion writes: 0.01 at the origin


def translate(y):
    m = np.eye(4)
    m[1, 3] = y
    return m


def chain(path, middle=PLACEHOLDER, glb=True):
    """A chain root - middle - tip, a unit apart up y, bound straight; the file's pose turns the tip a quarter about z.
    Three vertices: one on the root, one shared by the root and the tip, one on the tip; the middle joint carries none.
    ``middle`` is the middle joint's inverse bind."""
    positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], '<f4')
    joints = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0]], '<u2')
    weights = np.array([[1, 0, 0, 0], [.5, .5, 0, 0], [1, 0, 0, 0]], '<f4')
    matrices = np.array([np.eye(4), middle, np.linalg.inv(translate(2))]).transpose(0, 2, 1).astype('<f4')
    blobs = [positions.tobytes(), joints.tobytes(), weights.tobytes(), np.array([0, 1, 2], '<u2').tobytes() + b'\0\0',
             matrices.tobytes()]
    views, at = [], 0
    for blob in blobs:
        views.append({'buffer': 0, 'byteOffset': at, 'byteLength': len(blob)})
        at += len(blob)
    gltf = {
        'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 3]}],
        'nodes': [{'name': 'root', 'children': [1]}, {'name': 'middle', 'translation': [0, 1, 0], 'children': [2]},
                  {'name': 'tip', 'translation': [0, 1, 0], 'rotation': [0, 0, 2 ** -.5, 2 ** -.5]},
                  {'name': 'body', 'mesh': 0, 'skin': 0}],
        'skins': [{'joints': [0, 1, 2], 'inverseBindMatrices': 4, 'skeleton': 0}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'JOINTS_0': 1, 'WEIGHTS_0': 2}, 'indices': 3}]}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3',
                       'min': [0, 0, 0], 'max': [0, 2, 0]},
                      {'bufferView': 1, 'componentType': 5123, 'count': 3, 'type': 'VEC4'},
                      {'bufferView': 2, 'componentType': 5126, 'count': 3, 'type': 'VEC4'},
                      {'bufferView': 3, 'componentType': 5123, 'count': 3, 'type': 'SCALAR'},
                      {'bufferView': 4, 'componentType': 5126, 'count': 3, 'type': 'MAT4'}],
        'bufferViews': views, 'buffers': [{'byteLength': at}],
    }
    data = b''.join(blobs)
    if glb:
        text = json.dumps(gltf).encode()
        text += b' ' * (-len(text) % 4)
        body = struct.pack('<II', len(text), binds.JSON_CHUNK) + text + struct.pack('<II', len(data), binds.BIN_CHUNK)
        path.write_bytes(b'glTF' + struct.pack('<II', 2, 12 + len(body) + len(data)) + body + data)
    else:
        gltf['buffers'][0]['uri'] = 'chain.bin'
        path.with_name('chain.bin').write_bytes(data)
        path.write_text(json.dumps(gltf))
    return path


def inverse_binds(path):
    gltf, buffers, _ = binds._read(path)
    skin = gltf['skins'][0]
    matrices = binds._accessor(gltf, buffers, skin['inverseBindMatrices'])
    return matrices.astype(float).reshape(-1, 4, 4).transpose(0, 2, 1)


@pytest.mark.parametrize('glb', [True, False])
def test_a_joint_no_vertex_is_bound_to_takes_the_scale_of_the_joint_bound_below_it(tmp_path, glb):
    src = chain(tmp_path / ('chain.glb' if glb else 'chain.gltf'), glb=glb)
    out = binds.bound(src)
    assert out == src.with_name('chain.bound' + src.suffix) and out.is_file()
    before, after = inverse_binds(src), inverse_binds(out)
    assert np.allclose(after[1], np.eye(4))           # the placeholder's own turn and place, at the tip's scale
    assert np.array_equal(after[[0, 2]], before[[0, 2]])
    if glb:
        assert binds._read(out)[2] == binds._read(src)[2]          # the mesh's bytes are the file's own


def test_a_file_whose_unbound_joints_share_the_scale_below_them_is_imported_as_it_is(tmp_path):
    src = chain(tmp_path / 'chain.glb', middle=np.linalg.inv(translate(5)))       # a wrong place costs nothing
    assert binds.bound(src) == src
    assert sorted(p.name for p in tmp_path.iterdir()) == ['chain.glb']


@pytest.mark.blender
def test_a_vertex_shared_across_a_placeholder_stays_where_the_file_puts_it(tmp_path):
    """Read as a bind, the middle joint's 0.01 is dropped and the tip is bound 100 times too far up. The vertex the root
    and the tip share is carried half that way to the importer's bind and comes back by a blend of the root's move and
    the tip's turned one, which no longer agree: it landed 50 units off. Where the file puts it: half on the root at
    (0, 1, 0), half turned with the tip to (1, 2, 0)."""
    import bpy
    from rigswap.load import open as load_open
    bpy.ops.wm.read_homefile(use_empty=True)
    load_open._import_gltf(chain(tmp_path / 'chain.glb'))
    body = next(o for o in bpy.data.objects if o.type == 'MESH')
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = body.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    points = sorted(tuple(np.round(evaluated.matrix_world @ v.co, 4)) for v in mesh.vertices)
    evaluated.to_mesh_clear()
    assert np.allclose(points, [(0, 0, 0), (0, 0, 2), (0.5, 0, 1.5)], atol=1e-4)     # glTF's y up is Blender's z


def test_a_model_scaled_as_a_whole_above_its_bound_joints_is_imported_as_it_is(tmp_path):
    """Melee's own rips: an unbound Trans joint at 0.025, every joint bound below it at 0.025 too."""
    s = np.diag([40.0, 40, 40, 1])
    src = chain(tmp_path / 'chain.glb', middle=s @ np.linalg.inv(translate(1)))
    gltf, buffers, _ = binds._read(src)
    old = inverse_binds(src)
    old[[0, 2]] = s @ old[[0, 2]]
    assert np.allclose(binds._binds(gltf['skins'][0], gltf['nodes'], old, {0, 2}), old)
