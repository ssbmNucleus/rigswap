"""export/checks.py: every read-back check."""
import json
import struct

import numpy as np
import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee import dae as melee_dae, dat as melee_dat
from rigswap.plugins.melee.export import checks as melee_checks, dobjs as melee_dobjs
from tests.plugins.melee.test_dae import _arm, _dae, _dae_with_materials, _joint, _matrix, _objects_model, _triangle_model
from tests.plugins.melee.test_dat import _DatBuilder, _costume


# Kevin on Zelda (runs/pilot3/kevin-Zelda): the body, the hair and two eyes sharing one material. The eyes step moves
# the eyes to Zelda's animated places 26 and 25, each with a private copy of the eye material.
KEVIN = [('mc_kevin_0', 'Material-material'), ('mc_kevin_5', 'Material_005-material'),
         ('eyes_left', 'eyes-material'), ('eyes_right', 'eyes-material')]


KEVIN_EYES = [{'object': 'eyes_left', 'side': 'left', 'from_place': 27, 'to_place': 26, 'vanilla_side': 'left'},
              {'object': 'eyes_right', 'side': 'right', 'from_place': 28, 'to_place': 25, 'vanilla_side': 'right'}]


# What HSDRawViewer read back, by place: it names a material by its bytes, so how many materials the private eye copies
# are is not up to the export. Kevin on Zelda: the same bytes at 25 and 26, one material. Cammy on Mario: two, the copy
# at 14 the eye's old material, the one at 15 another only because unreferenced bytes follow its TObj in the file.
EYES_READ_BACK = {'alike': ['Material_2E3C2AED', 'Material_2E3C2AED'],
                  'different': ['Material_CD52313F', 'Material_36AEDF3F']}


def _kevin_read_back(eyes='alike', hair='Material_021F4A8E'):
    return _objects_model([('Joint_0_Object_29', 'Material_334B1CC4'), ('Joint_0_Object_44', hair),
                           ('Joint_0_Object_26', EYES_READ_BACK[eyes][0]),
                           ('Joint_0_Object_25', EYES_READ_BACK[eyes][1])])


def test_alpha_readback_rejects_modulate_and_missing_depth_writes(tmp_path):
    b = _DatBuilder()
    joint = b.add(bytes(0x40))
    dobj = b.add(bytes(0x10))
    mobj = b.add(bytes(0x18))
    tobj = b.add(bytes(0x5c))
    pe = b.add(bytes([0x33, 0x80]) + bytes(10))
    for at, value in ((joint + 0x10, dobj), (dobj + 8, mobj), (mobj + 8, tobj), (mobj + 0x14, pe)):
        b.point(at, value)
    model = _triangle_model()
    model.face_materials = ['skin']
    model.face_objects = ['Joint_0_Object_0']
    planned = {'skin': {'mode': 'Test', 'transparent': True}}
    def check():
        dat = b.save(tmp_path / 'alpha.dat', [('root', joint)])
        return melee_checks.alpha_mode_check(dat, 'root', model, model, planned)['pass']
    struct.pack_into('>I', b.data, tobj + 0x40, 4 << 20)
    assert check()
    b.data[pe] = 0x13  # compare without depth update
    assert not check()
    b.data[pe] = 0x33
    struct.pack_into('>I', b.data, tobj + 0x40, 1 << 20)
    assert not check()
    struct.pack_into('>I', b.data, tobj + 0x40, 2 << 20)
    struct.pack_into('>I', b.data, mobj + 4, 3 << 29)
    planned['skin']['mode'] = 'Blend'
    assert check()


def test_alpha_readback_leaves_out_the_slots_the_eye_step_keeps_as_passthroughs(tmp_path):
    # cammy-Zelda, goku-budokai-Zelda, kritter-Zelda (runs/pilot4): Zelda's eye DObjs have two TObjs, so the eye step
    # gives the eye material a second, passthrough slot (alpha op PASS, 5); the read-back called the eye's Test material
    # "Opaque or other" because that slot's alpha op is not REPLACE.
    b = _DatBuilder()
    joint = b.add(bytes(0x40))
    dobj = b.add(bytes(0x10))
    mobj = b.add(bytes(0x18))
    eye, slot = b.add(bytes(0x5c)), b.add(bytes(0x5c))
    pe = b.add(bytes([0x33, 0x80]) + bytes(10))
    for at, value in ((joint + 0x10, dobj), (dobj + 8, mobj), (mobj + 8, eye), (eye + 4, slot), (mobj + 0x14, pe)):
        b.point(at, value)
    struct.pack_into('>I', b.data, mobj + 4, 0x34)
    struct.pack_into('>I', b.data, eye + 0x40, (4 << 20) | (1 << 7))
    struct.pack_into('>I', b.data, slot + 0x40, melee_dobjs.TOBJ_PASS)
    model = _triangle_model()
    model.face_materials = ['eyes-material']
    model.face_objects = ['Joint_0_Object_0']
    planned = {'eyes-material': {'mode': 'Test', 'transparent': True}}

    def check():
        dat = b.save(tmp_path / 'eyes.dat', [('root', joint)])
        return melee_checks.alpha_mode_check(dat, 'root', model, model, planned)['materials']['eyes-material']
    row = check()
    assert row['pass'] and row['read_back'][0]['alpha_ops'] == [4] and row['read_back'][0]['passthrough_slots'] == 1
    # a second texture that does change the colour is the material's own, and its alpha op counts
    struct.pack_into('>I', b.data, slot + 0x40, (4 << 16) | (5 << 20))
    row = check()
    assert not row['pass'] and row['read_back'][0]['alpha_ops'] == [4, 5]


def _one_texture(fmt, w, h, pixels, op, flags=0x14, pe=None, palette=None):
    """A costume of one DObj whose material has one TObj: an image of GX ``fmt``, ``w`` x ``h``, ``pixels`` bytes,
    alpha op ``op``, render ``flags``, a PE descriptor's first two bytes ``pe``, a ``palette`` (format, colour bytes).
    -> (builder, root joint)"""
    b = _DatBuilder()
    joint, dobj, mobj, tobj = (b.add(bytes(n)) for n in (0x40, 0x10, 0x18, 0x5c))
    data = b.add(pixels, 32)
    image = b.add(struct.pack('>I2H2I2f', 0, w, h, fmt, 0, 0., 0.))
    for at, value in ((joint + 0x10, dobj), (dobj + 8, mobj), (mobj + 8, tobj), (tobj + 0x4C, image), (image, data)):
        b.point(at, value)
    struct.pack_into('>I', b.data, mobj + 4, flags)
    struct.pack_into('>I', b.data, tobj + 0x40, (op << 20) | (1 << 7 if op == 4 else 1 << 4))
    if pe is not None:
        b.point(mobj + 0x14, b.add(bytes(pe) + bytes(10)))
    if palette is not None:
        colours = b.add(palette[1], 32)
        tlut = b.add(struct.pack('>3IH2x', 0, palette[0], 0, len(palette[1]) // 2))
        b.point(tlut, colours)
        b.point(tobj + 0x50, tlut)
    return b, joint


def _rgb5a3(alpha):
    """One 4x4 RGB5A3 block of grey texels with ``alpha`` (16 values) as DecodeRGBA3 keeps them."""
    return b''.join(struct.pack('>H', 0xFFFF if a == 255 else (a // 32) << 12 | 0x888) for a in alpha)


def _checked(tmp_path, b, joint, planned):
    model = _triangle_model()
    model.face_materials, model.face_objects = ['skin'], ['Joint_0_Object_0']
    dat = b.save(tmp_path / 'gate.dat', [('root', joint)])
    return melee_checks.alpha_mode_check(dat, 'root', model, model, {'skin': planned})['materials']['skin']


TEST_PE = (0x33, 0x80)


def test_the_alpha_check_holds_an_opaque_texture_to_no_alpha_operation(tmp_path):
    # Cammy on Mario (runs/pilot4): Tex_0012's PNG has alpha 242 to 249 in places, which RGB5A3 keeps as opaque, so
    # the import gave the material no alpha operation; the plan had called it transparent and the check failed it.
    near = _rgb5a3([242] * 4 + [255] * 12)
    row = _checked(tmp_path, *_one_texture(5, 4, 4, near, 0), {'mode': 'Test', 'transparent': False})
    assert row['pass'] and row['read_back'][0]['mode'] == 'Opaque' and row['read_back'][0]['transparent'] is False
    # the plan reading the gate wrong fails, whatever the file did
    row = _checked(tmp_path, *_one_texture(5, 4, 4, near, 0), {'mode': 'Test', 'transparent': True})
    assert not row['pass'] and not row['read_back'][0]['plan_agrees']
    # an opaque texture still gets no alpha operation: a cutout's Test on it fails
    row = _checked(tmp_path, *_one_texture(5, 4, 4, near, 4, pe=TEST_PE), {'mode': 'Test', 'transparent': False})
    assert not row['pass'] and row['read_back'][0]['mode'] == 'Test'
    # nor a pixel engine compare
    row = _checked(tmp_path, *_one_texture(5, 4, 4, near, 0, pe=TEST_PE), {'mode': 'Test', 'transparent': False})
    assert not row['pass'] and row['read_back'][0]['mode'] == 'other'


def test_the_alpha_check_holds_a_transparent_texture_to_its_mode(tmp_path):
    clear = _rgb5a3([0] * 4 + [255] * 12)
    row = _checked(tmp_path, *_one_texture(5, 4, 4, clear, 4, pe=TEST_PE), {'mode': 'Test', 'transparent': True})
    assert row['pass'] and row['read_back'][0]['transparent_texels'] == [.25]
    row = _checked(tmp_path, *_one_texture(5, 4, 4, clear, 0), {'mode': 'Test', 'transparent': True})
    assert not row['pass'] and row['read_back'][0]['wanted'] == 'Test'
    # A CI8 texture with an RGB5A3 palette: the import's quantizer picked the colours, so the plan cannot tell
    # (transparent None) and the file decides. Index 1 is clear.
    palette = (2, struct.pack('>2H', 0xFFFF, 0x0888))
    ci8 = bytes([0] * 31 + [1])
    row = _checked(tmp_path, *_one_texture(9, 8, 4, ci8, 4, pe=TEST_PE, palette=palette),
                   {'mode': 'Test', 'transparent': None})
    assert row['pass'] and row['read_back'][0]['transparent'] is True
    row = _checked(tmp_path, *_one_texture(9, 8, 4, bytes(32), 0, palette=palette),
                   {'mode': 'Test', 'transparent': None})
    assert row['pass'] and row['read_back'][0]['mode'] == 'Opaque'
    # an RGB565 palette is never transparent, even where an index is past its colours (which would decode clear)
    row = _checked(tmp_path, *_one_texture(9, 8, 4, bytes([2] * 32), 0, palette=(1, bytes(4))),
                   {'mode': 'Test', 'transparent': False})
    assert row['pass']


def test_texel_alpha_decodes_as_hsdraw_does(tmp_path):
    def decoded(fmt, w, h, pixels, palette=None):
        b, joint = _one_texture(fmt, w, h, pixels, 0, palette=palette)
        d = melee_dat.Dat.read(b.save(tmp_path / 'texels.dat', [('root', joint)]))
        tobj = d.ptr(d.ptr(d.ptr(joint + 0x10) + 8) + 8)
        return melee_checks.texel_alpha(d, tobj)
    # CMPR: an 8x8 tile of four blocks; the top right one has a clear colour (first colour not above the second) and
    # its texel (1, 2) takes it (index 3); the top left has index 3 everywhere but four colours, so stays opaque
    opaque = struct.pack('>2H', 0xFFFF, 0) + b'\xff' * 4
    clear = struct.pack('>2H', 0, 0xFFFF) + bytes([0, 0b00001100, 0, 0])
    alpha = decoded(14, 8, 8, opaque + clear + opaque + opaque)
    assert {tuple(p) for p in np.argwhere(alpha < 255)} == {(1, 6)} and alpha[1, 6] == 0
    # RGB5A3, two blocks across, cropped to the image's 6 x 4
    alpha = decoded(5, 6, 4, _rgb5a3([255] * 15 + [0]) + _rgb5a3([64] + [255] * 15))
    assert alpha.shape == (4, 6) and alpha[3, 3] == 0 and alpha[0, 4] == 72 and (alpha < 255).sum() == 2
    # CI4 with an IA8 palette: the high byte of each colour is its alpha
    alpha = decoded(8, 8, 8, bytes([0x01] + [0] * 31), palette=(0, struct.pack('>2H', 0xFF00, 0x4000)))
    assert alpha[0, 1] == 0x40 and (alpha < 255).sum() == 1


def test_joint_comparison_on_two_small_armatures(tmp_path):
    before = melee_dae.read_dae(_arm(tmp_path, 'before'))
    assert list(before.joints) == ['JOBJ_0', 'JOBJ_1', 'JOBJ_2']
    assert np.allclose(before.positions()['JOBJ_2'], [0, 3, 0])
    assert before.parents == {'JOBJ_0': None, 'JOBJ_1': 'JOBJ_0', 'JOBJ_2': 'JOBJ_1'}

    same = melee_checks.compare_joints(before.joints, melee_dae.read_dae(_arm(tmp_path, 'same')).joints)
    assert not same.moved() and same.distance < 1e-9

    moved = melee_checks.compare_joints(before.joints, melee_dae.read_dae(_arm(tmp_path, 'moved', hand_x=1.5)).joints)
    assert moved.moved() and moved.worst == 'JOBJ_2' and moved.distance == pytest.approx(0.5)

    grown = melee_dae.read_dae(_arm(tmp_path, 'grown', extra=_joint('JOBJ_3', _matrix(x=1))))
    check = melee_checks.compare_joints(before.joints, grown.joints)
    assert check.extra == ['JOBJ_3'] and check.moved()
    assert melee_checks.compare_joints(grown.positions(), before.positions()).missing == ['JOBJ_3']


def test_joint_tree_order_matters(tmp_path):
    one = melee_dae.read_dae(_dae(tmp_path / 'one.dae', _joint('JOBJ_0', _matrix(), _joint('JOBJ_1', _matrix(x=1)),
                                                           _joint('JOBJ_2', _matrix(x=-1)))))
    other = melee_dae.read_dae(_dae(tmp_path / 'other.dae', _joint('JOBJ_0', _matrix(), _joint('JOBJ_2', _matrix(x=-1)),
                                                               _joint('JOBJ_1', _matrix(x=1)))))
    assert not melee_checks.compare_joints(one.joints, other.joints).moved()
    assert melee_checks.tree_differs(one.joints, one.parents, one.joints, one.parents) is None
    assert 'joint 1' in melee_checks.tree_differs(one.joints, one.parents, other.joints, other.parents)


@pytest.mark.parametrize('change', ['rotation', 'scale'])
def test_skeleton_check_rejects_changed_local_frame_at_the_same_origin(change):
    root = np.diag([2., 3., 4., 1.])
    root[:3, 3] = (1, 2, 3)
    local = np.eye(4)
    local[:3, 3] = (0, 1, 0)
    before = {'root': root, 'leaf': root @ local}
    if change == 'rotation':
        local[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    else:
        local[0, 0] = 1.1
    after = {'root': root, 'leaf': root @ local}
    parents = {'root': None, 'leaf': 'root'}
    check = melee_checks.compare_joints(before, after, parents, parents)
    assert check.moved() and check.distance == 0 and check.worst == 'leaf'
    assert check.matrix_error == pytest.approx(1 if change == 'rotation' else .1)


def test_low_detail_position_check_rejects_displacement_and_empty_geometry():
    model, low = _triangle_model(), _triangle_model()
    low.points += [0, 0, .1]
    check = melee_checks.position_check(melee_checks.compare_models(model, low))
    with pytest.raises(Failed, match='low_positions.*0.1'):
        melee_checks.require_checks({'low_positions': check}, 'test.dat')
    assert not melee_checks.position_check({})['pass']


def test_readback_checks_material_weight_and_winding_preservation():
    before, after = _triangle_model(), _triangle_model()
    checks = melee_checks.preservation_checks(before, after, melee_checks.compare_models(before, after))
    assert all(c['pass'] for c in checks.values())
    after.materials = 2
    after.weights[0] = {'JOBJ_0': .9, 'JOBJ_1': .1}
    after.faces = [[2, 1, 0]]
    checks = melee_checks.preservation_checks(before, after, melee_checks.compare_models(before, after))
    assert not any(c['pass'] for c in checks.values())
    assert checks['weights']['maximum_change'] == pytest.approx(.1)
    assert checks['winding']['fraction'] == 1
    with pytest.raises(Failed, match='materials.*weights.*winding'):
        melee_checks.require_checks(checks, 'test.dat')


@pytest.mark.parametrize('eyes', sorted(EYES_READ_BACK))
def test_the_material_check_leaves_out_the_eyes_the_eye_step_gave_materials_of_its_own(eyes):
    source, back = _objects_model(KEVIN), _kevin_read_back(eyes)
    # Counting the whole model, the eye copies decide the count: alike, 3 = 3; different, 3 against 4.
    assert melee_checks.material_check(source, back)['pass'] is (eyes == 'alike')
    check = melee_checks.material_check(source, back, KEVIN_EYES)
    assert check == {'before': 2, 'after': 2, 'pass': True,
                     'not_counted': {'eye_objects': ['eyes_left', 'eyes_right'], 'eye_places': [25, 26]},
                     'with_eyes': {'before': 3, 'after': 3 if eyes == 'alike' else 4}}
    # A material lost from the rest still fails: the hair reads back as the body's material.
    assert not melee_checks.material_check(source, _kevin_read_back(eyes, hair='Material_334B1CC4'), KEVIN_EYES)['pass']


def test_the_material_check_counts_materials_as_the_import_writes_them():
    # goku-jump-Mario (runs/pilot4): the body and the hair outline draw one image, texture_002.png, which the import
    # writes once and the read-back names one material (Material_86CC57E5 at places 16, 17 and 21).
    source = _objects_model([('body', 'body-material'), ('eyes', 'eyes-material'), ('face', 'Face-material'),
                             ('hair', 'Hair-material'), ('hair_outline', 'outline-material')])
    places = {'body': ('Joint_0_Object_16', 'Material_86CC57E5'), 'eyes': ('Joint_0_Object_14', 'Material_BE8BF4CA'),
              'face': ('Joint_0_Object_19', 'Material_4CB3C6BF'), 'hair': ('Joint_0_Object_20', 'Material_F1FEF4FB'),
              'hair_outline': ('Joint_0_Object_21', 'Material_86CC57E5')}
    back = _objects_model([places[o] for o in source.face_objects])
    eyes = [{'object': 'eyes', 'side': 'both', 'from_place': 18, 'to_place': 14, 'vanilla_side': 'right'}]
    alike = {'body-material': ('texture_002', 'CMP', 'Test'), 'outline-material': ('texture_002', 'CMP', 'Test'),
             'Face-material': ('texture_003', 'CMP', 'Test'), 'Hair-material': ('texture_001', 'CMP', 'Test'),
             'eyes-material': ('texture_000', 'CMP', 'Test')}
    assert not melee_checks.material_check(source, back, eyes)['pass']             # 4 against 3, as the run failed
    check = melee_checks.material_check(source, back, eyes, alike)
    assert (check['before'], check['after'], check['pass']) == (3, 3, True)
    assert check['written_alike'] == [['body-material', 'outline-material']]
    assert check['with_eyes']['before'] == check['with_eyes']['after'] == 4
    # The face read back as the body's material is still a loss: the import writes the two differently.
    places['face'] = ('Joint_0_Object_19', 'Material_86CC57E5')
    back = _objects_model([places[o] for o in source.face_objects])
    assert not melee_checks.material_check(source, back, eyes, alike)['pass']


def test_a_material_the_eyes_share_with_the_face_still_counts_where_the_face_is_drawn():
    source = _objects_model([('body', 'skin'), ('head', 'face'), ('eyes_left', 'face'), ('eyes_right', 'face')])
    eye_copy = [('Joint_0_Object_26', 'Material_EYECOPY'), ('Joint_0_Object_25', 'Material_EYECOPY')]
    back = _objects_model([('Joint_0_Object_29', 'Material_SKIN'), ('Joint_0_Object_30', 'Material_FACE')] + eye_copy)
    check = melee_checks.material_check(source, back, KEVIN_EYES)
    assert (check['before'], check['after'], check['pass']) == (2, 2, True)
    # the head's material lost (drawn with the body's): one where the source has two, eyes or no eyes
    back = _objects_model([('Joint_0_Object_29', 'Material_SKIN'), ('Joint_0_Object_30', 'Material_SKIN')] + eye_copy)
    assert not melee_checks.material_check(source, back, KEVIN_EYES)['pass']


def test_coincident_vertices_match_full_influences_and_detect_a_lost_set():
    before, after = _triangle_model(), _triangle_model()
    for m in (before, after):
        m.points = np.vstack([m.points, m.points[0]])
        m.weights.append({'JOBJ_1': .75, 'JOBJ_0': .25})
    after.weights[0], after.weights[3] = after.weights[3], after.weights[0]
    assert melee_checks.compare_models(before, after)['weight_change_max'] == 0
    # The two copies round in different directions; both are still within the export's position precision.
    after.points[0, 0] += .0004
    after.points[3, 0] += .0002
    assert melee_checks.compare_models(before, after)['weight_change_max'] == 0
    after.weights[0] = after.weights[3]
    assert melee_checks.compare_models(before, after)['weight_change_max'] == .75


def test_triangle_matching_keeps_two_sided_faces_and_rejects_missing_connectivity():
    before, after = _triangle_model(), _triangle_model()
    before.faces = [[0, 1, 2], [2, 1, 0]]
    after.faces = [[1, 0, 2], [1, 2, 0]]
    assert melee_checks.triangle_matches(before, after) == ([1, 0], 0, 0)
    after.faces = [[0, 1, 2], [0, 1, 2]]
    assert melee_checks.triangle_matches(before, after)[1] == 1
    after.points[2] = [1, 1, 0]
    assert melee_checks.triangle_matches(before, after)[2] == 2


def test_compression_carrying_a_thin_triangles_apex_over_its_base_is_no_reversal():
    before, after = _triangle_model(), _triangle_model()
    before.points = np.array([[0, 0, 0], [1, 0, 0], [.5, .002, 0.]])
    after.points = np.array([[0, 0, 0], [1, 0, 0], [.5, -.002, 0.]])
    before.faces = after.faces = [[0, 1, 2]]
    assert melee_checks.triangle_matches(before, after) == ([0], 0, 0)
    after.faces = [[0, 2, 1]]
    assert melee_checks.triangle_matches(before, after) == ([0], 1, 0)


def test_a_corner_rounded_on_every_axis_at_once_is_still_the_same_corner():
    # the import keeps each coordinate within 0.005: a corner 0.00497 off on all three axes is that corner (s4-mario's
    # face overlays came back 0.00497 off on two); one a hundredth off on an axis is not
    before, after = _triangle_model(), _triangle_model()
    after.points = after.points + 0.00497
    assert melee_checks.triangle_matches(before, after) == ([0], 0, 0)
    after.points[0] = [.01, 0, 0]
    assert melee_checks.triangle_matches(before, after) == ([None], 0, 1)


def test_image_count_uses_glb_as_reference(tmp_path):
    dae = _dae_with_materials(tmp_path / 'model.dae', ('skin',))
    doc = json.dumps({'images': [{}, {}], 'textures': [{'source': 0}, {'source': 1}],
                      'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': i}}}
                                    for i in range(2)],
                      'meshes': [{'primitives': [{'material': 0}, {'material': 1}]}]}).encode()
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    assert melee_checks.image_count_check(glb, dae) == {'before': 2, 'after': 1, 'pass': False}
    # Materials count only where faces use them: the DAE here draws with both, then with one.
    text = dae.read_text(encoding='utf-8')
    geo = ('<library_geometries><geometry id="g"><mesh>{}</mesh></geometry></library_geometries>')
    tri = '<triangles material="{}" count="1"><p>0 1 2</p></triangles>'
    dae.write_text(text.replace('</COLLADA>', geo.format(tri.format('skin-material') + tri.format('eye-material')) + '</COLLADA>'), encoding='utf-8')
    assert melee_checks.material_count_check(glb, dae) == {'before': 2, 'after': 2, 'pass': True}
    dae.write_text(text.replace('</COLLADA>', geo.format(tri.format('skin-material')) + '</COLLADA>'), encoding='utf-8')
    assert melee_checks.material_count_check(glb, dae) == {'before': 2, 'after': 1, 'pass': False}


def test_image_count_ignores_unused_and_non_colour_images(tmp_path):
    dae = _dae_with_materials(tmp_path / 'model.dae', ('skin',))
    text = dae.read_text(encoding='utf-8').replace('</library_images>',
        '<image id="normal"><init_from>normal.png</init_from></image></library_images>')
    text = text.replace('</lambert>', '<bump><texture texture="normal"/></bump></lambert>')
    dae.write_text(text, encoding='utf-8')
    doc = json.dumps({'images': [{}, {}, {}, {}], 'textures': [{'source': i} for i in range(4)],
                      'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                            'metallicRoughnessTexture': {'index': 1}},
                                     'normalTexture': {'index': 2}},
                                    {'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}]}).encode()
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    assert melee_checks.image_count_check(glb, dae) == {'before': 1, 'after': 1, 'pass': True}


@pytest.mark.parametrize('encoding,expected', [('CMP', True), ('RGB5A3', False), (None, False)])
def test_texture_formats_follow_drawn_materials_and_ignore_stale_images(tmp_path, encoding, expected):
    before, after = _triangle_model(), _triangle_model()
    before.face_materials, after.face_materials = ['skin'], ['read-skin']
    before.material_images = {'skin': ['skin.png']}
    asked = tmp_path / 'formats.json'
    asked.write_text(json.dumps({'skin': 'CMPR'}), encoding='utf-8')
    (tmp_path / 'Texture_99_CMP.png').write_bytes(b'stale')
    if encoding:
        image = tmp_path / f'Texture_0_{encoding}.png'
        image.write_bytes(b'pixels')
        after.material_images = {'read-skin': [str(image)]}
    result = melee_checks.texture_format_check(asked, before, after)
    assert result['pass'] is expected
    assert result['materials']['skin']['requested'] == 'CMPR'
    asked.write_text('{}', encoding='utf-8')
    assert not melee_checks.texture_format_check(asked, before, after)['pass']


def test_kept_readback_is_separated_by_place_including_rigid_names():
    m = _triangle_model()
    m.points = np.concatenate([m.points, m.points + 5])
    m.weights *= 2
    m.faces = [[0, 1, 2], [3, 4, 5]]
    m.face_objects = ['Joint_0_Object_1', 'Joint_25_Object_71_SINGLE']
    m.face_materials = ['skin', 'sword']
    m.material_images = {'skin': ['skin.png'], 'sword': ['sword.png']}
    m.textures = {'skin.png': 'a', 'sword.png': 'b'}
    body = melee_checks.without_kept(m, (71,))
    assert body.triangles == 1 and body.materials == 1 and len(body.points) == 3
    assert body.face_materials == ['skin'] and body.textures == {'skin.png': 'a'}


@pytest.mark.parametrize('damage', [None, 'material', 'tobj', 'image', 'data', 'short', 'size', 'format', 'palette'])
def test_written_dobj_images_require_real_buffers(tmp_path, damage):
    symbol = 'PlyTest_Share_joint'
    b = _DatBuilder()
    # A blank animated place, two drawn textured places, and an intentionally untextured place.
    root, _ = _costume(b, [(None, [((0,), False), ((0,), True), ((0,), True), ((), True)])], {})
    path = b.save(tmp_path / 'images.dat', [(symbol, root)])
    d = melee_dat.Dat.read(path)
    objects = melee_dat.joint_dobjs(d, root)
    for place in (1, 2):
        material = d.ptr(objects[place] + 8)
        tobj = d.ptr(material + 8)
        # All-zero pixels are valid; do not mistake a black image for an absent buffer.
        pixels = d.add(bytes(32), 32)
        image = d.add(struct.pack('>I2H2I2f', 0, 8, 8, 14, 0, 0., 0.))
        d.point(image, pixels)
        d.point(tobj + 0x4C, image)
        if place != 2:
            continue
        if damage == 'material':
            d.point(objects[place] + 8, None)
        elif damage == 'tobj':
            d.point(material + 8, None)
        elif damage == 'image':
            d.point(tobj + 0x4C, None)
        elif damage == 'data':
            d.point(image, None)
        elif damage == 'short':
            d.point(image, pixels + 4)
        elif damage == 'size':
            struct.pack_into('>H', d.data, image + 4, 0)
        elif damage == 'format':
            struct.pack_into('>I', d.data, image + 8, 99)
        elif damage == 'palette':
            # CI4 has the same 32-byte image size, but cannot draw without its palette.
            struct.pack_into('>I', d.data, image + 8, 8)
    d.save(path)
    check = melee_checks.check_images(melee_checks.image_buffers(path, symbol), {1: ['skin'], 2: ['eye']})
    assert check['checked'] == 2
    assert check['dobjs'][1]['pass']
    assert check['pass'] == (damage is None)
    assert check['dobjs'][2]['source_materials'] == ['eye']
    if damage:
        with pytest.raises(Failed, match='dobj_images'):
            melee_checks.require_checks({'dobj_images': check}, 'images.dat')


def test_dobj_image_expectations_follow_source_materials_not_readback_images(monkeypatch):
    # Two identical-looking read-back materials have lost their images. Only one was textured in the source.
    source = melee_dae.Model(face_materials=['skin', 'plain'], material_images={'skin': ['skin.png']})
    back = melee_dae.Model(face_materials=['renamed', 'renamed'],
                       face_objects=['Joint_0_Object_2', 'Joint_0_Object_3'])
    monkeypatch.setattr(melee_checks, 'triangle_matches', lambda *_: ([0, 1], 0, 0))
    monkeypatch.setattr(melee_checks, 'image_buffers', lambda *_: [{'drawn': True, 'textures': []}] * 4)
    check = melee_checks.dobj_image_check('unused.dat', 'root', source, back)
    assert not check['pass'] and set(check['dobjs']) == {2}
    assert check['dobjs'][2]['source_materials'] == ['skin']
    back.face_objects[0] = 'unrecognized'
    assert melee_checks.dobj_image_check('unused.dat', 'root', source, back)['unmapped_triangles'] == 1


def test_dobj_image_check_rejects_missing_or_blank_expected_places():
    valid = {'pass': True, 'image_bytes': 32, 'required_bytes': 32, 'size': [8, 8]}
    rows = [{'drawn': False, 'textures': [valid]}, {'drawn': True, 'textures': [valid]}]
    assert melee_checks.check_images(rows, {1: ['skin']})['pass']
    assert not melee_checks.check_images(rows, {0: ['skin']})['pass']
    assert not melee_checks.check_images(rows, {2: ['skin']})['pass']
    assert melee_checks.check_images(rows, {})['pass']


@pytest.mark.parametrize('count', [10, 11])
def test_export_checks_the_actual_pobj_palette(tmp_path, count):
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((), True)])], {})
    dobj = struct.unpack_from('>I', b.data, root + 0x10)[0]
    pobj = struct.unpack_from('>I', b.data, dobj + 0x0C)[0]
    struct.pack_into('>H', b.data, pobj + 0x0C, 0x2000)
    table = b.add(bytes(4 * (count + 1)))
    b.point(pobj + 0x14, table)
    for i in range(count):
        envelope = b.add(bytes(16))
        b.point(envelope, root)
        struct.pack_into('>f', b.data, envelope + 4, 1.)
        b.point(table + 4*i, envelope)
    path = b.save(tmp_path / 'palette.dat', [('test', root)])
    if count == 11:
        with pytest.raises(Failed, match='11 envelopes; limit 10'):
            melee_checks.check_palettes(path, 'test')
    else:
        assert melee_checks.check_palettes(path, 'test')['maximum'] == 10


def _coincident_layers(read_order):
    """goku-budokai-Mario: face and eyes triangles on the same three corners (a layer drawn over the face), the eyes'
    RGB5A3 and the face's CI8:RGB5A3 each read back as a material of its own. The read-back lists the triangles in
    ``read_order``, so the coincident pair is met in either order."""
    def tri(x):
        return [[x, 0., 0.], [x + 1, 0., 0.], [x, 1., 0.]]
    places = [0., 0., 10., 20., 30., 40.]
    sources = ['face-material', 'eyes-material', 'face-material', 'eyes-material', 'face-material', 'eyes-material']
    reads = {'face-material': 'read-face', 'eyes-material': 'read-eyes'}

    def model(order, materials):
        points = np.array([p for k in order for p in tri(places[k])])
        faces = [[3 * k, 3 * k + 1, 3 * k + 2] for k in range(len(order))]
        return melee_dae.Model(points=points, weights=[{'JOBJ_0': 1.} for _ in points], faces=faces,
                               triangles=len(order), materials=len(set(materials)), face_materials=materials)
    before = model(range(6), sources)
    return before, model(read_order, [reads[sources[k]] for k in read_order])


@pytest.mark.parametrize('read_order', [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4], [5, 1, 0, 4, 2, 3]])
def test_coincident_triangles_of_two_materials_are_matched_to_their_own_read_back_material(tmp_path, read_order):
    before, after = _coincident_layers(read_order)
    for n, (name, code) in enumerate((('face', 'CI8_RGB5A3'), ('eyes', 'RGB5A3'))):
        image = tmp_path / f'Texture_{n}_{code}.png'
        image.write_bytes(b'pixels')
        after.material_images[f'read-{name}'] = [str(image)]
    asked = tmp_path / 'formats.json'
    asked.write_text(json.dumps({'face-material': 'CI8:RGB5A3', 'eyes-material': 'RGB5A3'}), encoding='utf-8')
    result = melee_checks.texture_format_check(asked, before, after)
    assert result['pass'], result
    assert result['materials']['face-material']['read_materials'] == ['read-face']
    assert result['materials']['eyes-material']['read_materials'] == ['read-eyes']


def _stacked_layers(layers, read_order, spots=3):
    """``layers`` meshes on the same triangles at ``spots`` places (a rip's alternative expressions, each a material of
    its own on the same eyelid), source material ``layer-<n>``, read back as ``read-<n>`` in ``read_order``."""
    def model(order, name):
        corners = [[[x, 0., 0.], [x + 1, 0., 0.], [x, 1., 0.]] for x in range(0, 10 * spots, 10)]
        points = np.array([p for _ in order for tri in corners for p in tri], float)
        faces = [[3 * k, 3 * k + 1, 3 * k + 2] for k in range(len(order) * spots)]
        materials = [f'{name}-{n}' for n in order for _ in range(spots)]
        return melee_dae.Model(points=points, weights=[{'JOBJ_0': 1.} for _ in points], faces=faces,
                               triangles=len(faces), materials=len(set(order)), face_materials=materials,
                               face_uvs=[[[0., 0.], [1., 0.], [0., 1.]] for _ in faces])
    return model(range(layers), 'layer'), model(read_order, 'read')


def _picture(path, colour, noise=0):
    from PIL import Image
    pixels = np.zeros((32, 32, 4), np.uint8) + np.array(colour, np.uint8)
    pixels[8:24, 8:24, 3] = 0                                      # a hole, as an eyelid's texture has
    if noise:
        pixels[..., :3] = np.clip(pixels[..., :3].astype(int) + np.random.default_rng(0).integers(
            -noise, noise + 1, (32, 32, 3)), 0, 255)
    Image.fromarray(pixels, 'RGBA').save(path)
    return str(path)


@pytest.mark.parametrize('read_order', [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_stacked_layers_on_one_uv_layout_are_matched_by_their_images(tmp_path, read_order):
    """Layers on the same corners and UVs, with images of their own: each is matched to the read-back material whose
    image (as the import encoded it) is its own, so its requested format is found on it, in whatever order the
    read-back lists them."""
    before, after = _stacked_layers(3, read_order)
    colours = [(200, 150, 140, 255), (60, 40, 30, 255), (120, 160, 220, 255)]
    codes = ['CMP', 'RGB5A3', 'CI8_RGB5A3']
    for n, colour in enumerate(colours):
        before.material_images[f'layer-{n}'] = [_picture(tmp_path / f'texture_{n}.png', colour)]
        after.material_images[f'read-{n}'] = [_picture(tmp_path / f'Texture_{n}_{codes[n]}.png', colour, noise=6)]
    asked = tmp_path / 'formats.json'
    asked.write_text(json.dumps({f'layer-{n}': code.replace('CMP', 'CMPR').replace('_', ':')
                                 for n, code in enumerate(codes)}), encoding='utf-8')
    result = melee_checks.texture_format_check(asked, before, after)
    assert result['pass'], result
    for n in range(3):
        assert result['materials'][f'layer-{n}']['read_materials'] == [f'read-{n}']


@pytest.mark.parametrize('read_order', [[0, 1], [1, 0]])
def test_stacked_layers_with_uvs_of_their_own_are_matched_by_their_uvs(read_order):
    """Layers on the same corners with no image to tell them apart, each sampling its own part of the texture: a
    read-back triangle takes the source triangle with its UVs, its corners listed in any order."""
    before, after = _stacked_layers(2, read_order, spots=1)
    shift = {0: 0., 1: .5}
    before.face_uvs = [[[u + shift[n], v] for u, v in tri] for n, tri in enumerate(before.face_uvs)]
    after.face_uvs = [[[u + shift[n], v] for u, v in tri] for n, tri in zip(read_order, after.face_uvs)]
    after.faces = [face[1:] + face[:1] for face in after.faces]
    after.face_uvs = [tri[1:] + tri[:1] for tri in after.face_uvs]
    assert melee_checks.triangle_matches(before, after) == (list(read_order), 0, 0)


@pytest.mark.parametrize('planned', [3, 2])
def test_the_dobj_count_holds_the_import_to_what_the_join_planned(tmp_path, planned):
    """The join plans places for one DObj per mesh and material, the eyes' among them (prep/join.py); the import's
    DObjs must be that many, or the plan fitted something other than what was imported (golduck-zip on Bowser: 21
    planned, 22 made)."""
    b = _DatBuilder()
    symbol = 'PlyTest_Share_joint'
    root, _ = _costume(b, [(None, [((0,), True), ((0,), True)]), (0, [((), True)])], {})
    path = b.save(tmp_path / 'imported.dat', [(symbol, root)])
    check = melee_checks.dobj_count_check(planned, path, symbol)
    assert (check['planned'], check['imported'], check['pass']) == (planned, 3, planned == 3)


def test_the_nearest_point_is_the_one_measuring_every_point_finds():
    # near points (within a reach), far ones (past every reach) and ties: the same indices and distances, bit for bit,
    # as measuring each point against every point of b
    rng = np.random.default_rng(3)
    b = rng.uniform(-10, 10, (700, 3))
    b[350:] = b[:350]                           # each of these twice: a tie, the first by index wins
    a = np.concatenate([b[::3] + rng.normal(0, 1e-4, (234, 3)), rng.uniform(-30, 30, (100, 3)), b[:5]])
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    index, gap = melee_checks._nearest(a, b)
    assert (index == d.argmin(1)).all() and (gap == d.min(1)).all()
