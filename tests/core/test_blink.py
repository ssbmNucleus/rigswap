import json

import numpy as np
from PIL import Image

from rigswap.core import blink


def test_a_footprint_reads_a_repeating_image_in_the_tile_of_each_triangles_lowest_corner():
    tri = np.array([[[0.125, 0.125], [0.5, 0.125], [0.125, 0.5]]])
    here = blink.footprint(tri, (16, 16), 'REPEAT')
    assert (blink.footprint(tri + 1., (16, 16), 'REPEAT') == here).all() and here.any()
    x, y, w, h = blink.box_of(here)
    assert x == 2 and y >= 8 and y + h <= 15            # the image's top row first: v up is y down


def test_a_clamped_image_has_one_tile_so_a_triangle_past_it_draws_none_of_it():
    tri = np.array([[[0.125, 0.125], [0.5, 0.125], [0.125, 0.5]]])
    assert blink.footprint(tri, (16, 16), 'EXTEND').any()
    assert not blink.footprint(tri + 1., (16, 16), 'EXTEND').any()
    assert not blink.footprint(tri - 1., (16, 16), 'EXTEND').any()


def test_a_clamped_faces_uvs_stay_past_the_edge_of_the_crop_and_a_repeating_ones_come_back():
    uv = np.array([[-.5, .25], [-.25, .25], [-.5, .5]])
    whole = [0, 0, 16, 16]
    assert np.allclose(blink.crop_uvs(uv, (16, 16), whole, 'EXTEND'), uv)
    assert np.allclose(blink.crop_uvs(uv, (16, 16), whole, 'REPEAT'), uv + [1, 0])


def test_parts_that_touch_are_one_and_a_gap_makes_two():
    mask = np.zeros((10, 10), bool)
    mask[1:4, 1:4] = mask[4, 4] = mask[6:9, 6:9] = True
    assert [int(m.sum()) for m in blink.islands(mask)] == [10, 9]


def test_crop_uvs_draw_the_same_texels_in_the_crop():
    uv = np.array([[0.25, 0.75], [0.5, 0.75], [0.25, 0.5]])
    crop = [4, 2, 8, 8]
    moved = blink.crop_uvs(uv, (16, 16), crop, 'REPEAT')
    back = np.stack([moved[:, 0] * crop[2] + crop[0], (1 - moved[:, 1]) * crop[3] + crop[1]], -1)
    assert np.allclose(back, np.stack([uv[:, 0] * 16, (1 - uv[:, 1]) * 16], -1))


def test_a_layer_drawn_through_another_map_lands_where_its_triangle_does():
    base = np.full((8, 8, 4), 255, np.uint8)
    top = np.zeros((8, 8, 4), np.uint8)
    top[..., 3] = 255                                       # opaque black, drawn over the white
    whole = np.array([[[0., 0.], [1., 0.], [1., 1.]], [[0., 0.], [1., 1.], [0., 1.]]])
    left = whole * [0.5, 1.]                                # the base's left half takes the top's whole picture
    out = blink.layered(base, top, left, whole, 'REPEAT')
    assert (out[:, :4, :3] == 0).all() and (out[:, 4:, :3] == 255).all()


def test_a_copy_and_a_mirror_image_are_found_alike():
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 4), 128, np.uint8)
    eye = rng.integers(0, 255, (10, 12, 4)).astype(np.uint8)
    eye[..., 3] = 255
    img[5:15, 5:17] = eye
    img[40:50, 30:42] = eye[:, ::-1]
    found = blink.find_alike(img, [5, 5, 12, 10], np.ones((10, 12), bool), [[5, 5, 12, 10]])
    assert found[0] == ([30, 40, 12, 10], True)


def test_a_drawn_state_is_drawn_again_at_the_size_shown(tmp_path):
    opened = np.zeros((40, 40, 4), np.uint8)
    opened[..., 2] = opened[..., 3] = 255
    skin = np.array([200, 150, 100, 255], np.uint8)
    pictures, made, drawn = blink.blink(opened, [0, 0, 40, 40], [{'box': [0, 0, 40, 40], 'up': 'top'}], {}, skin)
    (tmp_path / 'eyes').mkdir()
    for state, px in pictures.items():
        Image.fromarray(px, 'RGBA').save(tmp_path / 'eyes' / f'{state}.png')
    record = json.loads(json.dumps({'frames': {s: f'eyes/{s}.png' for s in blink.STATES}, 'drawn': drawn}))
    small = blink.pictures_at(tmp_path, record, (8, 8))
    assert (small['half'][0] == skin).all() and (small['half'][-1] == opened[0, 0]).all()
    assert (small['closed'][3, :, :3] < skin[:3]).all()       # the line where the lids meet, a texel thick
    assert set(drawn) == {'half', 'closed'} and 'lid' in made['closed']
