import numpy as np
from PIL import Image

from rigswap.core import paint


def test_a_mask_shares_the_colour_image_s_name_less_its_last_suffix():
    assert paint.stem('M_TeamColor_Alb.png') == paint.stem('m_teamcolor_tcl.png') == 'm_teamcolor'
    assert paint.stem('m_eyelid_alb.0.png') == 'm_eyelid'
    assert paint.stem('skin.png') is None


def test_the_candidates_are_files_beside_it_of_its_name_and_shape_that_the_build_does_not_load():
    sizes = {'f/m_teamcolor_alb.png': (128, 64), 'f/m_teamcolor_tcl.png': (512, 256),
             'f/m_teamcolor_nrm.png': (256, 128), 'f/m_teamcolor_2cl.png': (128, 128),
             'f/m_stain_tcl.png': (512, 256)}
    loaded = {paint._key('f/m_teamcolor_nrm.png')}
    assert paint.siblings('f/m_teamcolor_alb.png', sizes, loaded) == ['f/m_teamcolor_tcl.png']
    hashes = {'k/1A33FB1E425ECBDC.png': (256, 256), 'k/198B6069776F548C.png': (256, 256)}
    assert paint.siblings('k/1A33FB1E425ECBDC.png', hashes, set()) == []


def test_a_paint_needs_more_than_half_the_answers_on_its_mask_and_takes_the_colour_nearest_the_rest():
    rows = {'A': [1, 2], 'B': [3], 'C': [4, 5]}
    answers = [
        {'rows': [{'row': 'A', 'mask': 2, 'colour': '#e6399b'}, {'row': 'B', 'mask': 3, 'colour': '#e6399b'},
                  {'row': 'C', 'mask': 4, 'colour': '#e6399b'}]},
        {'rows': [{'row': 'A', 'mask': 2, 'colour': '#6be04a'}, {'row': 'B', 'mask': 0, 'colour': '#6be04a'},
                  {'row': 'C', 'mask': 5, 'colour': '#6be04a'}]},
        {'rows': [{'row': 'A', 'mask': 0, 'colour': '#f04b23'}, {'row': 'C', 'mask': 0, 'colour': '#f04b23'}]}]
    # A: two of three name 2; pink, green and orange, and orange is nearest the other two (no blend of them).
    # B: one of three; C: one each
    assert paint.tally(answers, rows) == {'A': (2, (240, 75, 35))}


def test_a_mask_number_from_another_row_or_no_colour_paints_nothing():
    rows = {'A': [1]}
    wrong = [{'rows': [{'row': 'A', 'mask': 3, 'colour': '#ffffff'}]}] * 3
    broken = [{'rows': [{'row': 'A', 'mask': 1, 'colour': 'orange'}]}] * 3
    assert paint.tally(wrong, rows) == {} and paint.tally(broken, rows) == {}


def test_a_mask_s_light_and_dark_read_the_colour_image_under_each():
    # Inkling's eye: the colour image is light on the white and black on the iris; the mask is light on the white and
    # dark on the iris. Her shorts are the same in mirror
    px = np.zeros((1, 4, 4), np.uint8)
    px[0, :2, :3] = 220
    eye = np.array([[255, 255, 40, 40]], np.uint8)
    light, dark = paint.under(px, eye)
    assert dark < light and dark < paint.NEAR_BLACK
    assert paint.under(px, 255 - eye) == (dark, light)


def test_a_mask_paints_its_light_where_the_image_is_blank_there_and_asks_where_only_its_dark_is():
    px = np.zeros((1, 4, 4), np.uint8)
    px[0, :2, :3] = 220
    eye = np.array([[255, 255, 40, 40]], np.uint8)
    assert paint.side(px, 255 - eye) == 'light'        # her tank's cap, blank under a graded mask's light
    assert paint.side(px, eye) == 'ask'                # her eye, or her shorts: the files contradict themselves
    drawn = np.full((1, 4, 4), 200, np.uint8)
    assert paint.side(drawn, eye) is None              # Psyduck's shading layer over his drawn body


def test_a_mask_paints_its_dark_only_where_every_answer_picks_it_painted():
    def answer(**ways):
        return {'rows': [{'row': k, 'way': w} for k, w in ways.items()]}
    answers = [answer(A=2, B=2, C=1), answer(A=2, B=2, C=1), answer(A=2, B=1, C=1)]
    assert paint.picked_dark(answers, ['A', 'B', 'C', 'D']) == {'A': True, 'B': False, 'C': False, 'D': False}
    assert paint.picked_dark(answers[:2] + [None], ['A']) == {'A': False}
    assert paint.picked_dark([], ['A']) == {'A': False}


def test_the_way_ask_gives_each_row_its_colour_and_leads_to_neither_picture():
    rows = [{'letter': 'D', 'file': 'x/m_btm_000_alb.png', 'image': 'Bitmaptexture.006',
             'mask_file': 'x/m_btm_000_tcl.png', 'rgb': (241, 90, 36), 'numbers': [1, 2]}]
    q = paint.way_question(rows, 'Inkling girl', True)
    assert '(D: in #f15a24)' in q and 'tcl' not in q and 'blank' not in q
    assert 'as its files draw them (1), and painted in that colour (2)' in q and 'close, from the' in q
    item = paint.way_schema(rows)['properties']['rows']['items']
    assert item['properties']['way']['enum'] == [1, 2] and item['required'] == ['row', 'way']


def test_a_mask_over_a_drawn_colour_image_has_nothing_to_paint():
    # Psyduck's shading layer: about even grey over his yellow body, which the colour image already draws
    px = np.full((2, 2, 4), (240, 200, 100, 255), np.uint8)
    light, dark = paint.under(px, np.full((2, 2), 112, np.uint8))
    assert min(light, dark) >= paint.NEAR_BLACK


def test_the_paint_lays_the_colour_in_where_the_mask_says_and_keeps_alpha():
    px = np.zeros((1, 3, 4), np.uint8)
    px[..., 3] = [255, 128, 0]
    mask = np.array([[255, 128, 0]], np.uint8)
    light = paint.compose(px, mask, (200, 100, 0))
    assert light[0, :, :3].tolist() == [[200, 100, 0], [100, 50, 0], [0, 0, 0]]
    assert light[0, :, 3].tolist() == [255, 128, 0]
    dark = paint.compose(px, mask, (200, 100, 0), dark=True)
    assert dark[0, :, :3].tolist() == [[0, 0, 0], [100, 50, 0], [200, 100, 0]]


def test_the_ask_lists_each_row_s_files_and_names_the_character():
    rows = [{'letter': 'A', 'file': 'x/m_eye_alb.png', 'candidates': ['x/m_eye_tcl.png'], 'numbers': [1]}]
    q = paint.question(rows, 'Inkling girl', True)
    assert 'Picture 1 shows a 3D character (Inkling girl)' in q and 'A: m_eye_alb.png, then 1 m_eye_tcl.png' in q
    item = paint.schema(rows)['properties']['rows']['items']
    assert item['properties']['mask']['enum'] == [0, 1] and item['required'] == ['row', 'mask', 'colour']


def test_a_layer_row_takes_the_colour_more_than_half_the_answers_give():
    def answer(colour):
        return {'rows': [{'row': 'B', 'mask': 0, 'colour': colour}]}
    rows = {'B': []}
    assert paint.tally([answer('#101010'), answer('#000000'), answer('#ffffff')], rows) == {'B': (0, (16, 16, 16))}
    assert paint.tally([answer('#000000'), answer("can't tell"), answer('')], rows) == {}


def test_a_layer_s_grey_is_its_alpha_and_shows_its_colour():
    px = np.zeros((1, 3, 4), np.uint8)
    px[..., :3] = np.array([255, 128, 0], np.uint8)[:, None]
    px[..., 3] = 255
    assert paint.grey(px)
    shown = paint.cover(px, (0, 0, 0))
    assert shown[0, :, 3].tolist() == [255, 128, 0] and shown[0, :, :3].max() == 0
    assert paint.cover(px)[0, :, :3].tolist() == px[0, :, :3].tolist()
    eye = px.copy()
    eye[0, 0, :3] = (176, 170, 162)
    assert not paint.grey(eye)
    clear = px.copy()
    clear[0, 2, 3] = 0
    assert not paint.grey(clear)


def test_a_layer_s_ask_names_its_image_and_what_draws_it():
    rows = [{'letter': 'A', 'file': 'x/m_eyelid_alb.0.png', 'image': 'Bitmaptexture.003', 'materials': ['M_Eyelid'],
             'candidates': [], 'numbers': []}]
    q = paint.layer_question(rows, 'Inkling girl', True)
    assert 'Picture 1 shows a 3D character (Inkling girl)' in q and 'A: m_eyelid_alb.0.png, drawn by M_Eyelid' in q
    item = paint.layer_schema(rows)['properties']['rows']['items']
    assert item['required'] == ['row', 'colour'] and item['properties']['row']['enum'] == ['A']


def test_a_way_sheet_holds_its_four_tiles_whole_each_but_its_number():
    colours = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0)]
    sheet = np.asarray(paint.way_sheet('B', [Image.new('RGB', (100, 100), c) for c in colours]))
    for c in colours:
        assert np.all(sheet == c, axis=-1).sum() == 100 * 100 - 45 * 53
