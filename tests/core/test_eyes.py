"""The eye step's reading of an iris the model's file does not name (rigswap.core.eyes): Diddy Kong on Fox had no
eyes, then eyes all white (2026-11-10)."""
import numpy as np
from PIL import Image

from rigswap.core import eyes

WHOLE = np.array([[[0., 0], [1, 0], [1, 1]], [[0., 0], [1, 1], [0, 1]]])       # two triangles over the whole image


def _iris(colour=(90, 50, 20), size=16):
    """A picture clear but for a blob in its middle: an iris."""
    px = np.zeros((size, size, 4), np.uint8)
    px[size // 4:3 * size // 4, size // 4:3 * size // 4] = (*colour, 255)
    return px


def test_the_files_the_model_loads_none_of_are_offered_once(tmp_path):
    """A download keeps a file for each costume: of those with the same pixels the first is offered, and none the
    model already loads, nor one of another size."""
    white = np.full((16, 16, 4), 255, np.uint8)
    for folder, name, px in (('c00', 'white.png', white), ('c00', 'iris.png', _iris()), ('c01', 'iris.png', _iris()),
                             ('c00', 'gold.png', _iris((230, 190, 20))), ('c00', 'small.png', _iris(size=8))):
        (tmp_path / folder).mkdir(exist_ok=True)
        Image.fromarray(px, 'RGBA').save(tmp_path / folder / name)
    (tmp_path / 'c00' / 'notes.txt').write_text('no image')
    found = eyes.unloaded(tmp_path, (16, 16), [white])
    assert [p.replace('\\', '/').rsplit('/', 2)[-2:] for p, _ in found] == [['c00', 'gold.png'], ['c00', 'iris.png']]
    assert np.array_equal(found[1][1], _iris())
    assert eyes.unloaded(tmp_path / 'missing', (16, 16), []) == []


def test_a_picture_laid_over_another_shows_in_part_where_it_is_drawn():
    """An iris over its white is see-through round the iris; a white, a skin or a wholly clear image is no layer,
    and neither is an iris where the faces draw only its clear part."""
    assert eyes.layer_over(_iris(), WHOLE, 'REPEAT')
    assert not eyes.layer_over(np.full((16, 16, 4), 255, np.uint8), WHOLE, 'REPEAT')
    assert not eyes.layer_over(np.zeros((16, 16, 4), np.uint8), WHOLE, 'REPEAT')
    corner = np.array([[[0., 0], [.2, 0], [0, .2]]])                           # the image's clear bottom left corner
    assert not eyes.layer_over(_iris(), corner, 'REPEAT')


def test_the_iris_no_name_finds_is_the_one_the_ask_names(tmp_path):
    """Each file is drawn over the white through the eye's two maps on a numbered tile, and the tile the answer names
    is the iris; no tile, or an answer that cannot tell, is none, and with nothing to offer nothing is asked."""
    base = np.full((16, 16, 4), 255, np.uint8)
    files = [('a/gold.png', _iris((230, 190, 20))), ('a/iris.png', _iris())]
    layers = [(WHOLE, WHOLE)]
    asks = []

    def ask(answer):
        def asked(question, pictures, schema, default, route):
            asks.append((question, pictures, schema, route))
            return answer, {'source': 'test'}
        return asked
    got, answer, info = eyes.ask_iris(base, layers, files, 'REPEAT', tmp_path, ask({'iris': 2, 'cant_tell': False}))
    assert got[0] == 'a/iris.png' and answer == {'iris': 2, 'cant_tell': False} and info == {'source': 'test'}
    question, pictures, schema, route = asks[-1]
    assert route == 'core eyes iris' and pictures == [str(tmp_path / 'eyes-iris.png')]
    assert schema['properties']['iris']['enum'] == [0, 1, 2]
    sheet = np.asarray(Image.open(pictures[0]).convert('RGB'))
    tiles = [sheet[:, :eyes.TILE], sheet[:, eyes.TILE + 6:2 * eyes.TILE + 6]]
    middle = [t[36 + (eyes.TILE - 40) // 2, eyes.TILE // 2] for t in tiles]     # each file's blob over the white
    assert tuple(middle[0]) == (230, 190, 20) and tuple(middle[1]) == (90, 50, 20)
    assert eyes.ask_iris(base, layers, files, 'REPEAT', tmp_path, ask({'iris': 0, 'cant_tell': False}))[0] is None
    assert eyes.ask_iris(base, layers, files, 'REPEAT', tmp_path, ask({'iris': 1, 'cant_tell': True}))[0] is None
    assert eyes.ask_iris(base, layers, files, 'REPEAT', tmp_path, ask(None))[0] is None
    before = len(asks)
    assert eyes.ask_iris(base, layers, [], 'REPEAT', tmp_path, ask({'iris': 1, 'cant_tell': False})) == (None,) * 3
    assert len(asks) == before
