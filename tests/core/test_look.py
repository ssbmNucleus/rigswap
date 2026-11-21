from types import SimpleNamespace

import numpy as np

from rigswap.core import look, variants


def _unit(key, lo, hi, **kw):
    u = SimpleNamespace(key=key, lo=np.array(lo, float), hi=np.array(hi, float), meshes=[key], switched=False,
                        bone=0, bones={0}, triangles=100, extents=np.array([1., 2., 3.]))
    for k, v in kw.items():
        setattr(u, k, v)
    return u


def test_the_answer_names_what_goes_and_what_it_does_not_name_stays():
    us = [_unit(k, [0, 0, 0], [1, 1, 1]) for k in ('face', 'blink', 'shell', 'fist')]
    assert [u.key for u in look.left_out(us, {'leave_out': [2, 4]})] == ['blink', 'fist']
    assert look.left_out(us, {'leave_out': [2, 4], 'cant_tell': True}) == [], "can't tell leaves nothing out"
    assert look.left_out(us, None) == [] and look.left_out(us, {}) == []
    assert look.left_out(us, {'leave_out': list(range(1, len(us) + 1))}) == [], 'every piece left out is no look'
    assert [u.key for u in look.left_out(us, {'leave_out': [9, True, 'x', 2, 2]})] == ['blink'], 'only tiles, once'


def test_the_answers_schema_is_tile_numbers():
    s = look.schema(3)
    assert s['required'] == ['leave_out'] and s['properties']['leave_out']['items']['enum'] == [1, 2, 3]
    assert 'cant_tell' in s['properties']


def test_worn_copies_in_one_place_are_a_layer_over_the_other():
    eye = _unit('Player00__M_Eye', [0, 0, 0], [1, 1, 1], triangles=98)
    band = _unit('Player00__M_Eyelid', [0, -0.03, 0], [1, 0.97, 1], triangles=98)
    ken = _unit('link_ken', [5, 0, 0], [6, 1, 3], triangles=600)
    kenb = _unit('link_kenb', [-1, -2, 0], [0, -1, 3], triangles=600)
    keys = ['Player00__M_Eye', 'Player00__M_Eyelid', 'link_ken', 'link_kenb']
    assert look.layers([eye, band, ken, kenb], keys, 10.0) == [(['Player00__M_Eye'], ['Player00__M_Eyelid'])]
    assert look.layers([eye, band], ['Player00__M_Eye'], 10.0) == []


def _scene(monkeypatch, us, answers):
    """keep() on the units ``us``, the pictures not drawn, ``answers`` given in turn. -> (report, asked routes)."""
    rig = SimpleNamespace(points=np.array([[0., 0, 0], [10, 10, 10]]), names=['root'])
    monkeypatch.setattr(variants, 'units', lambda rig: (us, False))
    monkeypatch.setattr(variants, 'drop', lambda rig, names: rig)
    monkeypatch.setattr(look.pieces, 'draw', lambda *a, **k: ['tiles-1.png'])
    monkeypatch.setattr(look, 'views', lambda rig, meshes, up, front, path: path)
    asked = []

    def ask(question, pictures, schema, default, route):
        asked.append((route, question, pictures))
        return answers[len(asked) - 1], {'source': 'ai'}
    _, report = look.keep(rig, None, None, 'out', ask, 'Mewtwo', character='Frieza')
    return report, asked


def test_nothing_overrides_the_answers(monkeypatch):
    # Frieza, 2026-11-18: the rules dropped his head shell and forearms that the answers kept. A blink the answer
    # keeps stays too: a piece kept by mistake is a blemish the person can name
    us = [_unit(k, [0, 0, 0], [1, 1, 1], triangles=n) for k, n in
          (('F_01_normal', 28), ('F_02_damage', 32), ('head', 96), ('Mario_Blink', 30), ('ude', 50))]
    report, asked = _scene(monkeypatch, us, [{'leave_out': [2], 'cant_tell': False}])
    assert report['dropped'] == ['F_02_damage']
    assert [r for r, _, _ in asked] == [look.ROUTE], 'one ask: its answer stands (lab 2026-11-19)'
    assert 'Frieza' in asked[0][1] and 'Mewtwo' in asked[0][1] and '3 head (root)' in asked[0][1]
    assert look.EMPTY.strip() in asked[0][1], 'a fighter holding nothing is told so'
    assert asked[0][2][1:] == ['tiles-1.png'], 'the whole model, then the tiles'


def test_an_answer_leaving_nothing_out_drops_nothing(monkeypatch):
    us = [_unit(k, [0, 0, 0], [1, 1, 1]) for k in ('body', 'eyes')]
    report, asked = _scene(monkeypatch, us, [{'leave_out': [], 'cant_tell': False}])
    assert report['dropped'] == [] and len(asked) == 1


def test_a_settled_file_is_not_asked_and_only_its_coarser_copies_go(monkeypatch):
    us = [_unit('body_high', [0, 0, 0], [1, 1, 2]), _unit('body_low', [0, 0, 0], [1, 1, 2]),
          _unit('Mario_Blink', [0, 0, 0], [1, 1, 1])]
    rig = SimpleNamespace(points=np.array([[0., 0, 0], [3, 3, 3]]), names=['root'])
    monkeypatch.setattr(variants, 'units', lambda rig: (us, False))
    monkeypatch.setattr(variants, 'drop', lambda rig, names: rig)

    def ask(*a, **k):
        raise AssertionError('a settled file is not asked')
    _, report = look.keep(rig, None, None, 'out', ask, 'Mario', settled=True)
    assert report['dropped'] == ['body_low']
