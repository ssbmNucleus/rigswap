"""A piece of the model left out or brought back by a complaint (rigswap.core.pieces). Diddy Kong on Fox kept his
barrel jetpack on one import (2026-11-10) and "take the jetpack off" was a complaint the record could not hold."""
from types import SimpleNamespace

import pytest

from rigswap.core import pieces


def _unit(key, *meshes):
    return SimpleNamespace(key=key, meshes=list(meshes))


MODEL = [_unit('body', 'body'), _unit('pack', 'pack', 'pack.001'), _unit('gun', 'gun'),
         _unit('face', 'face', 'face.001')]


def test_the_pieces_offered_are_those_drawn_to_leave_one_out_and_those_left_out_to_bring_one_back():
    drop = ['gun', 'face.001']                  # the gun whole, one layer of the face
    assert [u.key for u in pieces.candidates(MODEL, drop, 'out')] == ['body', 'pack', 'face']
    assert [u.key for u in pieces.candidates(MODEL, drop, 'back')] == ['gun']


def test_an_answers_tiles_are_pieces_and_their_meshes_go_into_what_is_left_out_or_out_of_it():
    offered = pieces.candidates(MODEL, ['gun'], 'out')
    picked = pieces.chosen(offered, {'pieces': [2, 2, 9], 'cant_tell': False})        # a tile once; a number of none
    assert [u.key for u in picked] == ['pack']
    assert pieces.chosen(offered, {'pieces': [2], 'cant_tell': True}) == [] == pieces.chosen(offered, None)
    assert pieces.changed(['gun'], picked, 'out') == ['gun', 'pack', 'pack.001']
    assert pieces.changed(['gun', 'pack'], picked, 'out') == ['gun', 'pack', 'pack.001']
    assert pieces.changed(['gun', 'pack', 'pack.001'], picked, 'back') == ['gun']


def test_a_request_is_settled_by_one_ask_of_the_pieces_it_may_mean(monkeypatch, tmp_path):
    """The build has the model open: the pieces the costume draws are offered to leave one out, those it leaves out
    to bring one back, with the person's own words and the pieces' names; what the answer names changes ``drop``."""
    drawn = []
    monkeypatch.setattr(pieces.variants, 'units', lambda rig: (MODEL, True))
    monkeypatch.setattr(pieces, 'draw', lambda us, offered, gone, up, front, path:
                        drawn.append(([u.key for u in offered], set(gone))) or [path.with_name('pieces-1.png')])
    asks = []

    def ask(answer):
        def asked(question, pictures, schema, default, route):
            asks.append((question, pictures, schema, route))
            return answer, {'source': 'test'}
        return asked
    told = {'drop': ['gun'], 'pieces': {'way': 'out', 'piece': 'the barrel jetpack on his back',
                                        'said': 'take the "jetpack" off'}}
    drop, report = pieces.settle(None, told, (0, 0, 1), (0, -1, 0), tmp_path, ask({'pieces': [2], 'cant_tell': False}))
    assert drop == ['gun', 'pack', 'pack.001'] and told['drop'] == ['gun']
    assert report['way'] == 'out' and report['pieces'] == ['pack'] and report['meshes'] == ['pack', 'pack.001']
    question, pictures, schema, route = asks[-1]
    assert route == 'core complaint piece' and pictures == [str(tmp_path / 'pieces-1.png')]
    assert "take the 'jetpack' off" in question and 'the barrel jetpack on his back' in question
    assert '1 body; 2 pack; 3 face' in question and 'left out of the costume' in question
    assert schema['properties']['pieces']['items']['enum'] == [1, 2, 3]
    assert drawn[-1] == (['body', 'pack', 'face'], {'gun'})
    told = {'drop': ['gun'], 'pieces': {'way': 'back', 'piece': 'his gun'}}
    drop, report = pieces.settle(None, told, (0, 0, 1), (0, -1, 0), tmp_path, ask({'pieces': [1], 'cant_tell': False}))
    assert drop == [] and report['pieces'] == ['gun'] and drawn[-1][0] == ['gun'] and 'brought back' in asks[-1][0]


def test_a_request_no_piece_answers_ends_in_words_for_the_person(monkeypatch, tmp_path):
    monkeypatch.setattr(pieces.variants, 'units', lambda rig: (MODEL, True))
    monkeypatch.setattr(pieces, 'draw', lambda *a: [])
    none = lambda *a, **k: ({'pieces': [], 'cant_tell': False}, {})        # noqa: E731
    with pytest.raises(pieces.NoPiece, match='no piece of the model was found to be "his cape"'):
        pieces.settle(None, {'drop': [], 'pieces': {'way': 'out', 'piece': 'his cape'}}, (0, 0, 1), (0, -1, 0),
                      tmp_path, none)
    with pytest.raises(pieces.NoPiece, match='there is none to bring back'):
        pieces.settle(None, {'drop': [], 'pieces': {'way': 'back', 'piece': 'his cape'}}, (0, 0, 1), (0, -1, 0),
                      tmp_path, none)
    with pytest.raises(pieces.NoPiece, match='names no piece'):
        pieces.settle(None, {'drop': [], 'pieces': {'way': 'out', 'piece': ' '}}, (0, 0, 1), (0, -1, 0), tmp_path,
                      none)
