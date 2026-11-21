import pytest

from rigswap.library import moves_as
from rigswap.library.moves_as import RULES
from rigswap.vocabulary import Description, Piece


@pytest.mark.parametrize('piece, rule, layer, rendering, thin, held', [
    (Piece(1, 'body', ['hips']), 'body', False, False, False, False),
    (Piece(1, 'rigid', ['head']), 'rigid', False, False, False, False),
    (Piece(1, 'hangs', ['hips']), 'hangs', False, False, False, False),
    (Piece(1, 'layer', ['left_thigh']), 'layer', True, False, True, False),               # a skirt over the legs
    (Piece(1, 'body', ['hips'], belongs='layer_of', layer_of=2), 'layer', True, True, False, False),   # an edge line
    (Piece(1, 'layer', ['hips'], belongs='layer_of', layer_of=2), 'layer', True, True, False, False),
    (Piece(1, 'held', ['right_hand'], belongs='held'), 'held', False, False, False, True),
    (Piece(1, 'held', ['right_hand']), 'held', False, False, False, True),                # moves as held
    (Piece(1, 'body', ['right_hand'], belongs='held'), 'held', False, False, False, True),   # belongs held
    (Piece(1, 'rigid', ['right_hand'], belongs='held'), 'held', False, False, False, True),
])
def test_layer_and_held_mean_one_thing_everywhere(piece, rule, layer, rendering, thin, held):
    assert moves_as.rule_of(piece) == rule
    assert (piece.is_layer(), piece.is_rendering_layer(), moves_as.is_thin_layer(piece), piece.is_held()) == \
        (layer, rendering, thin, held)


def test_each_rule_says_how_it_constrains_the_solve():
    assert set(RULES) == {'body', 'rigid', 'layer', 'held', 'hangs'}
    assert {r: x.at_seam for r, x in RULES.items()} == {'body': 'meets', 'rigid': 'holds', 'layer': 'meets',
                                                       'held': 'holds', 'hangs': 'follows'}
    assert set(moves_as.SEAM_ORDER) == {x.at_seam for x in RULES.values()}
    assert {r: x.at_boundary for r, x in RULES.items()} == {'body': 'meets', 'rigid': 'holds', 'layer': None,
                                                            'held': None, 'hangs': None}
    assert [r for r, x in RULES.items() if x.reads_piece_under] == ['layer']
