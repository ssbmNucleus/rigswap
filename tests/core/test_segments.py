import numpy as np
import pytest

from rigswap.core import segments

pytestmark = pytest.mark.blender        # mathutils' BVH and KD trees


def _box(lo, hi):
    """A closed box: its 8 corners and 12 triangles."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    points = np.array([[(lo, hi)[i >> k & 1][k] for k in range(3)] for i in range(8)])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return points, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def _figure(parts):
    points, tris = [], []
    for lo, hi in parts:
        p, t = _box(lo, hi)
        tris.append(t + sum(len(q) for q in points))
        points.append(p)
    points = np.vstack(points)
    tris = np.vstack(tris)
    return points, tris, segments._pieces(points, tris)


# a figure standing on z, mirrored across x = 0: each limb's end sunk a little into the piece it hangs from; the feet
# pressed together on the midline
PARTS = {
    'trunk': ([-1, -.5, 3], [1, .5, 6]),
    'head': ([-.8, -.6, 5.9], [.8, .6, 7.5]),
    'thigh+': ([.2, -.3, 1.5], [.9, .3, 3.1]), 'thigh-': ([-.9, -.3, 1.5], [-.2, .3, 3.1]),
    'shin+': ([.25, -.25, 0], [.85, .25, 1.6]), 'shin-': ([-.85, -.25, 0], [-.25, .25, 1.6]),
    'foot+': ([0, -1, -.5], [1, .5, .1]), 'foot-': ([-1, -.95, -.45], [0, .45, .1]),
    'upper+': ([.9, -.3, 5.2], [2.5, .3, 5.8]), 'upper-': ([-2.5, -.3, 5.2], [-.9, .3, 5.8]),
    'fore+': ([2.4, -.25, 5.25], [4, .25, 5.75]), 'fore-': ([-4, -.25, 5.25], [-2.4, .25, 5.75]),
}


def test_pieces_that_touch_are_parent_and_child_from_the_trunk_out():
    names = list(PARTS)
    points, tris, piece = _figure(PARTS.values())
    name = {piece[8 * i]: n for i, n in enumerate(names)}
    parents, heads, _, twins, axis, loops = segments.skeleton(points, tris, piece)
    assert loops == []           # no contact the tree leaves out closes a loop through the trunk
    assert np.array_equal(axis, [1, 0, 0])
    for p, label in name.items():
        expected = label[:-1] + ('-' if label.endswith('+') else '+') if label[-1] in '+-' else label
        assert name[twins[p]] == expected
    tree = {name[p]: name[q] if q >= 0 else None for p, q in enumerate(parents)}
    assert tree == {'trunk': None, 'head': 'trunk', 'thigh+': 'trunk', 'thigh-': 'trunk', 'shin+': 'thigh+',
                    'shin-': 'thigh-', 'foot+': 'shin+', 'foot-': 'shin-', 'upper+': 'trunk', 'upper-': 'trunk',
                    'fore+': 'upper+', 'fore-': 'upper-'}
    # the hip is the thigh's end sunk in the trunk, not the middle of what overlaps
    hip = heads[[p for p in name if name[p] == 'thigh+'][0]]
    assert np.allclose(hip, [.55, 0, 3.1], atol=.05)


def test_a_piece_no_contact_reaches_joins_where_its_surface_comes_nearest():
    """sonic32x: the cuff (its shoe pressed in) touches nothing else. The shin's end faces its top 0.25 away, their
    corners 0.38 apart; the hanging glove's corner is 0.29 from the cuff's. By the nearest pair of vertices the cuff
    hung from the glove, and the shoe rode the hand."""
    half = {
        'trunk': ([-1, -.5, 3], [1, .5, 6]), 'head': ([-.8, -.6, 5.9], [.8, .6, 7.5]),
        'thigh': ([.2, -.3, 1.5], [.9, .3, 3.1]), 'shin': ([.35, -.2, .75], [.75, .2, 1.6]),
        'cuff': ([.15, -.4, .2], [.95, .4, .5]), 'shoe': ([.1, -.8, 0], [1, .4, .21]),
        'upper': ([.9, -.3, 5.2], [1.6, .3, 5.8]), 'fore': ([1.2, -.2, 1.5], [1.45, .2, 5.25]),
        'glove': ([1.1, -.4, .75], [1.5, .4, 1.52]),
    }
    parts = {k + side: ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])) if side == '+' or k in ('trunk', 'head') else
             ((-hi[0], lo[1], lo[2]), (-lo[0], hi[1], hi[2]))
             for k, (lo, hi) in half.items() for side in ('+', '-') if side == '+' or k not in ('trunk', 'head')}
    names = list(parts)
    points, tris, piece = _figure(parts.values())
    name = {piece[8 * i]: n for i, n in enumerate(names)}
    parents, heads, *_ = segments.skeleton(points, tris, piece)
    tree = {name[p]: name[q] if q >= 0 else None for p, q in enumerate(parents)}
    for side in '+-':
        assert tree['cuff' + side] == 'shin' + side and tree['shoe' + side] == 'cuff' + side
        assert tree['glove' + side] == 'fore' + side
    cuff = heads[[p for p in name if name[p] == 'cuff+'][0]]
    assert abs(cuff[2] - .625) < 1e-6 and .35 <= cuff[0] <= .75           # the middle of the gap, under the shin


def test_a_trunk_spanning_the_model_is_one_piece():
    points, tris, piece = _figure([([-1, -1, 0], [1, 1, 4]), ([.9, -.2, 2], [1.3, .2, 2.4]),
                                   ([-1.3, -.2, 2], [-.9, .2, 2.4])])
    with pytest.raises(segments.OnePiece):
        segments.skeleton(points, tris, piece)


def test_a_limb_hangs_from_the_trunk_where_it_begins_not_where_it_rests():
    # an arm hanging at the side: its upper arm (1) meets the trunk (0) at the shoulder by a little; its glove (2)
    # rests broadly on a flared skirt; the strongest contacts joined the arm to the trunk through the glove
    boxes = {0: ([-1, -.5, 0], [1, .5, 3]), 1: ([1, -.2, 1.5], [1.4, .2, 2.9]), 2: ([1, -.3, 1.1], [1.5, .3, 1.5]),
             3: ([1.1, -.2, .5], [1.4, .2, 1.1])}
    rng = np.random.default_rng(0)
    points = np.vstack([np.array(lo) + rng.random((20, 3)) * (np.array(hi) - lo) for lo, hi in boxes.values()])
    piece = np.repeat(np.arange(4), 20)
    contacts = {(0, 1): (np.array([1, 0, 2.85]), .1), (0, 2): (np.array([1, 0, 1.3]), 1.),
                (1, 2): (np.array([1.2, 0, 1.5]), .5), (2, 3): (np.array([1.25, 0, 1.1]), .9)}
    edges = {k: contacts[k][0] for k in [(0, 2), (2, 3), (1, 2)]}
    out = segments._from_the_root(points, piece, contacts, edges, 0, np.array([0, 1, 1, 1]))
    assert {frozenset(k) for k in out} == {frozenset(k) for k in [(0, 1), (1, 2), (2, 3)]}


def _resting():
    """Sofia's, in contacts: a trunk (0), thighs (1, 2), upper arms (3, 4), forearms (5, 6), each forearm's hand resting
    on its thigh more broadly than it meets the upper arm at the elbow; off the left forearm two whip pieces (7, 8)
    that touch each other too."""
    contacts = {(0, 1): (np.zeros(3), .5), (0, 2): (np.zeros(3), .5), (0, 3): (np.zeros(3), .3),
                (0, 4): (np.zeros(3), .3), (1, 5): (np.zeros(3), .2), (2, 6): (np.zeros(3), .2),
                (3, 5): (np.zeros(3), .1), (4, 6): (np.zeros(3), .1), (5, 7): (np.zeros(3), .3),
                (5, 8): (np.zeros(3), .3), (7, 8): (np.zeros(3), .1)}
    side = np.array([0, 1, -1, 1, -1, 1, -1, 1, 1])
    twin = np.array([0, 2, 1, 4, 3, 6, 5, 7, 8])
    return contacts, side, twin, np.arange(9), np.zeros((9, 3))


def test_the_contacts_on_a_loop_through_the_trunk_are_asked_about_with_their_twins():
    contacts, side, twin, piece, points = _resting()
    edges = segments._tree(points, piece, contacts, side, twin)
    assert (1, 5) in edges and (3, 5) not in edges           # the forearm hangs from the thigh
    # the elbow left out closes a loop through the trunk: its contacts, each with its mirror twin; the whip's knot,
    # a loop off the forearm, is not asked about
    assert segments._loops(edges, contacts, side, twin, 0) == [[(0, 1), (0, 2)], [(0, 3), (0, 4)],
                                                               [(1, 5), (2, 6)], [(3, 5), (4, 6)]]


def test_a_contact_only_touching_joins_pieces_only_where_nothing_else_does():
    contacts, side, twin, piece, points = _resting()
    edges = segments._tree(points, piece, contacts, side, twin, touching={(1, 5), (2, 6)})
    assert {(3, 5), (4, 6)} <= set(edges) and not {(1, 5), (2, 6)} & set(edges)
    # with every way but touching gone, it still joins
    edges = segments._tree(points, piece, {k: v for k, v in contacts.items() if k not in ((3, 5), (4, 6))}, side,
                           twin, touching={(1, 5), (2, 6)})
    assert {(1, 5), (2, 6)} <= set(edges)


def test_two_pieces_only_touch_when_most_answers_say_so():
    def said(*contacts):
        return {'character': 'x', 'contacts': [dict(picture=n, what='', contact=c) for n, c in contacts]}
    answers = [said((1, 'touching'), (2, 'joined')), said((1, 'touching'), (2, 'touching')), said((2, 'cant_tell'))]
    assert segments.tally_touching(answers, [1, 2, 3]) == {1: True, 2: False, 3: False}
