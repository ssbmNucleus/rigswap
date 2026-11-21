from types import SimpleNamespace

import numpy as np
import pytest

from rigswap.core import correspond, repose
from rigswap.core.rig import Rig


@pytest.mark.parametrize('height', [3.0, 0.3])
def test_the_scaled_sole_sets_the_ankle_of_a_tall_or_short_foot(height):
    from rigswap.core import repose
    # A raised source sole distinguishes sole-relative placement from simply copying fighter z.
    heads = np.array([[0, 0, 12], [0, 0, 8], [0, 0, 2 + height], [0, -2, 2.2]], float)
    points = np.array([[0, 0, 2], [0, -1, 2], [0, 0, 2 + height], [0, -3, 2.2]])
    weights = np.array([[0, 0, 1, 0]] * 3 + [[0, 0, 0, 1]], float)
    rig = Rig('leg', ['hip', 'shin', 'boot', 'toe'], np.array([-1, 0, 1, 2]), heads,
              heads + [0, 0, -0.1], ['foot'], points, weights)
    m = correspond.Chain(list(range(4)), heads, np.array([0, -3, 2.2]), [0, 2, 3])
    f = correspond.Chain(list(range(4)), np.array([[1, 0, 10], [1, 0, 5], [1, 0, 1], [1, -2, 1.]]),
                         np.array([1, -3, 1.]), [0, 2, 3])
    placed = correspond._place(f, m)
    original = [p.copy() for _, p in placed]
    evidence = correspond._place_ankle(rig, f, m, placed)
    assert evidence['source_sole_z'] == 2
    assert evidence['embedded_before'][2] == 2 + height
    assert evidence['embedded_after'][2] == 3
    assert np.allclose(placed[2][1], [0, 0, 3])
    assert all(np.array_equal(placed[i][1], original[i]) for i in (0, 1, 3))
    # The new shared point is the shin's tail and foot's head; the end remains a unit rigid move.
    embedded = {i: (placed[i][1], placed[i + 1][1] if i < 3 else m.end) for i in range(4)}
    rest = {i: (f.joints[i], f.joints[i + 1] if i < 3 else f.end) for i in range(4)}
    match = correspond.Match({}, embedded, rest, ends={2, 3}, feet={2}, end_parents={3: 2})
    moves = repose.transforms(match, list(range(4)))
    assert np.allclose(moves[1](np.array([placed[2][1]])), [f.joints[2]])
    assert np.allclose(moves[2](np.array([placed[2][1]])), [f.joints[2]])
    assert np.allclose(moves[2](points), moves[3](points))
    assert np.isclose(np.linalg.norm(np.diff(moves[2](points[:2]), axis=0)), 1)
    assert np.allclose(moves[2](points[:2])[:, 2], 0, atol=1e-14)
    # A tall boot's authored joint now precedes the ankle cut; a short foot's follows it.
    t, _ = m.fractions()
    cut = correspond._fraction_of(m, placed[2][1])
    assert (cut > t[2]) == (height > 1)


def test_an_unweighted_foot_reports_no_sole_instead_of_inventing_one():
    chain = correspond.Chain([0, 1], np.array([[0, 0, 4], [0, 0, 1.]]), np.array([0, -1, 0.]), [0, 1, None])
    rig = Rig('empty', ['hip', 'foot'], np.array([-1, 0]), chain.joints, chain.joints,
              [], np.empty((0, 3)), np.empty((0, 2)))
    placed = correspond._place(chain, chain)
    evidence = correspond._place_ankle(rig, chain, chain, placed)
    assert evidence['source_sole_z'] is None
    assert evidence['embedded_before'] == evidence['embedded_after']


def test_end_parts_follow_the_nearest_matched_ancestor_across_helpers_but_not_segments():
    # head (0), helper (1), jaw (2); hand (4), finger (5); foot (7), toe (8).
    # The neck (3) and arm (6) are ordinary segments even though they descend from another end.
    parents = [-1, 0, 1, 0, 3, 4, 4, 6, 7]
    embedded = {b: ((0, 0, b), (0, 0, b + 1)) for b in [0, 2, 3, 4, 5, 6, 7, 8]}
    ends = {0, 2, 4, 5, 7, 8}
    assert correspond._end_parents(parents, embedded, ends) == {2: 0, 5: 4, 8: 7}
    # with no length inside the model the neck is no segment: the end below it is placed by it
    embedded[3] = ((0, 0, 3), (0, 0, 3))
    assert correspond._end_parents(parents, embedded, ends) == {2: 0, 4: 3, 5: 4, 8: 7}


def _arm_holding():
    """One arm along +x: shoulder (0), forearm (1), hand (2); off the hand a finger skinned with it (3), a thumb cut as
    a rigid piece (4) and a sword bound to a bone of its own (5); off the shoulder a rigid pad (6)."""
    names = ['shoulder', 'forearm', 'hand', 'finger', 'thumb', 'sword', 'pad']
    parents = np.array([-1, 0, 1, 2, 2, 2, 0])
    heads = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.5, 0, 0], [2.1, 0.2, 0], [2.2, 0, 0], [0.2, 0, 0.3]], float)
    pieces = [  # (points, {bone: weight})
        (np.array([[0.5, 0, 0], [1.5, 0, 0]]), {0: 0.5, 1: 0.5}),
        (np.array([[2, -0.2, 0], [2.4, 0.2, 0]]), {2: 1.0}),                    # the palm
        (np.array([[2.6, 0, 0], [3, 0, 0]]), {2: 0.3, 3: 0.7}),                 # the finger, skinned
        (np.array([[2.1, 0.2, 0], [2.3, 0.4, 0]]), {4: 1.0}),                   # the thumb, rigid
        (np.array([[2.2, 0, 0], [2.2, 0, 3], [2.25, 0, 3]]), {5: 1.0}),         # the sword, rigid, 3 long
        (np.array([[0, 0, 0.3], [0.4, 0, 0.5]]), {6: 1.0}),                     # the pad, rigid
    ]
    points = np.vstack([p for p, _ in pieces])
    weights = np.zeros((len(points), len(names)))
    row = 0
    for p, w in pieces:
        for b, x in w.items():
            weights[row:row + len(p), b] = x
        row += len(p)
    rig = Rig('test', names, parents, heads, heads + [0.1, 0, 0], ['arm'], points, weights)
    return rig, SimpleNamespace(arms={'left': [0, 1, 2, 3]})


def test_a_held_prop_is_a_rigid_piece_in_the_fist_bigger_than_the_hand():
    rig, body = _arm_holding()
    held = correspond.held_props(rig, body)
    assert [(rig.names[b], hand) for b, _, _, hand in held] == [('sword', 'left_hand')]
    assert held[0][2].sum() == 3


def test_a_prop_skinned_into_the_hand_is_not_held():
    rig, body = _arm_holding()
    rig.weights[-5:-2, 5], rig.weights[-5:-2, 2] = 0.8, 0.2       # the sword blended with the hand
    assert correspond.held_props(rig, body) == []


def test_a_held_prop_tied_to_the_arm_by_a_strap_is_still_held():
    # u-link's shield: 12 of its 463 vertices are a strap tied to the forearm at 0.7. Here the sword is 20 vertices,
    # one of them tied to the forearm
    rig, body = _arm_holding()
    blade = np.column_stack([np.full(17, 2.2), np.zeros(17), np.linspace(0.2, 2.8, 17)])
    rig.points = np.vstack([rig.points, blade])
    rig.weights = np.vstack([rig.weights, np.eye(len(rig.names))[[5] * 17]])
    rig.weights[-1, [5, 1]] = 0.7, 0.3
    held = correspond.held_props(rig, body)
    assert [rig.names[b] for b, _, _, _ in held] == ['sword'] and held[0][2].sum() == 20
    rig.weights[-3:, [5, 1]] = 0.7, 0.3                         # three of twenty: skinned, no strap
    assert correspond.held_props(rig, body) == []


def test_a_fighter_holding_nothing_leaves_out_a_prop_it_holds_on_bones_of_its_own():
    # tmario's blasters, skinned over three bones of their own off each hand: a mesh all on them goes, and a mesh
    # only partly on them stays; the thumb, the pad and the rest of the arm stay
    rig, body = _arm_holding()
    rig.meshes, rig.counts = ['arm', 'palm', 'finger', 'thumb', 'blaster', 'pad'], [2, 2, 2, 2, 3, 2]
    assert correspond.empty_hands(rig, body) == ['blaster']
    rig.meshes, rig.counts = ['arm', 'palm', 'finger', 'thumb', 'sleeve'], [2, 2, 2, 2, 5]
    assert correspond.empty_hands(rig, body) == []


def _drawn_on_the_hand(glove=(0.3, 0.7), extra=()):
    """One arm along +x, shoulder (0), forearm (1), hand (2), its meshes as an Ocarina of Time rip has them: the
    forearm skinned, a glove skinned into the forearm and the hand (``glove``: its weights on them), and on the
    hand's own bone, each riding it alone, a fist, a shield (2 by 2, on its edge) and the shield's back plate; then
    ``extra`` meshes (name, points, weights)."""
    names, parents = ['shoulder', 'forearm', 'hand'], np.array([-1, 0, 1])
    heads = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    meshes = [('forearm', [[0.5, 0, 0], [1.5, 0, 0]], {0: 0.5, 1: 0.5}),
              ('glove', [[1.8, -0.2, 0], [2.4, 0.2, 0]], dict(zip((1, 2), glove))),
              ('fist', [[2.0, -0.2, 0.1], [2.3, 0.2, 0.1]], {2: 1.0}),
              ('shield', [[2.2, -1, -1], [2.2, 1, -1], [2.2, 1, 1], [2.2, -1, 1]], {2: 1.0}),
              ('plate', [[2.15, -0.9, -0.9], [2.15, 0.9, 0.9]], {2: 1.0}), *extra]
    points = np.array([q for _, p, _ in meshes for q in p], float)
    weights, row = np.zeros((len(points), len(names))), 0
    for _, p, w in meshes:
        for b, x in w.items():
            weights[row:row + len(p), b] = x
        row += len(p)
    rig = Rig('test', names, parents, heads, heads + [0.1, 0, 0], [m for m, _, _ in meshes], points, weights)
    rig.counts = [len(p) for _, p, _ in meshes]
    return rig, SimpleNamespace(arms={'left': [0, 1, 2]}, legs={})


def test_an_object_drawn_on_the_hands_own_bone_is_held_by_its_size_against_the_hands_skin():
    # the shield and its plate are one prop on the hand's bone, no bones of its own; the fist is smaller than the
    # glove. A hand with no skin of its own holds nothing so found, nor does a rig not read from a scene
    rig, body = _drawn_on_the_hand()
    held = correspond.held_props(rig, body)
    assert [(b, sub, hand) for b, sub, _, hand in held] == [(2, [], 'left_hand')]
    assert np.flatnonzero(held[0][2]).tolist() == [6, 7, 8, 9, 10, 11]
    rig, body = _drawn_on_the_hand(glove=(0.0, 1.0))
    assert correspond.held_props(rig, body) == []
    rig, body = _drawn_on_the_hand()
    rig.counts = None
    assert correspond.held_props(rig, body) == []


def test_the_pieces_welded_to_an_object_drawn_on_the_hand_are_its_and_the_hand_holds_it_alone():
    # tlink's left hand: the blade welded to its guard, beside them the hammer and a bomb, each riding the hand's bone
    # alone; his fists skinned into the wrist. Named a weapon, the object is held alone: the rest the file drew on
    # that bone go, the fist stays. Named none, or not asked, nothing goes
    guard = ('guard', [[2.2, -1, -1], [2.2, -1.2, -1.1]], {2: 1.0})
    hammer = ('hammer', [[2.1, 0.3, 0], [2.1, 0.3, 1.0]], {2: 1.0})
    bomb = ('bomb', [[2.4, -0.3, -0.3], [2.4, 0.2, 0.2]], {2: 1.0})
    rig, body = _drawn_on_the_hand(extra=(guard, hammer, bomb))
    rig.weights[4:6, 1], rig.weights[4:6, 2], body.root = 0.2, 0.8, 0
    held = correspond.held_props(rig, body)
    assert np.flatnonzero(held[0][2]).tolist() == [6, 7, 8, 9, 10, 11, 12, 13]
    assert correspond.unheld(rig, body, {'hand': 'shield'}) == ['hammer', 'bomb']
    assert correspond.unheld(rig, body, {'hand': None}) == []
    assert correspond.unheld(rig, body, {}) == []


def _off_the_root(at):
    """A root at the floor (0), the hips above it (1), the body on them, and a prop off the root (2, its tip 3), its
    bones' heads at ``at``, carrying a pole from the floor up."""
    names, parents = ['root', 'hips', 'prop', 'tip'], np.array([-1, 0, 0, 2])
    heads = np.array([[0, 0, 0], [0, 0, 1], at, at], float)
    points = np.array([[-0.3, 0, 0.6], [0.3, 0, 1.6], [0, 0, 0], [0, 0, 1.5]], float)
    weights = np.zeros((4, 4))
    weights[:2, 1], weights[2:, 2] = 1.0, 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, 0.1], ['body', 'pole'], points, weights)
    rig.counts = [2, 2]
    return rig, SimpleNamespace(root=1, arms={}, legs={}, left=np.array([1.0, 0, 0]))


def test_a_prop_no_slot_takes_on_bones_left_at_the_files_origin_goes():
    # tlink's pole: its bones where the root is, at the floor. Laid on him (its bones on his back) it rides there, as
    # one named a weapon goes where its slot takes it
    rig, body = _off_the_root([0, 0, 0])
    assert correspond.unheld(rig, body, {'prop': None}, _slots()) == ['pole']
    assert correspond.unheld(rig, body, {'prop': 'sword'}, _slots()) == []
    rig, body = _off_the_root([0, -0.2, 1.2])
    assert correspond.unheld(rig, body, {'prop': None}, _slots()) == []
    # named a sword (ns-8b; a rod in ns-7) beside the sword the slot takes, it is no weapon all the same: it goes
    rig, body = _off_the_root([0, 0, 0])
    rig = Rig('test', rig.names + ['blade'], np.append(rig.parents, 0), np.vstack([rig.heads, [0.6, 0, 1]]),
              np.vstack([rig.tails, [0.6, 0, 1.1]]), rig.meshes + ['sword'],
              np.vstack([rig.points, [[0.6, 0, 0.8], [0.6, 0, 2.6]]]),
              np.vstack([np.hstack([rig.weights, np.zeros((4, 1))]), np.eye(5)[[4, 4]]]))
    rig.counts = [2, 2, 2]
    assert correspond.unheld(rig, body, {'prop': 'sword', 'blade': 'sword'}, _slots()) == ['pole']
    assert correspond.unheld(rig, body, {'prop': 'sword', 'blade': 'shield'}, _slots()) == []


def test_a_branch_carrying_a_limb_the_body_took_is_no_prop():
    rig, body = _off_the_root([0, 0, 1])
    assert [p[0] for p in correspond.props(rig, body)] == [2]
    body.legs = {'left': [1, 2]}            # Swampert's feet hang off a floor control beside its hips
    assert correspond.props(rig, body) == []


def _drawn_shield(slots, kinds):
    """_two_hands with the buckler drawn on the right hand's own bone (Ocarina of Time's), its hand a glove skinned
    into the arm: _match_props's slots and whole."""
    rig, body, fighter = _two_hands()
    rig.weights[rig.weights[:, 6] > 0.5, 5], rig.weights[:, 6] = 1.0, 0.0
    glove = np.flatnonzero(rig.weights[:, 5] > 0.5)[:2]
    rig.weights[glove, 4], rig.weights[glove, 5] = 0.3, 0.7
    rig.counts = [2, 2, 2, 3, 2, 2, 4]
    owner, placed, replaced, notes, whole = {}, {}, [], [], {}
    hands = {1: (np.array([2., 0, 0]), np.array([2.5, 0, 0])), 3: (np.array([-2., 0, 0]), np.array([-2.5, 0, 0]))}
    embedded, rest = dict(hands), {b: (h + [0, 0, 8], t + [0, 0, 8]) for b, (h, t) in hands.items()}
    correspond._match_props(rig, body, fighter, {}, slots, owner, embedded, rest, placed, replaced, notes, kinds,
                            whole)
    return replaced, owner, whole


def test_an_object_drawn_on_the_hand_is_taken_only_by_the_slot_of_the_kind_it_is_named():
    replaced, owner, whole = _drawn_shield(_slots(), {'blade': 'sword', 'r_hand': 'shield'})
    assert [(r['part'], r['item']) for r in replaced] == [('sword', 'blade'), ('shield', 'r_hand')]
    assert 5 not in owner and np.flatnonzero(whole[4]).tolist() == [13, 14, 15, 16]
    replaced, _, whole = _drawn_shield(_slots(), {'blade': 'sword', 'r_hand': 'none'})
    assert [r['part'] for r in replaced] == ['sword'] and 4 not in whole
    # one kind of slot, nothing asked: the longer object drawn on the hand is no sword
    replaced, _, _ = _drawn_shield(_slots()[:1], None)
    assert [(r['part'], r['item']) for r in replaced] == [('sword', 'blade')]


def _leg(off_foot):
    """A leg down to its foot on the floor (hip, thigh, knee, foot); off the knee a plate in front of the shin,
    hanging short of the floor; off the foot ``off_foot``: [(name, head, its points)]."""
    names = ['hip', 'thigh', 'knee', 'foot', 'plate'] + [n for n, _, _ in off_foot]
    parents = np.array([-1, 0, 1, 2, 2] + [3] * len(off_foot))
    heads = np.array([[0, 0, 1], [0, 0, .9], [0, 0, .5], [0, -.02, .08], [0, -.12, .3]]
                     + [h for _, h, _ in off_foot], float)
    pieces = [[[0, 0, .95]], [[0, 0, .7]], [[0, 0, .3]], [[0, -.05, .02], [0, .02, .03]], [[0, -.14, .2], [0, -.14, .4]]]
    pieces += [p for _, _, p in off_foot]
    points = np.array([q for p in pieces for q in p], float)
    weights = np.zeros((len(points), len(names)))
    weights[np.arange(len(points)), [b for b, p in enumerate(pieces) for _ in p]] = 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, -.05], ['leg'], points, weights)
    return rig, SimpleNamespace(up=np.array([0., 0., 1.]), front=np.array([0., -1., 0.]), forks={}, groups={})


def test_a_plate_off_the_knee_is_no_toe():
    rig, shape = _leg([])
    assert correspond._forward_toe(rig, shape, [0, 1, 2, 3]) == [0, 1, 2, 3]


def test_a_foot_turns_to_its_toe():
    # the foot's own bone stands on the floor; off it a heel behind and a toe in front, both on the floor
    rig, shape = _leg([('heel', [0, .05, .04], [[0, .06, 0]]), ('toe', [0, -.08, .03], [[0, -.12, 0]])])
    assert correspond._forward_toe(rig, shape, [0, 1, 2, 3, 5]) == [0, 1, 2, 3, 6]


def test_a_skirt_strand_is_fitted_to_the_fighters_when_longer_and_keeps_its_length_when_shorter():
    strand = correspond.Chain([0, 1], np.array([[0, 0, 0], [0, 0, -1.]]), np.array([0, 0, -3.]))     # 3 long
    joints, tail = correspond._fit(strand, np.array([[0, 0, 0], [0, 0, -6.]]))
    assert np.allclose(joints[:, 2], [0, -2]) and np.isclose(tail[2], -6)       # its hem at the strand's end
    joints, tail = correspond._fit(strand, np.array([[0, 0, 0], [0, 0, -2.]]))
    assert np.allclose(joints[:, 2], [0, -1]) and np.isclose(tail[2], -3)       # the strand runs on past its hem


def test_a_skirt_strand_squashes_with_the_fighters_shorter_legs():
    strand = correspond.Chain([0, 1], np.array([[0, 0, 0], [0, 0, -1.]]), np.array([0, 0, -3.]))     # 3 long
    # on legs half the model's, a skirt 4 long hangs 2: its hem two thirds down the strand, not past it (to the floor)
    joints, tail = correspond._fit(strand, np.array([[0, 0, 0], [0, 0, -4.]]), 0.5)
    assert np.allclose(joints[:, 2], [0, -2]) and np.isclose(tail[2], -6)
    # one 8 long hangs 4, past the strand's end: fitted to it
    joints, tail = correspond._fit(strand, np.array([[0, 0, 0], [0, 0, -8.]]), 0.5)
    assert np.allclose(joints[:, 2], [0, -8 / 3]) and np.isclose(tail[2], -8)


def test_the_legs_squash_by_the_height_of_the_hip_joints_and_never_stretch():
    fch = {f'{side} leg': correspond.Chain([b], np.zeros((1, 3)), np.zeros(3))
           for side, b in (('left', 3), ('right', 4))}
    model = {3: (np.array([1, 0, 10.]), None), 4: (np.array([-1, 0, 10.]), None)}
    short = {3: (np.array([.5, 0, 8.]), None), 4: (np.array([-.5, 0, 8.]), None)}
    tall = {3: (np.array([.5, 0, 12.]), None), 4: (np.array([-.5, 0, 12.]), None)}
    assert np.isclose(correspond._squash(fch, model, short), 0.8) and correspond._squash(fch, model, tall) == 1.0


def test_a_skirt_strand_ends_where_its_mesh_does_and_a_cape_as_its_bones_do():
    """Hips (0) with three skirt strands off it, each a bone carrying mesh and a tip carrying none (Peach's), and a
    chest (1) with a cape of the same build (10: the spine between them, 11 the head, 12-15 the legs and arms)."""
    names = [f'JOBJ_{i}' for i in range(16)]
    parents = np.array([-1, 10, 0, 2, 0, 4, 0, 6, 1, 8, 0, 1, 0, 0, 1, 1])
    heads = np.array([[0, 0, 10], [0, 0, 14], [-1, 0, 9], [-1, 0, 5], [0, 0, 9], [0, 0, 5], [1, 0, 9], [1, 0, 5],
                      [0, 1, 14], [0, 1, 9], [0, 0, 12], [0, 0, 16], [1, 0, 10], [-1, 0, 10], [1, 0, 14],
                      [-1, 0, 14]], float)
    tails = heads + [0, 0, -4.]
    tails[[3, 5, 7, 9]] = heads[[3, 5, 7, 9]] + [0, 3, 0.]                  # the tips lie out flat
    points = np.array([[-1, 0, 7], [-1, 0, 6], [0, 0, 7], [0, 0, 6], [1, 0, 7], [1, 0, 6], [0, 1, 12], [0, 1, 10]],
                      float)
    weights = np.zeros((len(points), len(names)))
    for v, b in enumerate([2, 2, 4, 4, 6, 6, 8, 8]):
        weights[v, b] = 1.0
    frig = Rig('fighter', names, parents, heads, tails, ['skin'], points, weights)
    description = {'chains': [{'kind': 'other', 'bones': [3, 4]}, {'kind': 'other', 'bones': [5, 6]},
                              {'kind': 'other', 'bones': [7, 8]}, {'kind': 'cape', 'bones': [9, 10]}]}
    fch = {'spine': correspond.Chain([0, 10, 1, 11], heads[[0, 10, 1, 11]], np.array([0, 0, 18.]), [0, 2, 3])}
    for side, leg, arm in (('left', 12, 14), ('right', 13, 15)):
        fch[f'{side} arm'] = correspond.Chain([arm], heads[[arm]], heads[arm] * 2, [0, 0, 0])
        fch[f'{side} leg'] = correspond.Chain([leg], heads[[leg]], heads[leg] * [1, 1, 0], [0, 0, None])
    found = correspond.extra_chains(frig, description, fch)
    assert [c.bones for _, _, c in found] == [[2], [4], [6], [8, 9]]
    assert found[0][2].end[2] >= 6 and np.allclose(found[3][2].end, tails[9])


def _caped(collar=False):
    """A fighter whose game swings a cape off its chest (JOBJ_8 and 9 off JOBJ_1: 0 the hips, 10 the spine, 11 the
    head, 12-15 the legs and arms), and a model on it: hips 0, chest 1, head 2, arms 3-4, a cape (5, 6) and a wing (7)
    off its chest where the fighter's cape hangs, a sleeve (8) off its left arm and a ribbon (9) off its head, each
    bone carrying its own mesh; with ``collar``, the cape hangs off a bone at the chest's own joint (10), carrying
    none."""
    names = [f'JOBJ_{i}' for i in range(16)]
    parents = np.array([-1, 10, 0, 2, 0, 4, 0, 6, 1, 8, 0, 1, 0, 0, 1, 1])
    heads = np.array([[0, 0, 10], [0, 0, 14], [-1, 0, 9], [-1, 0, 5], [0, 0, 9], [0, 0, 5], [1, 0, 9], [1, 0, 5],
                      [0, 1, 14], [0, 1, 9], [0, 0, 12], [0, 0, 16], [1, 0, 10], [-1, 0, 10], [1, 0, 14],
                      [-1, 0, 14]], float)
    points = np.array([[0, 1, 12], [0, 1, 7]], float)
    weights = np.zeros((len(points), len(names)))
    weights[[0, 1], [8, 9]] = 1.0
    frig = Rig('fighter', names, parents, heads, heads + [0, 0, -4.], ['skin'], points, weights)
    description = {'bones': {'roles': {}}, 'chains': [{'kind': 'cape', 'bones': [9, 10]}]}
    fch = {'spine': correspond.Chain([0, 10, 1, 11], heads[[0, 10, 1, 11]], np.array([0, 0, 18.]), [0, 2, 3])}
    for side, leg, arm in (('left', 12, 14), ('right', 13, 15)):
        fch[f'{side} arm'] = correspond.Chain([arm], heads[[arm]], heads[arm] * 2, [0, 0, 0])
        fch[f'{side} leg'] = correspond.Chain([leg], heads[[leg]], heads[leg] * [1, 1, 0], [0, 0, None])
    model = ['hips', 'chest', 'head', 'arm.L', 'arm.R', 'cape', 'cape2', 'wing', 'sleeve', 'ribbon']
    mparents = np.array([-1, 0, 1, 1, 1, 1, 5, 1, 3, 2])
    mheads = np.array([[0, 0, 10], [0, 0, 14], [0, 0, 16], [1, 0, 14], [-1, 0, 14], [0, 1, 14], [0, 1, 11],
                       [0, 1, 14], [1.5, 0, 14], [0, 1, 17]], float)
    mpoints = np.array([[0, 0, 11], [0, 0, 14], [0, 0, 17], [2, 0, 14], [-2, 0, 14], [0, 1, 12], [0, 1, 8],
                        [0, 1.5, 13], [1.5, 0, 12], [0, 1.5, 16]], float)
    weights = np.eye(len(model))
    if collar:
        model, mparents = model + ['collar'], np.append(mparents, 1)
        mparents[5] = 10
        mheads, mpoints = np.vstack([mheads, [0, 0, 14]]), np.vstack([mpoints, [0, 0, 14]])
        weights = np.pad(weights, ((0, 1), (0, 1)))
    rig = Rig('model', model, mparents, mheads, mheads + [0, 0, -1.], ['body'], mpoints, weights)
    mch = {'spine': correspond.Chain([0, 1, 2], mheads[[0, 1, 2]], np.array([0, 0, 18.]), [0, 1, 2])}
    owner = {0: 0, 1: 1, 2: 11, 3: 14, 4: 15}
    return rig, frig, description, fch, mch, owner


def test_a_chain_swings_on_the_fighters_only_if_it_is_of_a_kind_that_fits_it_and_hair_and_cloth_hang():
    """What a chain is (labels.chains) decides what it does: the cape goes on the fighter's cape; the wing hanging
    beside it is no cape, and rides; the sleeve is cloth off a limb, and hangs; the ribbon nobody named rides."""
    rig, frig, description, fch, mch, owner = _caped()
    named = {'cape': {'kind': 'cloth', 'held_out': False}, 'cape2': {'kind': 'cloth', 'held_out': False},
             'wing': {'kind': 'wing', 'held_out': False}, 'sleeve': {'kind': 'cloth', 'held_out': True}}
    notes = []
    loose, swung, _ = correspond._match_extra(rig, frig, description, fch, mch, owner, {}, {}, notes, named=named)
    assert {owner[5], owner[6]} <= {8, 9} and swung == {8, 9}
    assert 7 not in owner and 9 not in owner                                    # riding the bone they hang from
    (attach, chain, carries), = loose
    assert (attach, chain.bones, carries, chain.kind, chain.held_out) == (3, [8], [8], 'cloth', True)
    # with nothing named, nothing goes on the fighter's chains and nothing hangs
    rig, frig, description, fch, mch, owner = _caped()
    loose, swung, _ = correspond._match_extra(rig, frig, description, fch, mch, owner, {}, {}, [])
    assert loose == [] and swung == set() and 5 not in owner


def test_a_chain_off_a_bone_at_a_joint_of_the_body_hangs_from_that_joint():
    # the cape hangs off a bone at the chest's own joint (Lucario's tail off the Hips at his Waist): it is the chest's
    rig, frig, description, fch, mch, owner = _caped(collar=True)
    body = SimpleNamespace(groups={0: [0], 1: [1, 10], 2: [2]}, plan='biped', root=0, skull=2)
    named = {'cape': {'kind': 'cloth', 'held_out': False}, 'cape2': {'kind': 'cloth', 'held_out': False}}
    _, swung, _ = correspond._match_extra(rig, frig, description, fch, mch, owner, {}, {}, [], body, named=named)
    assert swung == {8, 9} and {owner[5], owner[6]} <= {8, 9}


def _trunk(hips, chest, head, shoulders):
    spine = correspond.Chain([0, 1, 2], np.array([hips, chest, head], float), np.array(head, float) + [0, 0, .3],
                             [0, 1, 2], level=True)
    arms = [correspond.Chain([3], np.array([[x * shoulders[0], shoulders[1], shoulders[2]]]), np.zeros(3), [0, 0, 0])
            for x in (1, -1)]
    return spine, arms


def test_a_trunk_carried_level_is_measured_along_it():
    # hips behind, chest ahead at their height, the neck rising to the head: along the trunk, then up the neck
    spine, arms = _trunk([0, .4, 1], [0, -.3, 1], [0, -.6, 1.4], [.15, -.35, .95])
    assert np.isclose(correspond._model_level(spine, arms, np.array([0, -.3, 1.])), 1.7)
    assert np.isclose(correspond._model_level(spine, arms, np.array([0, -.6, 1.4])), 2.2)
    assert np.isclose(correspond._model_level(spine, arms, np.array([0, -.6, 1.7])), 2.5)     # the head stands up
    assert np.allclose(correspond._at_level(np.array([[0, .4, 1], [0, -.3, 1], [0, -.6, 1.4]]), 1.35), [0, .05, 1])


def test_an_upright_trunk_is_measured_by_height():
    spine, arms = _trunk([0, 0, 1], [0, 0, 1.5], [0, 0, 1.8], [.15, 0, 1.55])
    assert correspond._model_level(spine, arms, np.array([.3, -.2, 1.23])) == 1.23


def _segmented():
    """A segmented rip standing on +z, facing -y: a one-piece torso (its pivot at the chest, the arms and the head hung
    from it), a pelvis piece hung from the torso below its pivot, the legs hung from the pelvis, each bone carrying the
    mesh down to the next joint."""
    names, parents, heads, pieces = [], [], [], []

    def bone(name, parent, head, mesh):
        names.append(name)
        parents.append(names.index(parent) if parent else -1)
        heads.append(head)
        pieces.append(mesh)
    bone('torso', None, [0, 0, 24], [[0, 0, 22], [1.5, 0, 25], [-1.5, 0, 25], [0, 0, 27]])
    bone('pelvis', 'torso', [0, 0, 20], [[0, 0, 18], [1, 0, 19], [-1, 0, 19]])
    bone('head', 'torso', [0, 0, 27.4], [[0, 0, 28], [0, -1, 30], [0, 0, 32]])
    for side, x in (('L', 1), ('R', -1)):
        bone(f'arm{side}', 'torso', [x * 3.7, 0, 27], [[x * 4.5, 0, 27]])
        bone(f'fore{side}', f'arm{side}', [x * 5.3, 0, 21], [[x * 5.5, 0, 18]])
        bone(f'hand{side}', f'fore{side}', [x * 6, 0, 15.6], [[x * 6, 0, 14]])
        thigh = f'thigh{side}'
        bone(thigh, 'pelvis', [x * 1.8, 0, 17.25], [[x * 1.8, 0, 14]])
        bone(f'shin{side}', thigh, [x * 1.9, 0, 10.3], [[x * 1.9, 0, 6]])
        bone(f'foot{side}', f'shin{side}', [x * 1.9, 0, 1.9], [[x * 1.9, 0, 0], [x * 1.9, -1.5, 0]])
    points = np.array([q for p in pieces for q in p], float)
    weights = np.zeros((len(points), len(names)))
    weights[np.arange(len(points)), [b for b, p in enumerate(pieces) for _ in p]] = 1.0
    heads = np.array(heads, float)
    rig = Rig('test', names, np.array(parents), heads, heads + [0, 0, .5], ['body'], points, weights)
    b = {n: i for i, n in enumerate(names)}
    legs = {side: [b['torso'], b['pelvis'], b[f'thigh{s}'], b[f'shin{s}'], b[f'foot{s}']]
            for side, s in (('left', 'L'), ('right', 'R'))}
    body = SimpleNamespace(up=np.array([0., 0., 1.]), front=np.array([0., -1., 0.]), left=np.array([1., 0., 0.]),
                           forks={}, groups={},
                           root=b['torso'], spine=[b['torso'], b['head']], skull=b['head'], chest=b['torso'],
                           legs=legs, arms={side: [b[f'arm{s}'], b[f'fore{s}'], b[f'hand{s}']]
                                            for side, s in (('left', 'L'), ('right', 'R'))},
                           plan='biped')
    return rig, body


def test_a_one_piece_trunks_hips_are_where_its_legs_part_not_at_its_pivot():
    # the torso piece turns about its chest; its legs hang from the pelvis piece below: the torso is read from there
    rig, body = _segmented()
    chains, pelvis = correspond.model_chains(rig, body)
    assert np.allclose(chains['spine'].joints[0], [0, 0, 20])
    assert [rig.names[b] for b in pelvis] == ['torso', 'pelvis']
    hips, shoulders, head, top = correspond.levels(chains)
    assert (hips, shoulders, head) == (20, 27, 27.4) and np.isclose(top, 32)


def _line(n, heads, parents=None):
    """A fighter of ``n`` bones in a line up z (each the last one's child), heads at ``heads``."""
    return SimpleNamespace(names=[f'JOBJ_{i}' for i in range(n)], heads=np.array(heads, float),
                           parents=np.array(parents if parents is not None else [-1] + list(range(n - 1))))


def _pieces(zs, bones=None):
    """A model of rigid pieces up z, one vertex each at ``zs`` on its bone (``bones``, default one per vertex)."""
    bones = bones if bones is not None else list(range(len(zs)))
    n = max(bones) + 1
    heads = np.array([[0, 0, float(i)] for i in range(n)])
    weights = np.zeros((len(zs), n))
    weights[np.arange(len(zs)), bones] = 1.0
    return Rig('pieces', [f'piece{i}' for i in range(n)], np.array([-1] + list(range(n - 1))), heads,
               heads + [0, 0, 1.], ['body'], np.array([[0, 0, z] for z in zs], float), weights)


def test_a_rigid_piece_at_a_chains_end_rides_whole_the_bone_paired_with_its_joint():
    # Scyther's blade, Kritter's one-piece arm, Kamek's boot: nothing beyond them on their chain to keep on, so the
    # fighter joints inside them don't bend them: each turns about its own joint, as its author built it
    rig = _pieces([0.1, 3.5])
    blade = correspond.Chain([0, 1], np.array([[0, 0, 0.], [0, 0, 1.]]), np.array([0, 0, 4.]))
    owner = {0: 10, 1: 10}
    left = correspond._whole(rig, _line(3, [[0, 0, 0]] * 3), owner, {1: 12}, [(1, blade, [(0.6, 11)])], [])
    assert left == [] and owner[1] == 12


def test_a_rigid_piece_bends_where_riding_whole_would_part_it_from_the_next_piece():
    # Leon on Falcon: the torso piece (bone 1) has the fighter's chest and neck joints inside it, and the head piece
    # beyond rides the fighter's head. Whole on the chest, the head would part from it at the neck: so it bends there
    rig = _pieces([0.5, 1.5, 2.5, 3.5])
    spine = correspond.Chain([0, 1, 2], np.array([[0, 0, 0.], [0, 0, 1.], [0, 0, 3.]]), np.array([0, 0, 4.]))
    fighter = _line(4, [[0, 0, 0], [0, 0, 1], [0, 0, 1.8], [0, 0, 2.6]])
    owner = {0: 0, 1: 1, 2: 3}
    notes = []
    left = correspond._whole(rig, fighter, owner, {}, [(1, spine, [(0.45, 2)])], notes)
    assert left == [(1, spine, [(0.45, 2)])] and owner[1] == 1 and 'bends at JOBJ_2' in notes[0]


def _leg_of_pieces(ankle):
    """A leg of rigid pieces down z, thigh (bone 0, hip at 9), shin (1, knee at 5) and foot (2, its own ankle at 1),
    on a fighter's thigh, shin and foot (JOBJ_0..2), the fighter's ankle placed at height ``ankle`` in it: (rig,
    fighter, owner, paired, the cuts its joints make)."""
    rig = _pieces([7., 4., 2., 0.5], bones=[0, 1, 1, 2])
    leg = correspond.Chain([0, 1, 2], np.array([[0, 0, 9.], [0, 0, 5.], [0, 0, 1.]]), np.array([0, 0, 0.]))
    fighter = _line(3, [[0, 0, 9], [0, 0, 5], [0, 0, 3]])
    cuts = correspond._cuts(leg, [np.array([0, 0, 5.]), np.array([0, 0, float(ankle)])], [1, 2])
    return rig, fighter, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}, cuts


def test_a_rigid_piece_rides_whole_when_the_next_piece_turns_about_its_end():
    # a shin and a foot built as pieces, the fighter's ankle where they meet: no joint falls inside the shin, nothing
    # cuts it, and it rides the fighter's shin whole as its author built it
    rig, fighter, owner, paired, cuts = _leg_of_pieces(ankle=1.)
    assert cuts == [] and correspond._whole(rig, fighter, owner, paired, cuts, []) == [] and owner[1] == 1


def test_a_rigid_piece_bends_where_the_next_piece_turns_about_a_joint_inside_it():
    # Frieza on Mewtwo: the floor puts the fighter's ankle well up the shin piece, and the foot piece beyond rides the
    # fighter's foot, the bone right after the shin's. Whole on the shin, the piece would hang past the joint the foot
    # turns about and part from it there: so it bends at that joint, and what lies below goes with the foot
    rig, fighter, owner, paired, cuts = _leg_of_pieces(ankle=3.)
    notes = []
    left = correspond._whole(rig, fighter, owner, paired, cuts, notes)
    assert [(mb, [fb for _, fb in at]) for mb, _, at in left] == [(1, [2])] and np.isclose(left[0][2][0][0], 6 / 9)
    assert owner[1] == 1 and 'bends at JOBJ_2' in notes[0]
    bones = [0, 1, 2]
    w = repose.split(rig, repose.fighter_weights(rig, owner, bones), bones, left, owner)
    assert w[1].tolist() == [0, 1, 0] and w[2].tolist() == [0, 0, 1]      # above the fighter's ankle, and below it


def test_a_helper_at_the_next_pieces_joint_inside_a_rigid_piece_bends_it_too():
    # Melee's leg helpers turn at the joint of the bone below them: a cut there is at that bone's joint
    rig, _, owner, paired, cuts = _leg_of_pieces(ankle=3.)
    fighter = _line(4, [[0, 0, 9], [0, 0, 5], [0, 0, 3], [0, 0, 3]])
    owner[2] = paired[2] = 3
    assert correspond._whole(rig, fighter, owner, paired, cuts, []) == cuts


def test_a_rigid_piece_rides_whole_when_the_next_piece_rides_its_bone_with_it():
    # the joint inside it is one the next piece does not turn about: both ride one bone, and nothing parts them
    rig = _pieces([0.5, 1.5, 2.5, 3.5])
    spine = correspond.Chain([0, 1, 2], np.array([[0, 0, 0.], [0, 0, 1.], [0, 0, 3.]]), np.array([0, 0, 4.]))
    fighter = _line(4, [[0, 0, 0], [0, 0, 1], [0, 0, 1.8], [0, 0, 2.6]])
    owner = {0: 0, 1: 1, 2: 1}
    assert correspond._whole(rig, fighter, owner, {}, [(1, spine, [(0.45, 2)])], []) == [] and owner[1] == 1


def test_bones_are_next_to_each_other_through_helpers_turning_at_the_joint_only():
    # 0 thigh, 1 a helper at the knee, 2 the shin from the knee; 3 a helper away from the joint below it, 4 the foot
    fighter = _line(5, [[0, 0, 10], [0, 0, 5], [0, 0, 5], [0, 0, 3], [0, 0, 1]])
    assert correspond._next_to(fighter, 0, 2) and correspond._next_to(fighter, 0, 1)
    assert not correspond._next_to(fighter, 2, 4) and not correspond._next_to(fighter, 2, 0)


def test_a_bone_hanging_off_a_cut_bone_rides_the_stretch_it_hangs_from():
    # a torso cut at the fighter's neck joint; a collar hung from it near the top rides the neck, a belt near the
    # bottom the chest
    rig = _pieces([0.5, 2.5, 2.9, 0.6], bones=[0, 0, 1, 2])
    rig.parents = np.array([-1, 0, 0])
    rig.heads = np.array([[0, 0, 0.], [0, 0, 2.8], [0, 0, 0.4]])
    torso = correspond.Chain([0], np.array([[0, 0, 0.]]), np.array([0, 0, 3.]))
    owner = {0: 'chest'}
    cut_at = {0: (0, torso, [(0.5, 'neck')])}
    correspond._rest_of_tree(rig, owner, 'hips', lambda mb, p: correspond._carrier(owner, cut_at, mb, p))
    assert owner == {0: 'chest', 1: 'neck', 2: 'chest'}


def _spine(m_heads, weights_on, parents, f_heads=None, f_parents=None):
    """A model spine (its first four bones, arms after) and a fighter's (hips, spine, chest, neck, head, two arms)."""
    f_heads = f_heads or [[0, 0, 8], [0, 0, 9], [0, 0, 10], [0, 0, 11.5], [0, 0, 12], [1, 0, 11], [-1, 0, 11]]
    frig = _line(7, f_heads, parents=f_parents or [-1, 0, 1, 2, 3, 2, 2])
    fch = {'spine': correspond.Chain([0, 1, 2, 3, 4], frig.heads[:5], np.array([0, 0, 14.]), [0, 2, 4])}
    m_heads = np.array(m_heads, float)
    n = len(m_heads)
    zs = [z for z, _ in weights_on]
    weights = np.zeros((len(zs), n))
    weights[np.arange(len(zs)), [b for _, b in weights_on]] = 1.0
    rig = Rig('spine', [f'b{i}' for i in range(n)], np.array(parents), m_heads, m_heads + [0, 0, .5],
              ['body'], np.array([[0, 0, z] for z in zs], float), weights)
    k = n - 2
    mch = {'spine': correspond.Chain(list(range(k)), m_heads[:k], np.array([0, 0, 16.]), [0, 1, k - 1])}
    for side, b, fb in (('left', k, 5), ('right', k + 1, 6)):
        mch[f'{side} arm'] = correspond.Chain([b], m_heads[[b]], m_heads[b] * 2, [0, 0, 0])
        fch[f'{side} arm'] = correspond.Chain([fb], frig.heads[[fb]], frig.heads[fb] * 2, [0, 0, 0])
    return rig, frig, fch, mch


def test_a_fighter_joint_inside_a_model_spine_bone_cuts_it_and_the_head_stays_on():
    # the fighter's neck (bone 3, from 11.5) falls well inside the model's long neck bone (2, from 11 to the head at
    # 14); its spine and chest joints are near the model's own (10, 11): they are those joints, and cut nothing
    rig, frig, fch, mch = _spine([[0, 0, 9], [0, 0, 10], [0, 0, 11], [0, 0, 14], [1, 0, 11.5], [-1, 0, 11.5]],
                                 [(9.5, 0), (10.5, 1), (11.2, 2), (13, 2), (14.5, 3), (15, 3)], [-1, 0, 1, 2, 1, 1])
    owner, embedded, rest = {}, {}, {}
    cuts = correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert owner == {0: 0, 1: 1, 2: 2, 3: 4}
    assert [(mb, [fb for _, fb in c]) for mb, _, c in cuts] == [(2, [3])]
    assert np.allclose(embedded[1][0], [0, 0, 10]) and np.allclose(embedded[2][0], [0, 0, 11])
    assert correspond._apart_at(rig, frig, owner, lambda mb, p: owner[mb]) != []        # whole, the head parts
    cut_at = {c[0]: c for c in cuts}
    assert correspond._apart_at(rig, frig, owner, lambda mb, p: correspond._carrier(owner, cut_at, mb, p)) == []


def test_a_neckless_head_sits_on_its_torso_where_the_fighter_neck_starts():
    # Leon: his arms hang just under his head joint, so the fighter's neck lands a hair under it, in his torso. It is
    # the head joint, of no length in the model: no sliver of torso stretches up the fighter's neck, and the head,
    # placed by the neck, sits on the torso where the fighter's neck starts, not a neck's length above it
    from rigswap.core import repose
    rig, frig, fch, mch = _spine([[0, 0, 9], [0, 0, 10], [0, 0, 11], [0, 0, 12.2], [1, 0, 12], [-1, 0, 12]],
                                 [(9.5, 0), (10.5, 1), (11.5, 2), (12.1, 2), (13, 3), (15, 3)], [-1, 0, 1, 2, 2, 2])
    owner, embedded, rest = {}, {}, {}
    cuts = correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert owner == {0: 0, 1: 1, 2: 2, 3: 4} and cuts == []
    assert np.allclose(embedded[3][0], [0, 0, 12.2]) and correspond._no_length(embedded[3])
    end_parents = correspond._end_parents(frig.parents, embedded, {4})
    assert end_parents == {4: 3}
    moves = repose.transforms(correspond.Match(owner, embedded, rest, ends={4}, end_parents=end_parents),
                              sorted(embedded))
    joint = np.array([[0, 0, 12.2]])
    assert np.allclose(moves[4](joint), frig.heads[3]) and np.allclose(moves[2](joint), frig.heads[3])
    at = lambda mb, p: owner[mb]
    assert correspond._apart_at(rig, frig, owner, at) != []                # by the head bone alone, it parts
    assert correspond._apart_at(rig, frig, owner, at, through={4: 3}) == []


def test_a_head_joint_a_hair_under_its_shoulders_leaves_the_fighter_neck_no_length():
    # the CarnEvil skeleton on Ganondorf (the app's import of 2026-11-10): the joint named for its head stood at 80.5 %
    # of its height, its arm roots at 80.6 %. The head's level is lifted to the shoulders' (a neckless head starts
    # there), and the fighter's head was placed at that level, the hair above the joint: the hair was the fighter's
    # neck, and the model's neck bone was stretched 14.58 times along it
    rig, frig, fch, mch = _spine([[0, 0, 9], [0, 0, 10], [0, 0, 11], [0, 0, 11.95], [1, 0, 12], [-1, 0, 12]],
                                 [(9.5, 0), (10.5, 1), (11.5, 2), (11.9, 2), (13, 3), (15, 3)], [-1, 0, 1, 2, 2, 2])
    owner, embedded, rest = {}, {}, {}
    cuts = correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert np.allclose(embedded[3][0], [0, 0, 11.95]) and correspond._no_length(embedded[3])    # no sliver of a neck
    assert np.allclose(embedded[4][0], [0, 0, 11.95])                                   # the head is on its joint
    assert owner == {0: 0, 1: 1, 2: 2, 3: 4} and cuts == []                             # and the head bone is the head's
    assert correspond._end_parents(frig.parents, embedded, {4}) == {4: 3}


def test_a_spine_place_snaps_only_between_the_places_either_side_of_it():
    # Meta Knight: his head joint lies below the fighter's chest joint's place (his arms hang above it), near enough
    # the neck's to snap there. Snapped, the neck would come before the chest, whose bone would run down the model and
    # turn his ball upside down onto it. Past the chest's place (Leon), the neck is the head joint
    from types import SimpleNamespace
    rig = SimpleNamespace(points=np.array([[0, 0, 0], [0, 0, 10.]]))
    places = [[0, 0, z] for z in (8.43, 8.64, 8.98, 9.64, 9.8)]
    for head, snapped in ((8.2, False), (9.2, True)):
        m = correspond.Chain([0, 1, 2], np.array([[0, 0, 0], [0, 0, 1], [0, 0, head]]), np.array([0, 0, 10.]))
        out, on = correspond._snap(rig, m, places, 2, neck=3)
        assert (on[3] == 2) is snapped and np.allclose(out[3], [0, 0, head] if snapped else places[3])


def test_a_model_joint_goes_to_the_fighter_joint_nearest_it_not_the_first_that_reaches_it():
    # Leon: his torso joint (1, at 10.85) is near the fighter's spine joint's place and nearer its chest's. It is the
    # chest's: given to the spine joint, the spine bone would carry a sliver of torso up to the chest's place
    rig, frig, fch, mch = _spine([[0, 0, 9], [0, 0, 10.85], [0, 0, 12.2], [1, 0, 11.5], [-1, 0, 11.5]],
                                 [(9.5, 0), (10.2, 0), (11, 1), (12, 1), (13, 2), (15, 2)], [-1, 0, 1, 1, 1])
    owner, embedded, rest = {}, {}, {}
    cuts = correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert owner == {0: 0, 1: 2, 2: 4}
    assert np.allclose(embedded[2][0], [0, 0, 10.85]) and 9.5 < embedded[1][0][2] < 10.5
    assert [(mb, [fb for _, fb in c]) for mb, _, c in cuts] == [(0, [1])]


def test_a_model_joint_at_the_hips_is_the_hips():
    # Gardevoir: her spine bone (1) starts at her hip joint. The fighter's spine joint is not on it: the fighter's hips
    # would have no length inside her, a gap theirs between the pieces either side
    rig, frig, fch, mch = _spine([[0, 0, 9], [0, 0, 9], [0, 0, 11], [0, 0, 12.2], [1, 0, 11.5], [-1, 0, 11.5]],
                                 [(9.5, 1), (10.5, 1), (11.5, 2), (13, 3), (15, 3)], [-1, 0, 1, 2, 2, 2])
    owner, embedded, rest = {}, {}, {}
    correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert not correspond._no_length(embedded[0]) and embedded[1][0][2] > 9.5
    assert owner[0] == 0 and owner[1] == 0


def test_a_model_joint_at_the_hips_is_the_hips_where_the_chain_comes_back_down_to_it():
    # Gardevoir: her pelvis (0) and spine (2) hang at her hips from her root (1), above and behind them. The chain
    # climbs to the root and comes back down; along it, her spine's joint lies near the fighter spine joint's place
    rig, frig, fch, mch = _spine(
        [[0, 0, 9], [0, .5, 10], [0, 0, 9], [0, 0, 11], [0, 0, 12.2], [1, 0, 11.5], [-1, 0, 11.5]],
        [(8.5, 0), (9.5, 2), (10.5, 2), (11.5, 3), (13, 4), (15, 4)], [1, -1, 1, 2, 3, 3, 3])
    owner, embedded, rest = {}, {}, {}
    cuts = correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert not correspond._no_length(embedded[0]) and embedded[1][0][2] > 9.5
    assert owner == {0: 0, 1: 0, 2: 0, 3: 2, 4: 4}
    assert [(mb, [fb for _, fb in c]) for mb, _, c in cuts] == [(2, [1])]


def test_a_chain_bone_carrying_no_mesh_is_no_part_the_one_before_it_reaches_over_it():
    # Spider-Man: two neck bones (2, 3) carry nothing, his neck skinned to his chest (1) and head (4). The fighter's
    # chest and neck joints are those neck bones' joints; they cut the model's chest, so its neck rides the fighter's
    rig, frig, fch, mch = _spine(
        [[0, 0, 9], [0, 0, 10], [0, 0, 11], [0, 0, 11.6], [0, 0, 12.2], [1, 0, 11.5], [-1, 0, 11.5]],
        [(9.5, 0), (10.5, 1), (11.2, 1), (11.8, 1), (12.5, 4), (13, 4)], [-1, 0, 1, 2, 3, 1, 1])
    owner, embedded, rest = {}, {}, {}
    cuts = correspond._match_spine(rig, frig, fch, mch, owner, embedded, rest, [])
    assert [(mb, [fb for _, fb in c]) for mb, _, c in cuts] == [(1, [2, 3])]
    cut_at = {c[0]: c for c in cuts}
    assert correspond._apart_at(rig, frig, owner, lambda mb, p: correspond._carrier(owner, cut_at, mb, p)) == []


def _on_a_second_bone():
    """A Colosseum rip's leg: hip (0), knee (1), ankle (2), and at each of the last two joints a second bone (3, 4), which
    carries that joint's skin: the shin, and the foot, forward along the floor. The ankle's tail, as an importer makes
    one, runs back up the shin."""
    names, parents = ['hip', 'knee', 'ankle', 'shin', 'foot'], np.array([-1, 0, 1, 1, 2])
    heads = np.array([[0, 0, 2], [0, 0, 1], [0, 0, .2], [0, 0, 1], [0, 0, .2]], float)
    points = np.array([[0, 0, 1.8], [0, 0, 1.2], [0, 0, .8], [0, 0, .4], [0, -.2, 0], [0, -.5, 0], [0, -.8, 0]], float)
    weights = np.zeros((7, 5))
    weights[:2, 0], weights[2:4, 3], weights[4:, 4] = 1.0, 1.0, 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, .5], ['leg'], points, weights)
    return rig, SimpleNamespace(groups={1: [1, 3], 2: [2, 4]})


def test_a_legs_end_is_read_off_every_bone_at_its_last_joint():
    rig, body = _on_a_second_bone()
    assert correspond._chain(rig, [0, 1, 2], carriers=[2]).end[2] > .2         # the ankle alone: the tail, up the shin
    end = correspond._chain(rig, [0, 1, 2], carriers=correspond._last_joint(body, [2])).end
    assert end[2] < .2 and end[1] < -.5                                         # the foot, along the floor


def test_a_joint_carries_what_all_its_bones_do():
    rig, body = _on_a_second_bone()
    m = correspond.Chain([0, 1, 2], rig.heads[[0, 1, 2]], np.array([0, -.8, 0.]))
    assert correspond._parts(rig, m, {}).bones == [0]                           # read off the first bones, one part
    assert correspond._parts(rig, m, body.groups).bones == [0, 1, 2]


def test_a_trunk_bone_is_no_part_of_a_limb_reaching_through_it():
    rig, body = _on_a_second_bone()
    m = correspond.Chain([0, 1, 2], rig.heads[[0, 1, 2]], np.array([0, -.8, 0.]))
    assert correspond._parts(rig, m, body.groups, [0]).bones == [1, 2]          # Kirby's legs start inside his body


def test_the_model_hips_ride_the_fighter_hips_where_its_spine_starts_there_too():
    # Falco's spine bone starts at his hip joint: the model's hips bone (0) is still the fighter's hips
    rig, frig, fch, mch = _spine([[0, 0, 9], [0, 0, 10.5], [0, 0, 11.5], [0, 0, 12.2], [1, 0, 11.5], [-1, 0, 11.5]],
                                 [(9.5, 0), (10.8, 1), (11.8, 2), (13, 3)], [-1, 0, 1, 2, 1, 1],
                                 f_heads=[[0, 0, 8], [0, 0, 8], [0, 0, 10], [0, 0, 11.5], [0, 0, 12], [1, 0, 11],
                                          [-1, 0, 11]])
    owner = {}
    correspond._match_spine(rig, frig, fch, mch, owner, {}, {}, [])
    assert owner[0] == 0


def test_bones_meet_where_their_skin_does_not_where_a_flat_rig_hangs_them():
    # a flat rig: the upper (1) and lower (2) limb both hang off the root (0), but its surface runs root, upper, lower
    rig = _pieces([0.5, 1.5, 2.5])
    rig.parents = np.array([-1, 0, 0])
    rig.edges = np.array([[0, 1], [1, 2]])
    owner = {0: 0, 1: 1, 2: 2}
    fighter = _line(3, [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb]) == []
    rig.edges = np.array([[0, 1], [0, 2]])          # its surface runs from the root to each: the lower one parts
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb]) == [('piece2', 'piece0', 'JOBJ_2',
                                                                                    'JOBJ_0')]


def test_a_seam_the_model_spans_across_a_bone_of_its_own_parts_as_its_own_would():
    # a chest (0), a clavicle carrying no skin (1) and an upper arm (2), the chest's skin running on to the arm's, as
    # the game's own costumes are built: on the fighter's chest, clavicle and upper arm, no seam parts
    fighter = _line(3, [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    rig = _pieces([0.5, 2.5], bones=[0, 2])
    rig.edges = np.array([[0, 1]])
    owner = {0: 0, 1: 1, 2: 2}
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb]) == []
    owner = {0: 0, 1: 0, 2: 2}                      # the fighter's clavicle carries none of it: the seam parts there
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb]) == [('piece2', 'piece0', 'JOBJ_2',
                                                                                    'JOBJ_0')]
    # no clavicle bone in the model (Kritter's arm off his chest): the fighter's carries nothing, and the seam parts
    rig = _pieces([0.5, 1.5])
    rig.edges = np.array([[0, 1]])
    owner = {0: 0, 1: 2}
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb]) == [('piece1', 'piece0', 'JOBJ_2',
                                                                                    'JOBJ_0')]


def test_a_prop_in_a_weapon_slot_parts_from_the_hand_by_design():
    rig = _pieces([0.5, 1.5])
    fighter = _line(3, [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    owner = {0: 0, 1: 2}
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb]) != []
    assert correspond._apart_at(rig, fighter, owner, lambda mb, p: owner[mb], placed={2: None}) == []


def _two_hands():
    """A trunk (0) and two arms out along x, each a shoulder and a hand: off the left hand (+x) a sword bound to a bone
    of its own, 3 long and thin (blade, 3); off the right a shield, flat, 4 by 3 (buckler, 6), longer than the sword.
    Its body as body.read gives it, and a fighter with a sword bone and a shield bone under its hands."""
    names = ['trunk', 'l_upper', 'l_hand', 'blade', 'r_upper', 'r_hand', 'buckler']
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    heads = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.2, 0, 0], [-1, 0, 0], [-2, 0, 0], [-2.2, 0, 0]], float)
    pieces = [[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1.5, 0, 0]], [[2, -0.2, 0], [2.4, 0.2, 0]],
              [[2.2, 0, 0], [2.2, 0, 3], [2.25, 0, 3]], [[-1, 0, 0], [-1.5, 0, 0]], [[-2, -0.2, 0], [-2.4, 0.2, 0]],
              [[-2.2, -2, -1.5], [-2.2, 2, -1.5], [-2.2, 2, 1.5], [-2.2, -2, 1.5]]]
    points = np.array([q for p in pieces for q in p], float)
    weights = np.zeros((len(points), len(names)))
    weights[np.arange(len(points)), [b for b, p in enumerate(pieces) for _ in p]] = 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, 0.1], ['body'], points, weights)
    body = SimpleNamespace(arms={'left': [1, 2], 'right': [4, 5]}, legs={}, root=0, left=np.array([1., 0, 0]))
    fighter = SimpleNamespace(names=['JOBJ_0', 'L_hand', 'sword_bone', 'R_hand', 'shield_bone'],
                              heads=np.zeros((5, 3)), tails=np.zeros((5, 3)) + [0, 0, 1])
    return rig, body, fighter


def _slots(sword_hand='left_hand', shield_hand='right_hand'):
    side = {'left_hand': 'L_hand', 'right_hand': 'R_hand'}
    return [dict(part='sword', kind='sword', swung=True, hand=sword_hand, hand_bone=side[sword_hand],
                 bone='sword_bone', grip=[5, 0, 8], tip_direction=[0, -1, 0], flat=[1, 0, 0], length=6.0),
            dict(part='shield', kind='shield', swung=False, hand=shield_hand, hand_bone=side[shield_hand],
                 bone='shield_bone', grip=[-5, 0, 8], tip_direction=[0, 1, 0], flat=[1, 0, 0], length=5.0)]


def _props(slots, kinds):
    rig, body, fighter = _two_hands()
    owner, placed, replaced, notes = {}, {}, [], []
    hands = {1: (np.array([2., 0, 0]), np.array([2.5, 0, 0])), 3: (np.array([-2., 0, 0]), np.array([-2.5, 0, 0]))}
    embedded, rest = dict(hands), {b: (h + [0, 0, 8], t + [0, 0, 8]) for b, (h, t) in hands.items()}
    used = correspond._match_props(rig, body, fighter, {}, slots, owner, embedded, rest, placed, replaced, notes,
                                   kinds)
    return used, owner, embedded, rest, placed, replaced


def test_a_prop_goes_to_the_slot_of_its_kind():
    # Link's sword and shield: the shield, the longer, is no sword; each is fitted to the vanilla one, and both
    # vanilla weapons go
    used, owner, embedded, rest, placed, replaced = _props(_slots(), {'blade': 'sword', 'buckler': 'shield'})
    assert used == {2, 4} and [(r['part'], r['item']) for r in replaced] == [('sword', 'blade'), ('shield', 'buckler')]
    assert 2 in placed and 4 in placed
    assert owner[3] == 2 and owner[6] == 4


def test_a_slot_takes_no_prop_of_another_kind():
    used, owner, _, _, _, replaced = _props(_slots(), {'buckler': 'shield'})
    assert [(r['part'], r['item']) for r in replaced] == [('shield', 'buckler')] and 3 not in owner


@pytest.mark.parametrize('shield_hand', ['right_hand', 'left_hand'])
def test_a_shield_from_either_hand_is_fitted_into_the_vanilla_shields_place_at_its_own_size(shield_hand):
    # the buckler, held in the right hand, stands on its edge (face out along x, 4 long front to back, 3 tall); the
    # vanilla shield lies face up, 5 long front to back, its middle at (-5, 2.5, 8) on the right or (5, 2.5, 8) on the
    # left (a right-handed Link, u-link, on Link). From either hand it is turned the least way onto the vanilla's
    # box, a quarter turn about its long side, and keeps its size; it is never mirrored
    sword_hand = 'left_hand' if shield_hand == 'right_hand' else 'right_hand'
    slots = _slots(sword_hand, shield_hand)
    if shield_hand == 'left_hand':
        slots[1]['grip'] = [5, 0, 8]
    used, owner, embedded, rest, placed, replaced = _props(slots, {'blade': 'sword', 'buckler': 'shield'})
    assert sorted((r['part'], r['item']) for r in replaced) == [('shield', 'buckler'), ('sword', 'blade')]
    assert owner[3] == 2 and owner[6] == 4
    m = placed[4]
    assert np.isclose(np.linalg.det(m[:, :3]), 1) and np.isclose(np.trace(m[:, :3]), 1)        # a quarter turn
    rig, _, _ = _two_hands()
    moved = rig.points[rig.weights[:, 6] > 0.5] @ m[:, :3].T + m[:, 3]
    x = slots[1]['grip'][0]
    assert np.allclose(np.sort(moved, axis=0), np.sort([[x - 1.5, 0.5, 8], [x + 1.5, 4.5, 8], [x - 1.5, 4.5, 8],
                                                        [x + 1.5, 0.5, 8]], axis=0))
    assert np.allclose(m[:, :3] @ embedded[4][0] + m[:, 3], rest[4][0])


def test_a_shield_in_the_slots_own_hand_comes_before_one_in_the_other():
    rig, body, fighter = _two_hands()
    rig.names.append('targe')                                   # a second shield, in the left hand
    rig.parents = np.append(rig.parents, 2)
    rig.heads = np.vstack([rig.heads, [2.2, 0, 0]])
    rig.tails = np.vstack([rig.tails, [2.2, 0, 0.1]])
    rig.points = np.vstack([rig.points, [[2.2, -1, -1], [2.2, 1, -1], [2.2, 1, 1], [2.2, -1, 1]]])
    rig.weights = np.pad(rig.weights, ((0, 4), (0, 1)))
    rig.weights[-4:, -1] = 1.0
    owner, placed, replaced, notes, whole = {}, {}, [], [], {}
    hands = {1: (np.array([2., 0, 0]), np.array([2.5, 0, 0])), 3: (np.array([-2., 0, 0]), np.array([-2.5, 0, 0]))}
    embedded, rest = dict(hands), {b: (h + [0, 0, 8], t + [0, 0, 8]) for b, (h, t) in hands.items()}
    correspond._match_props(rig, body, fighter, {}, _slots()[1:], owner, embedded, rest, placed, replaced, notes,
                            {'buckler': 'shield', 'targe': 'shield'}, whole)
    # the shield slot is in the right hand: the buckler there takes it, as the model holds it; the targe stays in
    # its hand. The buckler's vertices ride the slot alone
    assert [r['item'] for r in replaced] == ['buckler'] and 4 in placed and 7 not in owner
    assert set(whole) == {4} and np.array_equal(whole[4], rig.weights[:, 6] > 0.5)


def test_with_one_kind_of_slot_the_longest_held_prop_is_the_weapon():
    used, owner, _, _, placed, replaced = _props(_slots()[:1], None)
    assert [(r['part'], r['item']) for r in replaced] == [('sword', 'buckler')] and 2 in placed


def _rod(slots, kinds):
    """_two_hands with a rod of its own off no bone of the body, down beside the left leg (Toon Link's): a prop carried
    outside the body. The owners _match_props gives, the hands' item joints the fighter's hand bones."""
    rig, body, fighter = _two_hands()
    rig.names.append('rod')
    rig.parents = np.append(rig.parents, -1)
    rig.heads = np.vstack([rig.heads, [0.3, 0, -1]])
    rig.tails = np.vstack([rig.tails, [0.3, 0, -0.9]])
    rig.points = np.vstack([rig.points, [[0.3, 0, -1], [0.3, 0, -5], [0.32, 0, -5]]])
    rig.weights = np.pad(rig.weights, ((0, 3), (0, 1)))
    rig.weights[-3:, -1] = 1.0
    owner, placed, replaced, notes = {}, {}, [], []
    hands = {1: (np.array([2., 0, 0]), np.array([2.5, 0, 0])), 3: (np.array([-2., 0, 0]), np.array([-2.5, 0, 0]))}
    embedded, rest = dict(hands), {b: (h + [0, 0, 8], t + [0, 0, 8]) for b, (h, t) in hands.items()}
    correspond._match_props(rig, body, fighter, {'left_hand': 'L_hand', 'right_hand': 'R_hand'}, slots, owner,
                            embedded, rest, placed, replaced, notes, kinds)
    return owner


def test_a_prop_named_none_rides_where_it_lies_and_one_never_asked_goes_to_the_hand():
    # Link's slots, the props asked: the rod is named neither, and no slot nor hand takes it (the rest of the tree
    # gives it its place on the model)
    assert 7 not in _rod(_slots(), {'blade': 'sword', 'buckler': 'shield'})
    # one kind of slot, nothing asked: a prop outside the body no slot takes is a weapon its game puts in the hand
    assert _rod(_slots()[:1], None)[7] == 1


def test_a_prop_from_outside_the_body_goes_on_the_item_joint_not_where_it_lay_beside_the_hand():
    # the Barbarian (Clash Royale): his sword's bones hang off the root and it stands on the floor beside him, held by
    # no hand; the game puts it in his fist by animation. Donkey Kong carries what his right hand holds on an item
    # joint under that hand, an end. Placed through the hand (_end_parents), the sword kept its distance from the
    # model's hand and floated beside Donkey Kong's fist. Its grip goes on the item joint, its own way up and size
    rig, body, _ = _two_hands()
    rig.names.append('rod')
    rig.parents = np.append(rig.parents, -1)
    rig.heads = np.vstack([rig.heads, [3.5, 0, -1]])
    rig.tails = np.vstack([rig.tails, [3.5, 0, -0.9]])
    rod = np.array([[3.5, 0, -1.5], [3.5, 0, 2], [3.52, 0, 2]])
    rig.points = np.vstack([rig.points, rod])
    rig.weights = np.pad(rig.weights, ((0, 3), (0, 1)))
    rig.weights[-3:, -1] = 1.0
    fighter = SimpleNamespace(names=['JOBJ_0', 'L_hand', 'L_item'], parents=np.array([-1, 0, 1]),
                              heads=np.array([[0, 0, 8], [3, 0, 8], [3.3, 0, 7.9]]),
                              tails=np.array([[0, 0, 9], [3.5, 0, 8], [3.3, 0.2, 7.9]]))
    owner, placed, notes = {}, {}, []
    embedded = {0: (np.zeros(3), np.array([0, 0, 1.])), 1: (np.array([2., 0, 0]), np.array([2.5, 0, 0]))}
    rest = {0: (fighter.heads[0], fighter.tails[0]), 1: (fighter.heads[1], fighter.tails[1])}
    used = correspond._match_props(rig, body, fighter, {'left_hand': 'L_item'}, (), owner, embedded, rest, placed, [],
                                   notes)
    assert used == {2} and owner[7] == 2 and notes == ['prop rod -> L_item (left_hand)']
    ends = {1} | used
    match = correspond.Match(owner, embedded, rest, ends=ends, placed=placed,
                             end_parents=correspond._end_parents(fighter.parents, embedded, ends))
    moves = repose.transforms(match, [0, 1, 2])
    assert np.allclose(moves[2](np.asarray([rig.heads[7]]))[0], fighter.heads[2])     # the grip on the item joint
    assert np.allclose(moves[2](rod) - rod, fighter.heads[2] - rig.heads[7])          # its own way up and size


def _hand(heads, parents, names):
    """A rig of one vertex per bone, just past its head, riding it alone."""
    heads = np.array(heads, float)
    points = heads + [0.1, 0, 0]
    return Rig('hand', names, np.array(parents), heads, heads + [0.3, 0, 0], ['hand'], points, np.eye(len(heads)))


def test_the_models_fingers_are_the_fighters_by_where_they_leave_the_hand_and_point():
    # Mario's hand: an index and a pinky across the back of the hand, a thumb below and in front, two joints each. A
    # model's: a palm bone (named digits) splitting into an index of three bones and a pinky of two, and a thumb. The
    # fighter's second joint lies at the same share along the model's finger: at its third bone, on the index
    frig = _hand([[0, 0, 8], [1, 0, 8], [2, -.3, 8], [2.5, -.3, 8], [2, .3, 8], [2.5, .3, 8], [1.5, -.6, 7.8],
                  [1.9, -.8, 7.8]], [-1, 0, 1, 2, 1, 4, 1, 6], [f'JOBJ_{i}' for i in range(8)])
    description = {'chains': [{'kind': 'finger', 'bones': [3, 4]}, {'kind': 'finger', 'bones': [5, 6]},
                              {'kind': 'finger', 'bones': [7, 8]}, {'kind': 'tail', 'bones': [1, 2]}]}
    rig = _hand([[0, 0, 8], [1, 0, 8], [1.4, 0, 8], [1.8, -.25, 8], [2.1, -.25, 8], [2.4, -.25, 8], [1.8, .25, 8],
                 [2.1, .25, 8], [1.3, -.5, 7.8], [1.6, -.7, 7.8]], [-1, 0, 1, 2, 3, 4, 2, 6, 1, 8],
                ['forearm', 'hand', 'palm', 'index1', 'index2', 'index3', 'pinky1', 'pinky2', 'thumb1', 'thumb2'])
    arm = lambda r: {'left arm': correspond.Chain([0, 1], r.heads[[0, 1]], np.array([2.6, 0, 8]))}
    owner, embedded, rest, notes = {}, {}, {}, []
    tips = correspond._match_fingers(rig, frig, description, arm(frig), arm(rig), owner, embedded, rest, notes,
                                     {'palm', 'thumb1'})
    assert {rig.names[b]: frig.names[f] for b, f in owner.items()} == {
        'index1': 'JOBJ_2', 'index2': 'JOBJ_2', 'index3': 'JOBJ_3', 'pinky1': 'JOBJ_4', 'pinky2': 'JOBJ_5',
        'thumb1': 'JOBJ_6', 'thumb2': 'JOBJ_7'}                   # the palm rides the hand (_rest_of_tree)
    assert tips == {3, 5, 7}                                       # the last joints keep their own size
    assert np.allclose(embedded[2][0], rig.heads[3]) and np.allclose(rest[2][0], frig.heads[2])
    assert correspond._match_fingers(rig, frig, description, arm(frig), arm(rig), {}, {}, {}, [], set()) == set()


def test_a_hand_is_read_in_its_own_frame_however_the_arm_holds_it():
    # Mario's hand: index, middle and pinky along it, the thumb below and in front. The model's, turned about its arm
    # a quarter, a half and three quarters of the way round (a T pose's palm down against Mario's thumb up), its pinky
    # begun at its palm bone by the wrist: the digit ending nearest the wrist is the thumb, the rest go across from it
    frig = _hand([[0, 0, 8], [1, 0, 8], [2, -.3, 8], [2.5, -.3, 8], [2.05, 0, 8], [2.55, 0, 8], [2, .3, 8], [2.5, .3, 8],
                  [1.5, -.6, 7.8], [1.9, -.8, 7.8]], [-1, 0, 1, 2, 1, 4, 1, 6, 1, 8], [f'JOBJ_{i}' for i in range(10)])
    description = {'chains': [{'kind': 'finger', 'bones': [3 + 2 * i, 4 + 2 * i]} for i in range(4)]}
    hand = [[.4, -.25, 0], [.7, -.25, 0], [1, -.25, 0], [.45, 0, 0], [.8, 0, 0], [1.1, 0, 0], [.1, .1, 0],
            [.8, .25, 0], [1.1, .25, 0], [.3, -.5, -.2], [.6, -.7, -.2]]
    names = ['forearm', 'hand', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky0', 'pinky1',
             'pinky2', 'thumb1', 'thumb2']
    arm = lambda r: {'left arm': correspond.Chain([0, 1], r.heads[[0, 1]], np.array([2.6, 0, 8]))}
    for turn in (np.pi / 2, np.pi, -np.pi / 2):
        c, s = np.cos(turn), np.sin(turn)
        heads = [[0, 0, 8], [1, 0, 8]] + [[1 + x, y * c - z * s, 8 + y * s + z * c] for x, y, z in hand]
        rig = _hand(heads, [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11], names)
        owner = {}
        correspond._match_fingers(rig, frig, description, arm(frig), arm(rig), owner, {}, {}, [],
                                  {'index1', 'middle1', 'pinky0', 'thumb1'})
        assert {n: frig.names[owner[names.index(n)]] for n in ('index1', 'middle1', 'pinky0', 'thumb1')} == {
            'index1': 'JOBJ_2', 'middle1': 'JOBJ_4', 'pinky0': 'JOBJ_6', 'thumb1': 'JOBJ_8'}
    # a mitten: a thumb and one digit along the hand. Thumb to thumb, though it sits nearer the middle of the hand's
    # width than Mario's; the mitten to the finger along the middle
    mitten = _hand([[0, 0, 8], [1, 0, 8], [1.3, -.1, 7.95], [1.6, -.15, 7.95], [1.4, 0, 8], [1.9, 0, 8]],
                   [-1, 0, 1, 2, 1, 4], ['forearm', 'hand', 'thumb1', 'thumb2', 'mitten1', 'mitten2'])
    owner = {}
    correspond._match_fingers(mitten, frig, description, arm(frig), arm(mitten), owner, {}, {}, [],
                              {'thumb1', 'mitten1'})
    assert {mitten.names[b]: frig.names[f] for b, f in owner.items() if b in (2, 4)} == {
        'thumb1': 'JOBJ_8', 'mitten1': 'JOBJ_4'}
    one = _hand(heads[:5], [-1, 0, 1, 2, 3], names[:5])      # one digit: no thumb apart from it, no frame; it rides
    assert correspond._match_fingers(one, frig, description, arm(frig), arm(one), {}, {}, {}, [], {'index1'}) == set()


def test_fingers_pair_one_to_one_as_many_as_the_fewer_side_has():
    assert correspond._assign(np.array([[0, 1.], [1, 0], [.5, .5]])) == [(0, 0), (1, 1)]
    assert correspond._assign(np.array([[1, 0, .5], [0, 1, .5]])) == [(0, 1), (1, 0)]


def test_an_arm_of_three_joints_starts_at_its_shoulder_however_narrow_the_chest():
    """A carnival skeleton's shoulders sit a fifth of the way out to its hands held out; read as a clavicle, the
    fighter's shoulder rode its elbow and the fighter's elbow bent its forearm in the middle. A clavicle needs an upper
    arm and a forearm after it, before the hand."""
    heads = np.array([[0, 0, 1.5], [.167, 0, 1.48], [.516, 0, 1.49], [.797, 0, 1.49],      # chest, shoulder, elbow, hand
                      [.03, 0, 1.5], [.167, 0, 1.48], [.516, 0, 1.49], [.797, 0, 1.49]])   # a clavicle before them
    rig, body = SimpleNamespace(heads=heads), SimpleNamespace(chest=0, left=np.array([1., 0, 0]))
    assert correspond._shoulder_joint(rig, body, [1, 2, 3]) == 0
    assert correspond._shoulder_joint(rig, body, [4, 5, 6, 7]) == 1


def test_a_bone_at_a_cut_bones_joint_is_cut_with_it():
    """A carnival skeleton's ribcage is skinned to a torso bone whose joint is its pelvis's: body reading makes them
    one joint (body.groups), but only the pelvis's own skin was cut at the fighter's chest joint, so the ribcage rode
    the fighter's hips whole and the arms came away from it as the chest leaned."""
    from rigswap.core import repose
    # pelvis (0) and torso (1) at z 0, neck (2) at z 2; the pelvis carries the hips (z 0.1), the torso the ribcage (z
    # 0.5 and 1.5); the fighter's chest joint (its bone 11) falls inside, at z 1
    weights = np.array([[1., 0, 0], [0, 1, 0], [0, 1, 0]])
    rig = SimpleNamespace(points=np.array([[0, 0, .1], [0, 0, .5], [0, 0, 1.5]]), weights=weights,
                          carried=lambda: weights.sum(axis=0))
    spine = correspond.Chain([0, 2], np.array([[0, 0, 0.], [0, 0, 2]]), np.array([0, 0, 3.]))
    owner, splits = {0: 10, 2: 12}, [(0, spine, [(1 / 3, 11)])]
    added = correspond._at_joint(rig, SimpleNamespace(groups={0: [0, 1], 2: [2]}), owner, splits)
    assert owner[1] == 10 and [s[0] for s in added] == [1] and added[0][1].bones == [1, 2]
    w = repose.split(rig, repose.fighter_weights(rig, owner, [10, 11, 12]), [10, 11, 12], splits + added, owner)
    assert w[1, 0] > .9 and w[2, 1] > .9            # below the chest joint the ribcage rides the hips, above it the chest


def test_a_chest_with_no_clavicle_widens_to_the_fighters_shoulders():
    """A carnival skeleton's upper arms are tucked into its ribcage's corners, 2 from its middle, with no clavicle:
    they ride Ganondorf's shoulder joints at 2.9, so his clavicles' length opened between them and a ribcage still 1.8
    wide, and the fighter pick read every fighter with clavicles as coming apart. The chest above the hips widens
    until its surface is as wide as the shoulders; a chest already that wide, or a model with a clavicle of its own,
    keeps its width."""
    from dataclasses import replace
    from rigswap.core import repose
    spine = correspond.Chain([4, 17, 18, 54], np.zeros((4, 3)), np.zeros(3), anchors=[0, 2, 3])
    frig = SimpleNamespace(names={4: 'hips', 17: 'spine', 18: 'chest', 54: 'head'})
    bare = {'left arm': np.array([2.9, 0, 17.8]), 'right arm': np.array([-2.9, 0, 17.8])}
    ends = {17: ([0, 0, 14], [0, 0, 15.5]), 18: ([0, 0, 15.5], [0, 0, 18])}
    match = correspond.Match({}, ends, ends, trunk=[17, 18])
    owner = {0: 18, 1: 17}

    def chest(half):        # a ribcage ``half`` wide each side on model bone 0, the chest; a spine on bone 1 below
        points = np.array([[half, 0, 17.5], [-half, 0, 17.5], [half, 0, 16], [-half, 0, 16], [0, 0, 14.5]])
        return SimpleNamespace(points=points, weights=np.array([[1, 0]] * 4 + [[0, 1]], float))
    widen = correspond._widen(chest(2.), frig, {'spine': spine}, owner, bare, match, [])
    assert set(widen) == {17, 18} and np.isclose(widen[18][0], 1.45) and np.isclose(widen[18][1], 0)
    # the ribcage's corner beside the model's shoulder lands beside the fighter's; the spine on the midline stays
    corner, middle = repose.transforms(replace(match, widen=widen), [18])[18](np.array([[1.8, 0, 17.5], [0, .5, 16]]))
    assert np.isclose(corner[0], 2.61) and np.allclose(middle, [0, .5, 16])
    assert correspond._widen(chest(3.), frig, {'spine': spine}, owner, bare, match, []) == {}
    assert correspond._widen(chest(2.), frig, {'spine': spine}, owner, {'left arm': bare['left arm']}, match, []) == {}


def test_a_tail_the_fighters_chain_cannot_carry_swings_on_joints_of_its_own_from_where_the_fighters_tail_hangs():
    """Diddy on Fox (size-3, the owner: "diddy tail on fox looks really strange"): his tail, 15.5 long on Fox's 11.1,
    kept its length, so Fox's joints stopped short and 41 % of it rode Fox's last tail bone as a rod. A tail past the
    fighter's chain by more than that chain's last bone is returned with the hair and cloth, to swing on joints of
    its own at its own length; so is a tail on a fighter with no tail. It hangs where the fighter's tail hangs: all
    of it rides the fighter's first tail bone and it hangs from the fighter bone carrying its own root (limbs-1: left
    on Fox's waist it sat above his tail and tipped up with his back; the owner: "higher up on his body than fox").
    A tail the chain can carry, and a cape of any length, go on the fighter's as before."""
    def run(kind, long, fighter='tail'):
        rig, frig, description, fch, mch, owner = _caped()
        description['chains'][0]['kind'] = fighter
        if long:        # the model's chain 13 long (12 and a tip of 1) on the fighter's 7 (5, and 2 to its mesh's end)
            rig.heads[6], rig.tails[6], rig.points[6] = [0, 1, 2], [0, 1, 1], [0, 1, 1]
        named = {'cape': {'kind': kind, 'held_out': False}, 'cape2': {'kind': kind, 'held_out': False}}
        notes, embedded, rest = [], {}, {}
        loose, swung, _ = correspond._match_extra(rig, frig, description, fch, mch, owner, embedded, rest, notes,
                                                  named=named)
        return loose, swung, owner, notes, embedded, rest
    loose, swung, owner, notes, embedded, rest = run('tail', long=True)     # 6 past the fighter's 7; its last bone is 2
    assert swung == set() and owner[5] == 8 and owner[6] == 8               # all of it on the fighter's first bone
    (attach, chain, carries), = loose
    assert (attach, chain.bones, carries, chain.kind) == (5, [5, 6], [5, 6], 'tail')
    assert np.allclose(embedded[8], ([0, 1, 14], [0, 1, 9])) and np.allclose(rest[8], ([0, 1, 14], [0, 1, 9]))
    assert 9 not in embedded and any('on joints of its own from JOBJ_8' in n for n in notes)
    loose, swung, owner, *_ = run('tail', long=False)               # 6 long on the fighter's 7: on the fighter's
    assert loose == [] and swung == {8, 9} and {owner[5], owner[6]} <= {8, 9}
    loose, swung, owner, *_ = run('tail', long=False, fighter='cape')   # no tail of the fighter's to go on
    assert swung == set() and [(a, c.bones) for a, c, _ in loose] == [(1, [5, 6])] and 5 not in owner
    loose, swung, owner, *_ = run('cloth', long=True, fighter='cape')   # a long cape stays on the fighter's cape
    assert loose == [] and swung == {8, 9}


def test_a_tail_on_its_own_joints_starts_at_the_fighters_tails_level_and_its_own_depth():
    """Fox's tail leaves his hips 0.79 behind and 0.65 below their joint; Diddy's, 1.7 behind and 0.16 below. The
    fighter's first tail bone is laid inside the model so that the model's root comes to rest that much farther out
    along it: at the fighter's level, as far behind the hips as the model has it."""
    f = correspond.Chain([0, 1], np.array([[0, 1, 9.], [0, 4, 9.]]), np.array([0, 6, 9.]), [0, 1, None])
    m = correspond.Chain([0, 1], np.array([[0, 3, 10.], [0, 6, 10.]]), np.array([0, 20, 10.]), [0, 1, None])
    hips_m, hips_f = np.array([0, 0, 10.2]), np.array([0, 0, 9.6])
    (head, tail), (rest_head, rest_tail) = correspond._rooted(m, f, hips_m, hips_f)
    assert np.allclose(head, [0, 1, 10]) and np.allclose(tail, [0, 4, 10])              # 2 farther in than its root
    assert np.allclose(rest_head, [0, 1, 9]) and np.allclose(rest_tail, [0, 4, 9])      # one length: only moved
    # its root, 2 along that bone, comes to rest at [0, 3, 9]: Fox's level, its own 3 behind the hips
    near = correspond.Chain([0, 1], np.array([[0, .5, 10.], [0, 6, 10.]]), np.array([0, 20, 10.]), [0, 1, None])
    assert np.allclose(correspond._rooted(near, f, hips_m, hips_f)[0][0], [0, .5, 10])  # nearer the hips: at the joint


def test_a_chains_length_runs_along_its_joints_to_its_end():
    f = correspond.Chain([0, 1], np.array([[0, 0, 10.], [0, 0, 6.]]), np.array([0, 0, 2.]), [0, 1, None])
    assert correspond._long(f) == 8 and f.fractions()[0].tolist() == [0, .5, 1]
