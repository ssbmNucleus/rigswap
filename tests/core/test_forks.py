"""The fork ask's code side: the forks it lists, how the answers are counted and mirrored, and the limbs body reading
follows from them (the pictures need Blender)."""
from types import SimpleNamespace

import numpy as np

from rigswap.core import body, correspond, forks
from rigswap.core.rig import Rig
from tests.core.test_body import _bones, _figure, _names, _roles

UP = np.array([0., 0., 1.])


def _sleeved():
    """The figure with a sleeve off each forearm, heavier than the hand, and two fingers off each hand."""
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'sleeve.{side}', f'fore.{side}', [sign * .7, 0, 1.3], 20),
                  (f'sleevetip.{side}', f'sleeve.{side}', [sign * .7, 0, .9], 20),
                  (f'finger1.{side}', f'hand.{side}', [sign * 1.1, .03, 1.5], 3),
                  (f'finger2.{side}', f'hand.{side}', [sign * 1.1, -.03, 1.5], 3)]
    return _figure(bones)


def _answer(*said):
    """An answer: per fork number, {colour: kind}."""
    return {'character': '', 'forks': [{'number': n, 'branches': [{'colour': c, 'what': '', 'kind': k}
                                                                  for c, k in kinds.items()]} for n, kinds in said]}


def _tree(kids):
    """A stand-in for body reading's joints, ``kids`` by bone."""
    parents = {k: b for b, ks in kids.items() for k in ks}
    return SimpleNamespace(kids=kids, parents=[parents.get(b, -1) for b in range(len(kids))])


def test_an_unanswered_arm_runs_into_what_outweighs_its_hand():
    rig = _sleeved()
    shape = body.read(rig, UP, (0, -1, 0), _roles(rig))
    assert _names(rig, shape.arms['left']) == ['arm.L', 'fore.L', 'sleeve.L', 'sleevetip.L']


def test_an_arm_goes_where_its_forks_are_answered_and_ends_at_its_hand():
    rig = _sleeved()
    goes = {'fore.L': 'hand.L', 'fore.R': 'hand.R', 'hand.L': None, 'hand.R': None}
    shape = body.read(rig, UP, (0, -1, 0), _roles(rig), goes)
    assert _names(rig, shape.arms['left']) == ['arm.L', 'fore.L', 'hand.L']
    assert _names(rig, shape.arms['right']) == ['arm.R', 'fore.R', 'hand.R']
    chains, _ = correspond.model_chains(rig, shape)
    assert _names(rig, chains['left arm'].bones) == ['arm.L', 'fore.L', 'hand.L']


def test_an_answer_naming_no_branch_of_the_fork_is_no_answer():
    rig = _sleeved()
    shape = body.read(rig, UP, (0, -1, 0), _roles(rig), {'fore.L': 'toe.L', 'fore.R': 'toe.R'})
    assert _names(rig, shape.arms['left'])[2] == 'sleeve.L'


def test_forks_are_listed_out_from_the_root_once_per_twin_pair():
    rig = _sleeved()
    mass, found, mirrored, roots = forks.candidates(rig, UP, _roles(rig))
    assert [rig.names[f['bone']].split('.')[0] for f in found] == ['fore', 'hand']
    side = rig.names[found[0]['bone']][-1]
    assert {rig.names[b] for b in mirrored} == {f'fore.{"R" if side == "L" else "L"}',
                                                f'hand.{"R" if side == "L" else "L"}'}
    assert [rig.names[b].split('.')[0] for b in found[0]['branches']] == ['sleeve', 'hand']    # heaviest first
    assert {rig.names[b] for b in roots} == {'arm.L', 'arm.R', 'thigh.L', 'thigh.R'}


def test_a_branch_is_the_kind_more_than_half_the_answers_give_it():
    found = [dict(bone=10, branches=[11, 12]), dict(bone=20, branches=[21, 22])]
    said = [_answer((1, {'red': 'carried', 'blue': 'limb'}), (2, {'red': 'digits'})),
            _answer((1, {'red': 'carried', 'blue': 'limb'}), (2, {'red': 'helper'})),
            _answer((1, {'red': 'limb', 'blue': 'cant_tell'}))]
    assert forks.tally(said, found) == {10: {11: 'carried', 12: 'limb'}, 20: {}}
    assert forks.tally([_answer((1, {'blue': 'digits'}))] * 2 + [None], found[:1]) == {10: {12: 'digits'}}
    assert forks.tally([_answer((1, {'red': 'cant_tell'}))] * 3, found[:1]) == {10: {}}


def test_a_twin_fork_takes_its_mirrors_answers(monkeypatch, tmp_path):
    rig = _sleeved()
    monkeypatch.setattr(forks, 'draw', lambda *a: [])
    _, found, _, _ = forks.candidates(rig, UP, _roles(rig))
    side = rig.names[found[0]['bone']][-1]
    twin = 'R' if side == 'L' else 'L'

    def ask(question, pictures, schema, default, route):     # the forearm's sleeve (red) and hand (blue), then fingers
        return _answer((1, {'red': 'carried', 'blue': 'limb'}), (2, {'red': 'digits', 'blue': 'digits'})), None
    goes, report = forks.read(rig, UP, (0, -1, 0), tmp_path, ask, _roles(rig))
    assert goes == {f'fore.{side}': f'hand.{side}', f'hand.{side}': None,
                    f'fore.{twin}': f'hand.{twin}', f'hand.{twin}': None}
    assert report['forks'][0]['branches'] == {'red': f'sleeve.{side}', 'blue': f'hand.{side}'}
    assert report['forks'][0]['kinds'] == {f'sleeve.{side}': 'carried', f'hand.{side}': 'limb'}


def test_the_hand_is_where_the_fingers_spread_whatever_the_way_to_it_is_called(monkeypatch, tmp_path):
    # the hand taken for a helper and the sleeve for the arm going on: the fingers spread from the hand all the same
    rig = _sleeved()
    monkeypatch.setattr(forks, 'draw', lambda *a: [])
    _, found, _, _ = forks.candidates(rig, UP, _roles(rig))
    side = rig.names[found[0]['bone']][-1]

    def ask(question, pictures, schema, default, route):
        return _answer((1, {'red': 'limb', 'blue': 'helper'}), (2, {'red': 'digits', 'blue': 'carried'})), None
    goes, _ = forks.read(rig, UP, (0, -1, 0), tmp_path, ask, _roles(rig))
    assert goes[f'fore.{side}'] == f'hand.{side}' and goes[f'hand.{side}'] is None


def test_fingers_spreading_one_knuckle_at_a_time_spread_from_the_first():
    # arm 0, forearm 1, hand 2; the thumb 3 off the hand, the palm 4 going on to the index 5 and the palm 6, which
    # carries the last two fingers 7 and 8
    mass = _tree({0: [1], 1: [2], 2: [3, 4], 3: [], 4: [5, 6], 5: [], 6: [7, 8], 7: [], 8: []})
    kinds = {2: {3: 'digits', 4: 'limb'}, 4: {5: 'digits', 6: 'limb'}, 6: {7: 'digits', 8: 'digits'}}
    assert forks.ways(mass, [0], kinds) == {2: None}
    # of two forks as near the root, the one with more digits
    mass = _tree({0: [1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [], 4: [], 5: [], 6: [], 7: []})
    kinds = {1: {3: 'digits', 4: 'carried'}, 2: {5: 'digits', 6: 'digits', 7: 'digits'}}
    assert forks.ways(mass, [0], kinds) == {0: 2, 2: None}


def test_a_limb_without_digits_goes_on_along_its_one_limb_branch():
    # thigh 0, shin 1 carrying a flame 2 and the hoof 3
    mass = _tree({0: [1], 1: [2, 3], 2: [], 3: []})
    assert forks.ways(mass, [0], {1: {2: 'carried', 3: 'limb'}}) == {1: 3}
    assert forks.ways(mass, [0], {1: {2: 'carried', 3: 'helper'}}) == {}
    # a foot in parts, heel 4 and front 5: the leg ends at it
    mass = _tree({0: [1], 1: [2, 3], 2: [], 3: [4, 5], 4: [], 5: []})
    assert forks.ways(mass, [0], {1: {2: 'carried', 3: 'limb'}, 3: {4: 'limb', 5: 'limb'}}) == {1: 3, 3: None}


def test_the_toe_is_not_sought_at_a_fork_the_leg_goes_on_through():
    # a panel off the knee reaching the floor ahead of the foot: the forward-most tip on the floor, unless the leg is
    # known to go on through the knee to its foot
    names = ['hip', 'thigh', 'knee', 'foot', 'panel']
    parents = np.array([-1, 0, 1, 2, 2])
    heads = np.array([[0, 0, 1], [0, 0, .9], [0, 0, .5], [0, -.02, .08], [0, -.1, .4]], float)
    pieces = [[[0, 0, .95]], [[0, 0, .7]], [[0, 0, .3]], [[0, -.05, 0], [0, .02, .02]], [[0, -.3, .2], [0, -.3, 0]]]
    points = np.array([q for p in pieces for q in p], float)
    weights = np.zeros((len(points), len(names)))
    weights[np.arange(len(points)), [b for b, p in enumerate(pieces) for _ in p]] = 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, -.05], ['leg'], points, weights)
    shape = SimpleNamespace(up=UP, front=np.array([0., -1., 0.]), forks={}, groups={})
    assert correspond._forward_toe(rig, shape, [0, 1, 2, 3]) == [0, 1, 2, 4]
    shape.forks = {2: 3}
    assert correspond._forward_toe(rig, shape, [0, 1, 2, 3]) == [0, 1, 2, 3]


def test_a_foot_whose_own_pad_is_lowest_has_no_toe():
    # Pidgeot's curled foot: the last bone's own pad is the lowest point, and its talons stop short of it
    names = ['hip', 'thigh', 'shin', 'foot', 'talon1', 'talon2']
    parents = np.array([-1, 0, 1, 2, 3, 3])
    heads = np.array([[0, 0, 1], [0, 0, .9], [0, 0, .5], [0, 0, .1], [0, -.05, .1], [0, .05, .1]], float)
    pieces = [[[0, 0, .95]], [[0, 0, .7]], [[0, 0, .3]], [[0, 0, 0]], [[0, -.1, .06]], [[0, .1, .05]]]
    points = np.array([q for p in pieces for q in p], float)
    weights = np.zeros((len(points), len(names)))
    weights[np.arange(len(points)), [b for b, p in enumerate(pieces) for _ in p]] = 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, -.05], ['leg'], points, weights)
    shape = SimpleNamespace(up=UP, front=np.array([0., -1., 0.]), forks={}, groups={})
    assert correspond._forward_toe(rig, shape, [0, 1, 2, 3]) == [0, 1, 2, 3]


def test_faces_take_the_value_most_of_their_corners_have():
    values = np.array([2, 2, 0, 1, 0, -1, 3, 3, 3, -1])
    first = np.array([0, 3, 6])                 # a triangle, a triangle, a quad
    assert forks._most(values, first).tolist() == [2, -1, 3]


def test_a_foot_joint_of_two_bones_branches_from_both():
    # the ankle is two bones at one place, the second carrying the heel and the toe
    names = ['hip', 'thigh', 'knee', 'ankle', 'ankle2', 'heel', 'toe']
    parents = np.array([-1, 0, 1, 2, 3, 4, 4])
    heads = np.array([[0, 0, 1], [0, 0, .9], [0, 0, .5], [0, 0, .08], [0, 0, .08], [0, .05, .04], [0, -.08, .03]],
                     float)
    pieces = [[[0, 0, .95]], [[0, 0, .7]], [[0, 0, .3]], [[0, 0, .1]], [[0, 0, .07]], [[0, .06, 0]], [[0, -.12, 0]]]
    points = np.array([q for p in pieces for q in p], float)
    weights = np.zeros((len(points), len(names)))
    weights[np.arange(len(points)), [b for b, p in enumerate(pieces) for _ in p]] = 1.0
    rig = Rig('test', names, parents, heads, heads + [0, 0, -.05], ['leg'], points, weights)
    shape = SimpleNamespace(up=UP, front=np.array([0., -1., 0.]), forks={3: None}, groups={3: [3, 4]})
    assert correspond._forward_toe(rig, shape, [0, 1, 2, 3]) == [0, 1, 2, 3, 4, 6]
