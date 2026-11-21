from types import SimpleNamespace

import numpy as np
import pytest

from rigswap.core import correspond
from rigswap.core.rig import Rig
from rigswap.core.run import scale


def model(hips=4.0, chest=8.0, head=10.0, top=12.0):
    """A grounded body with named roles, a head that turns at the neck, and two three-joint limbs per side."""
    roles = ['hips', 'chest', 'head']
    parents = [-1, 0, 1]
    heads = [[0, 0, hips], [0, 0, chest], [0, 0, head]]
    points = [[0, 0, hips], [0, 0, chest], [0, 0, head], [0, 0, top]]
    owners = [0, 1, 2, 2]
    arms, legs = {}, {}
    for side, sign in [('left', 1), ('right', -1)]:
        for kind, parts, joints, parent in [
            ('arm', ['upper_arm', 'forearm', 'hand'],
             [[sign, 0, chest], [3 * sign, 0, chest], [5 * sign, 0, chest]], 1),
            ('leg', ['thigh', 'shin', 'foot'],
             [[sign, 0, hips], [sign, 0, hips / 2], [sign, 0, 1]], 0),
        ]:
            chain = []
            for part, joint in zip(parts, joints):
                bone = len(roles)
                roles.append(f'{side}_{part}')
                parents.append(parent)
                heads.append(joint)
                points.append(joint)
                owners.append(bone)
                chain.append(bone)
                parent = bone
            (arms if kind == 'arm' else legs)[side] = chain
            points.append([6 * sign, 0, chest] if kind == 'arm' else [sign, -1, 0])
            owners.append(parent)
    weights = np.zeros((len(points), len(roles)))
    weights[np.arange(len(points)), owners] = 1
    heads = np.array(heads, float)
    rig = Rig('synthetic', [f'JOBJ_{i}' for i in range(len(roles))], np.array(parents), heads,
              heads + [0, 0, 0.1], ['skin'], np.array(points, float), weights)
    body = SimpleNamespace(plan='biped', root=0, chest=1, skull=2, spine=[0, 1, 2], arms=arms, legs=legs,
                           up=np.array([0., 0., 1.]), front=np.array([0., -1., 0.]), left=np.array([1., 0., 0.]),
                           forks={}, groups={})
    return rig, body, {'bones': {'roles': {str(i + 1): role for i, role in enumerate(roles)}}}


@pytest.mark.parametrize('ours,theirs,k', [(8., 1., .5), (1., 27., 3.), (5., 5., 1.)])
def test_the_scale_is_one_reading_of_two_rooms(ours, theirs, k):
    assert scale(ours, theirs) == pytest.approx(k)


def test_a_quadrupeds_head_height_runs_from_hind_hips_through_chest_to_neck():
    rig, body, _ = model(hips=4, chest=4, head=7, top=9)
    body.plan = 'quadruped'
    rig.heads[1:3, 1] = [-6, -10]
    for arm in body.arms.values():
        rig.heads[arm, 1] = -6
    # 4 above the floor + 6 along the back + 5 from chest to neck = 15.
    assert correspond.levels(correspond.model_chains(rig, body)[0])[2] == 15


def test_a_biped_with_dense_hair_and_a_neck_still_matches_head_joint_height():
    rig, body, _ = model()
    rig.names.append('hair')
    rig.parents = np.append(rig.parents, body.skull)
    rig.heads = np.vstack([rig.heads, [0, 0, 11]])
    rig.tails = np.vstack([rig.tails, [0, 0, 12]])
    rig.weights = np.pad(rig.weights, ((0, 0), (0, 1)))
    # Dense hair belongs to a child of the skull; density does not erase the neck below it.
    hair = np.tile([[0, 0, 11]], (30, 1))
    skin = np.zeros((len(hair), len(rig.names)))
    skin[:, -1] = 1
    rig.points = np.vstack([rig.points, hair])
    rig.weights = np.vstack([rig.weights, skin])
    assert np.count_nonzero(rig.weights[:, [body.skull, len(rig.names) - 1]].sum(axis=1) > 0.5) > len(rig.points) / 2
    assert correspond.levels(correspond.model_chains(rig, body)[0])[1:3] == [8, 10]


def ball():
    """A ball with mid-body pivots and its skin on a coincident child of the last spine joint."""
    rig, body, _ = model(hips=2, chest=4, head=6, top=12)
    body.plan = 'ball'
    rig.names.append('surface')
    rig.parents = np.append(rig.parents, body.skull)
    rig.heads = np.vstack([rig.heads, rig.heads[body.skull]])
    rig.tails = np.vstack([rig.tails, rig.tails[body.skull]])
    rig.weights = np.column_stack([rig.weights, rig.weights[:, body.skull]])
    rig.weights[:, body.skull] = 0
    return rig, body


@pytest.mark.parametrize('top_carriers', ['coincident', 'root', 'blended'])
def test_a_ball_with_mid_body_spine_joints_is_sized_by_its_spines_skin(top_carriers):
    rig, body = ball()
    if top_carriers != 'coincident':
        rig.weights[3] = 0
        carriers = [body.root] if top_carriers == 'root' else body.spine
        rig.weights[3, carriers] = 1 / len(carriers)
    chains, _ = correspond.model_chains(rig, body)
    assert chains['spine'].end[2] == correspond.levels(chains)[3] == 12


def test_a_separate_chain_above_a_balls_body_does_not_set_its_size():
    rig, body = ball()
    rig.names.append('antenna')
    rig.parents = np.append(rig.parents, len(rig.names) - 2)
    rig.heads = np.vstack([rig.heads, [0, 0, 12]])
    rig.tails = np.vstack([rig.tails, [0, 0, 24]])
    rig.points = np.vstack([rig.points, [0, 0, 24]])
    rig.weights = np.pad(rig.weights, ((0, 1), (0, 1)))
    rig.weights[-1, -1] = 1
    assert correspond.levels(correspond.model_chains(rig, body)[0])[3] == 12


def test_a_ball_whose_spine_already_ends_at_its_skin_top_keeps_its_size():
    rig, body, _ = model(hips=2, chest=4, head=6, top=12)
    body.plan = 'ball'
    old_end = correspond._chain(rig, body.spine, top=True).end
    assert np.array_equal(correspond.model_chains(rig, body)[0]['spine'].end, old_end)


def _sized(monkeypatch, height, plan='biped', fighter='biped'):
    """run.grown for a model whose height says ``height`` times its present size, of ``plan``, on a ``fighter``."""
    from rigswap.core import run
    monkeypatch.setattr(run, 'stands', lambda rig, shape, frig, description: height)
    description = {'bones': {'roles': {'1': 'body' if fighter == 'ball' else 'hips', '2': 'foot'}}}
    return lambda said=1.0: run.grown(None, SimpleNamespace(plan=plan), None, description, said)


def test_a_two_legged_model_stands_as_tall_to_its_head_joint_as_the_fighter():
    """The size before 2026-11-01, and again: one length of the whole standing body. Read bone by bone in between,
    the middle point's bone said 0.65 of the room for Crash where this says 0.87."""
    from rigswap.core import run
    seen = []

    def levels(chains, fighter=False):
        seen.append((chains, fighter))
        return [4.37, 7.01, 8.27, 14.21] if fighter else [4.07, 4.56, 5.78, 9.77]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(correspond, 'model_chains', lambda rig, shape: ('the model\'s', 'shared'))
        patch.setattr(correspond, 'fighter_chains', lambda frig, description: 'the fighter\'s')
        patch.setattr(correspond, 'levels', levels)
        assert run.stands(None, None, None, None) == pytest.approx(8.27 / 5.78)
        assert seen == [('the model\'s', False), ('the fighter\'s', True)]
        patch.setattr(correspond, 'levels', lambda chains, fighter=False: [0., 0., 0., 0.])
        assert run.stands(None, None, None, None) is None                       # no height to its head joint


def test_a_models_size_is_its_height_times_what_the_record_says(monkeypatch):
    # the CarnEvil skeleton stood at Ganondorf's room (the app's import of 2026-11-10): twice his height, every bone
    # squeezed to about half, its skull twice too big. As tall to the head joint as he is, it is 0.52 of that
    grown = _sized(monkeypatch, .52)
    assert grown() == (pytest.approx(.52), pytest.approx(.52))
    assert grown(1.2)[0] == pytest.approx(.624)                                 # the record's size multiplies it


def test_a_model_sized_by_its_height_takes_no_more_than_one_and_a_half_times_the_fighters_room(monkeypatch):
    # Kevin on Ness: a body that is its head, the head joint at its shoulders, 1.85 times its room by that height
    from rigswap.core.run import ROOM
    grown = _sized(monkeypatch, 1.85)
    assert grown() == (pytest.approx(ROOM ** (1 / 3)), pytest.approx(1.85)) and ROOM == 1.5


def test_a_four_legged_body_a_ball_and_any_body_on_a_ball_keep_the_fighters_room(monkeypatch):
    # Vulpix on Pikachu (size-2): sized by its bones it stood at 0.42 of its room, tiny and thin. A body that does
    # not stand as the fighter does has no height of the fighter's to match
    from rigswap.core.run import alike
    two_legs = {'bones': {'roles': {'1': 'hips', '2': 'spine'}}}
    ball = {'bones': {'roles': {'1': 'body', '2': 'foot'}}}
    assert alike(SimpleNamespace(plan='biped'), two_legs)
    assert not alike(SimpleNamespace(plan='quadruped'), two_legs) and not alike(SimpleNamespace(plan='ball'), two_legs)
    assert not alike(SimpleNamespace(plan='biped'), ball)
    for plan, fighter in (('quadruped', 'biped'), ('ball', 'biped'), ('biped', 'ball')):
        grown = _sized(monkeypatch, .42, plan, fighter)
        assert grown() == (1.0, None) and grown(1.1)[0] == pytest.approx(1.1)
    assert _sized(monkeypatch, None)() == (1.0, None)                           # no height read: the room
