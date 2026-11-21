"""The record: a body written down and read back is the body that was read; a hand's names its joints."""
import json

import numpy as np
import pytest

from rigswap.core import body, correspond, record
from tests.core.test_body import _bones, _figure, _read

UP, FRONT = np.array([0., 0., 1.]), np.array([0., -1., 0.])


@pytest.fixture(autouse=True)
def _no_record_left_behind():
    yield
    body.RECORD = None


def _written(rig, shape):
    chains, shared = correspond.model_chains(rig, shape)
    made = record.of_reading(rig, shape, chains, shared, UP, FRONT, {}, set(), {}, [], 'Mario', None)
    return chains, shared, json.loads(json.dumps(made))


def test_a_body_written_to_a_record_and_read_back_is_the_body_that_was_read():
    rig = _figure(_bones())
    first = _read(rig)
    chains, shared, told = _written(rig, first)
    body.RECORD = told
    second = body.read(rig, UP, FRONT)
    for field in ('root', 'spine', 'skull', 'legs', 'arms', 'chest', 'floor', 'plan', 'forks'):
        assert getattr(first, field) == getattr(second, field), field
    again, shared_again = correspond.model_chains(rig, second)
    assert shared == shared_again
    for name, chain in chains.items():
        assert chain.bones == again[name].bones, name
        assert list(chain.anchors[:3]) == list(again[name].anchors[:3]), name
        assert np.array_equal(chain.joints, again[name].joints) and np.array_equal(chain.end, again[name].end), name


def test_a_record_with_a_body_is_read_without_roles_or_forks():
    """What a pair is, the skeleton doesn't say (body.read: no roles, no limbs); the record does."""
    rig = _figure(_bones())
    _, _, told = _written(rig, _read(rig))
    assert not body.read(rig, UP, FRONT).legs
    body.RECORD = told
    assert body.read(rig, UP, FRONT).legs['left']


def test_a_hand_record_names_joints_and_the_chains_are_the_paths_between_them():
    rig = _figure(_bones())
    names = rig.names
    body.RECORD = {'plan': 'biped', 'hips': 'hips', 'chest': 'chest', 'neck': 'neck', 'skull': 'head',
                   'legs': {'left': {'hip': 'thigh.L', 'knee': 'shin.L', 'ankle': 'foot.L', 'toe': 'toe.L'},
                            'right': {'hip': 'thigh.R', 'knee': 'shin.R', 'ankle': 'foot.R', 'toe': 'toe.R'}},
                   'arms': {'left': {'shoulder': 'arm.L', 'elbow': 'fore.L', 'hand': 'hand.L'},
                            'right': {'shoulder': 'arm.R', 'elbow': 'fore.R', 'hand': 'hand.R'}}}
    seen = body.read(rig, UP, FRONT)
    assert [names[b] for b in seen.spine] == ['hips', 'spine', 'chest', 'neck', 'head']
    assert [names[b] for b in seen.legs['left']] == ['hips', 'thigh.L', 'shin.L', 'foot.L', 'toe.L']
    chains, _ = correspond.model_chains(rig, seen)
    leg, arm = chains['left leg'], chains['left arm']
    assert [names[leg.bones[a]] for a in leg.anchors] == ['thigh.L', 'foot.L', 'toe.L', 'shin.L']
    assert [names[arm.bones[a]] for a in arm.anchors] == ['arm.L', 'arm.L', 'hand.L', 'fore.L']
    assert names[chains['spine'].bones[chains['spine'].anchors[2]]] == 'neck'


def test_a_hand_record_naming_a_bone_the_model_lacks_says_so():
    rig = _figure(_bones())
    body.RECORD = {'hips': 'hips', 'chest': 'chest', 'skull': 'no such bone'}
    with pytest.raises(ValueError, match='no such bone'):
        body.read(rig, UP, FRONT)
