"""Decisions before build: the file's origin, else the picture's nearest candidate up, which the skeleton's legs may
overrule once the labels name them; and the size."""
import numpy as np
import pytest

from rigswap.core import run


class OrientationChecked(Exception):
    pass


def _swap(monkeypatch, tmp_path, floor, picture, sure, stands, roles):
    from types import SimpleNamespace
    rig = SimpleNamespace(mirror=np.array([1., 0., 0.]))
    seen = {'legs': None, 'fronts': []}
    monkeypatch.setenv('RIGSWAP_FILL', '0')     # these are the reading's own decisions: no look fills the body
    monkeypatch.setattr(run.rigs, 'pick', lambda *_: {'file': 'synthetic', 'blend': 'body.blend', 'armatures': True})
    monkeypatch.setattr(run.assemble, 'read', lambda *_: (rig, None))
    monkeypatch.setattr(run.segments, 'lost', lambda *_: False)
    monkeypatch.setattr(run.body_reading, 'on_origin', lambda *_: floor)

    def choose_up(*_):
        assert floor is None, 'a file standing on its origin is not asked'
        return picture, 'C', sure

    def standing(_rig, _side, legs):
        seen['legs'] = sorted(legs)
        return stands

    def choose(facing, _out, _ask, up):
        seen['fronts'].append(up)
        return facing, 'A'

    def forks(_rig, up, *_):
        seen['up'] = up
        raise OrientationChecked

    monkeypatch.setattr(run.front_reading, 'choose_up', choose_up)
    monkeypatch.setattr(run.body_reading, 'standing', standing)
    monkeypatch.setattr(run.front_reading, 'choose', choose)
    monkeypatch.setattr(run.labels, 'read', lambda *_: (roles, {'roles': {}}))
    monkeypatch.setattr(run.fork_reading, 'read', forks)
    with pytest.raises(OrientationChecked):
        run.swap(tmp_path, 'synthetic', tmp_path / 'output', None)
    return seen


@pytest.mark.parametrize('noise', [0., 4.371138828673793e-8, -4.371138828673793e-8])
@pytest.mark.parametrize('sure, upright_score, expected', [(True, 20, 'picture'), (True, 0, 'skeleton'),
                                                         (False, 20, 'skeleton')])
def test_picture_up_matches_nearest_candidate_before_standing_override(monkeypatch, tmp_path,
                                                                       noise, sure, upright_score, expected):
    picture = np.array([noise, 0., 1.])
    sideways = np.array([0., -1., 0.])
    stands = [(np.array([0., 1., 0.]), 4.), (sideways, 36.), (np.array([0., 0., 1.]), upright_score),
              (np.array([0., 0., -1.]), 0.)]
    roles = {'thigh.L': 'legs', 'thigh.R': 'legs', 'plume.L': 'other', 'arm.L': 'arms'}
    seen = _swap(monkeypatch, tmp_path, None, picture, sure, stands, roles)
    # the skeleton stands by the pairs the labels name legs only (a crest's plumes, Pidgeot's, hold nothing up)
    assert seen['legs'] == ['thigh.L', 'thigh.R']
    assert np.array_equal(seen['up'], picture if expected == 'picture' else sideways)
    # the front is asked for the picture's up, and again for the skeleton's when it overrules it
    assert len(seen['fronts']) == (1 if expected == 'picture' else 2)


def test_a_file_standing_on_its_origin_stands_that_way_unasked_and_unoverruled(monkeypatch, tmp_path):
    floor = np.array([0., 0., 1.])
    stands = [(np.array([0., 0., -1.]), 50.)]
    seen = _swap(monkeypatch, tmp_path, floor, None, None, stands, {'thigh.L': 'legs', 'thigh.R': 'legs'})
    assert np.array_equal(seen['up'], floor) and seen['legs'] is None and len(seen['fronts']) == 1
