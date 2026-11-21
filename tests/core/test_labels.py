"""The labels ask's code side: the pairs it numbers and how the answers are counted (the pictures need Blender)."""
import numpy as np

from rigswap.core import body, labels
from tests.core.test_body import _bones, _figure, _roles


def _answer(*roles):
    return {'character': '', 'pairs': [{'number': n, 'what': '', 'role': r} for n, r in roles]}


def test_a_role_goes_to_the_pair_most_answers_give_it():
    said = [_answer((1, 'other'), (2, 'legs')), _answer((1, 'legs'), (2, 'legs')), _answer((2, 'legs'))]
    assert labels.tally(said, [1, 2]) == {'legs': [2]}


def test_a_role_half_the_answers_give_is_no_answer():
    said = [_answer((1, 'legs')), _answer((2, 'legs')), _answer((1, 'cant_tell')), _answer((2, 'other'))]
    assert labels.tally(said, [1, 2]) == {}


def test_every_pair_most_answers_give_a_role_takes_it_a_limb_in_pieces():
    # Boomerang Bro's thighs, shins and feet, each a pair off his hips, all named legs
    said = [_answer((1, 'legs'), (2, 'legs'), (3, 'legs'), (4, 'arms'))] * 2 + [_answer((1, 'legs'), (3, 'other'))]
    assert labels.tally(said, [1, 2, 3, 4]) == {'legs': [1, 2, 3], 'arms': [4]}


def test_missing_answers_count_against_a_role():
    said = [_answer((1, 'arms')), None, {}]
    assert labels.tally(said, [1]) == {}


def test_each_role_is_counted_on_its_own():
    said = [_answer((1, 'legs'), (2, 'front legs'), (3, 'other'))] * 2 + [_answer((1, 'legs'), (3, 'arms'))]
    assert labels.tally(said, [1, 2, 3]) == {'legs': [1], 'front legs': [2]}


def _parts(*parts):
    return {'character': 'x', 'parts': [dict(number=n, what='', kind=k, rest=r) for n, k, r in parts]}


def test_a_chain_is_the_kind_most_answers_give_it_and_held_out_only_if_every_one_says_so():
    out, drawn = 'held out by the pose', 'as drawn'
    said = [_parts((1, 'hair', out), (2, 'armour', drawn), (3, 'cloth', drawn)),
            _parts((1, 'hair', out), (2, 'prop', drawn), (3, 'cloth', drawn)),
            _parts((1, 'tail', drawn), (3, 'cant_tell', 'cant_tell'))]
    assert labels.tally_chains(said, [1, 2, 3, 4]) == {1: ('hair', True), 2: (None, False), 3: ('cloth', False),
                                                       4: (None, False)}
    wavering = [_parts((1, 'hair', out))] * 2 + [_parts((1, 'hair', drawn))]
    assert labels.tally_chains(wavering, [1]) == {1: ('hair', False)}      # turning it takes every answer
    held = [_parts((1, 'tail', out))] * 3
    assert labels.tally_chains(held, [1]) == {1: ('tail', False)}          # only hair and cloth hang
    assert labels.tally_chains([None, None, None], [1]) == {1: (None, False)}



def test_a_prop_is_the_kind_most_answers_give_it():
    def said(*objects):
        return {'character': 'x', 'objects': [dict(number=n, what='', kind=k) for n, k in objects]}
    answers = [said((1, 'sword'), (2, 'shield'), (3, 'none')), said((1, 'sword'), (2, 'none'), (3, 'none')),
               said((1, 'shield'), (2, 'shield'))]
    assert labels.tally_equipment(answers, [1, 2, 3, 4], ['shield', 'sword']) == {1: 'sword', 2: 'shield', 3: None,
                                                                                 4: None}
    assert labels.tally_equipment([said((1, 'sword')), None, said((1, 'cant_tell'))], [1], ['shield', 'sword']) == {
        1: None}

def _shape(rig):
    return body.read(rig, [0, 0, 1], [0, -1, 0], _roles(rig))


def test_the_chains_are_every_branch_off_the_body_a_twin_and_its_mirror_one():
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'sleeve.{side}', f'fore.{side}', [sign * .7, .05, 1.45], 6),
                  (f'helper.{side}', f'thigh.{side}', [sign * .3, 0, .85], 0)]     # carries no surface of its own
    bones += [('braid', 'head', [0, .1, 2.0], 5), ('braid2', 'braid', [0, .15, 1.7], 5),
              ('cape', 'chest', [0, .2, 1.5], 9)]
    rig = _figure(bones)
    found = labels.branches(rig, _shape(rig), [0, 0, 1])
    roots = [sorted(rig.names[b] for b in f['roots']) for f in found]
    assert roots == [['sleeve.L', 'sleeve.R'], ['braid'], ['cape']]           # heaviest first
    assert [rig.names[b] for b in found[1]['bones'][0]] == ['braid', 'braid2']
    found = labels.branches(rig, _shape(rig), [0, 0, 1], {'sleeve.L': 'carried', 'braid': 'helper'})
    assert [sorted(rig.names[b] for b in f['roots']) for f in found] == [['sleeve.L', 'sleeve.R'], ['cape']]


def test_a_pair_is_named_by_every_end_it_carries_not_one_path():
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:     # a sleeve off the forearm, heavier than the hand
        bones += [(f'sleeve.{side}', f'fore.{side}', [sign * .7, 0, 1.3], 20),
                  (f'sleevetip.{side}', f'sleeve.{side}', [sign * .7, 0, .9], 20)]
    rig = _figure(bones)
    found = labels.candidates(rig, np.array([0., 0., 1.]))
    arms = next(f for f in found if rig.names[f['roots'][0]].startswith('arm'))
    named = labels._named(rig, rig.children(), arms)
    assert named.startswith('arm.') and 'hand.' in named and 'sleevetip.' in named


def test_candidates_are_every_twin_pair_with_what_each_side_carries():
    rig = _figure(_bones())
    found = labels.candidates(rig, np.array([0., 0., 1.]))
    roots = {frozenset(rig.names[b] for b in f['roots']) for f in found}
    assert roots == {frozenset({'thigh.L', 'thigh.R'}), frozenset({'arm.L', 'arm.R'})}
    legs = next(f for f in found if rig.names[f['roots'][0]].startswith('thigh'))
    side = {rig.names[b] for b in legs['bones'][0]}
    assert side in ({'thigh.L', 'shin.L', 'foot.L', 'toe.L'}, {'thigh.R', 'shin.R', 'foot.R', 'toe.R'})
