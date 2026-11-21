"""A complaint changes one entry of the record: a size, the front, or joints named by their numbers."""
import json

import numpy as np
import pytest

from rigswap import ai
from rigswap.core import body, complaint, correspond, fill, record
from tests.core.test_body import _bones, _figure, _read

UP, FRONT = np.array([0., 0., 1.]), np.array([0., -1., 0.])


@pytest.fixture(autouse=True)
def _no_record_left_behind():
    yield
    body.RECORD = None


def _full(rig):
    shape = _read(rig)
    chains, shared = correspond.model_chains(rig, shape)
    return json.loads(json.dumps(record.of_reading(rig, shape, chains, shared, UP, FRONT, {}, set(), {}, [], 'Mario',
                                                    None)))


def test_a_size_named_outright_is_multiplied_in_and_kept_within_bounds():
    made, changed = complaint.sized({'head': 1.0, 'limbs': 0.8}, {'head': 0.8, 'limbs': 5})
    assert made == {'head': 0.8, 'limbs': 1.6} and changed == {'head': [1.0, 0.8], 'limbs': [0.8, 1.6]}
    with pytest.raises(complaint.Unfixable, match='no size'):
        complaint.sized({}, {'nose': 2})


def test_a_look_naming_a_size_changes_that_size_alone():
    made, changed = complaint.applied({'head': 1.0, 'hips': 'hips'}, {'change': 'head', 'times': 0.75, 'why': 'big'})
    assert made['head'] == 0.75 and made['hips'] == 'hips' and list(changed) == ['head']


def test_a_look_naming_the_front_turns_it_to_that_side():
    made, changed = complaint.applied({'front': [0, -1, 0]}, {'change': 'front', 'front_view': 'C', 'why': 'back'},
                                      fronts=fill.sides(UP))
    assert made['front'] == [0.0, 1.0, 0.0] and changed['front'][0] == [0, -1, 0]


def test_a_full_record_turned_into_joints_builds_the_body_it_was_read_as():
    """A record a run wrote lists chains; to rename one joint it is first said as joints, and nothing else moves."""
    rig = _figure(_bones())
    full = _full(rig)
    body.RECORD = full
    first = body.read(rig, UP, FRONT)
    joints = record.as_joints(full)
    assert 'body' not in joints and joints['hips'] == 'hips' and joints['legs']['left']['hip'] == 'thigh.L'
    body.RECORD = joints
    second = body.read(rig, UP, FRONT)
    for field in ('spine', 'skull', 'legs', 'arms', 'chest', 'plan'):
        assert getattr(first, field) == getattr(second, field), field


def test_a_look_naming_joints_renames_them_by_their_numbers():
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    n = {name: j['n'] for j in rows for name in j['names']}
    answer = {'change': 'joints', 'why': 'the ankles are one joint too low',
              'joints': [{'part': 'left leg ankle', 'to': n['shin.L']}, {'part': 'right leg ankle', 'to': n['shin.R']}]}
    made, changed = complaint.applied(_full(rig), answer, rows)
    assert made['legs']['left']['ankle'] == 'shin.L' and 'bones' not in made['legs']['left']
    assert changed == {'left leg ankle': ['foot.L', 'shin.L'], 'right leg ankle': ['foot.R', 'shin.R']}
    body.RECORD = made
    assert body.read(rig, UP, FRONT).record['legs']['left']['ankle'] == rig.names.index('shin.L')


@pytest.mark.parametrize('answer, why', [
    ({'change': 'nothing', 'why': 'the texture is wrong, not the body'}, 'texture'),
    ({'change': 'head', 'why': 'x'}, 'no factor'),
    ({'change': 'joints', 'joints': [{'part': 'left wing tip', 'to': 3}], 'why': 'x'}, 'no part'),
    ({'change': 'joints', 'joints': [{'part': 'neck', 'to': 999}], 'why': 'x'}, 'no joint')])
def test_a_change_the_record_cannot_hold_says_so_in_words(answer, why):
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    with pytest.raises(complaint.Unfixable, match=why):
        complaint.applied(_full(rig), answer, rows)


def test_the_answers_the_question_asks_for_pass_the_asks_own_check():
    assert ai._check({'change': 'head', 'times': 0.8, 'why': 'the head is too big'}, complaint.SCHEMA) is None
    assert ai._check({'change': 'joints', 'joints': [{'part': 'neck', 'to': 4}], 'why': 'x'}, complaint.SCHEMA) is None
    assert ai._check({'change': 'bigger', 'why': 'x'}, complaint.SCHEMA)


def test_the_record_is_described_to_the_look_by_joint_numbers():
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    said = complaint.described(_full(rig), rows)
    n = {name: j['n'] for j in rows for name in j['names']}
    assert f'hips {n["hips"]}' in said and f'ankle {n["foot.L"]}' in said


def test_what_an_assistant_made_of_the_words_is_shown_apart_and_only_when_it_differs(tmp_path):
    """The person's own words are the complaint; an assistant's summary of them only says which part they mean, and a
    size it added is not theirs. A summary that is the words again is not shown twice."""
    rig = _figure(_bones())
    (tmp_path / 'record.json').write_text(json.dumps(_full(rig)), encoding='utf-8')
    for name in ('fill-colours.png', 'tripwire.png', 'tripwire-fighter.png'):
        (tmp_path / name).write_bytes(b'png')
    asked = []

    def ask(question, pictures, schema, default, route=None):
        asked.append(question)
        return {'change': 'head', 'times': 0.8, 'why': 'smaller head'}, {}
    complaint.fix(tmp_path, "it's   still too big", ask, heard='make the head much smaller, around 0.6')
    assert 'The person says: "it\'s still too big"' in asked[0]
    assert 'summed it up as: "make the head much smaller, around 0.6"' in asked[0]
    assert "is the assistant's guess, not theirs" in asked[0]
    complaint.fix(tmp_path, 'His head is too big.', ask, heard='his head is too big.')
    complaint.fix(tmp_path, 'His head is too big.', ask)
    assert all('summed it up' not in q for q in asked[1:])


def test_the_look_is_told_what_a_fitted_skeletons_joints_are_and_no_other_costumes_question_changes(tmp_path):
    """A file that had no skeleton has one fitted inside it (onepiece.py): the look is told what its numbered joints
    are and that all past the neck is one piece with the fighter's head. A rigged file's costume is told what the
    neck joint does, in the words for a skeleton of its own (the head bone's joint, not the neck bone's)."""
    asked = []

    def ask(question, pictures, schema, default, route=None):
        asked.append(question)
        return {'change': 'head', 'way': 'smaller', 'why': 'x'}, {}
    for folder, names in (('fit', ['slice7', 'slice4', 'slice2', 'slice1']), ('rigged', ['hips', 'chest', 'neck', 'head'])):
        swap = tmp_path / folder
        swap.mkdir()
        (swap / 'record.json').write_text(json.dumps(dict(zip(complaint.CORE, names))), encoding='utf-8')
        (swap / 'fill-joints.json').write_text(json.dumps([{'n': i + 1, 'names': [name]}
                                                           for i, name in enumerate(names)]), encoding='utf-8')
        for name in ('fill-colours.png', 'tripwire.png', 'tripwire-fighter.png', 'fill-joints-A.png',
                     'fill-joints-B.png'):
            (swap / name).write_bytes(b'png')
        complaint.fix(swap, 'his face is squashed', ask)
    assert "The fighter's head is put on the neck joint: everything past it moves as one stiff piece" in asked[0]
    assert complaint.OWN in asked[1] and complaint.FITTED not in asked[1]
    assert "the neck joint belongs on the head bone's joint, at the top of the neck" in asked[1]
    assert asked[0].replace(complaint.FITTED, '') == asked[1].replace(complaint.OWN, '')


def test_a_size_with_no_amount_takes_a_modest_step_the_way_the_look_says():
    """The look says which way; how much is the person's when they named it, else one step. No way and no amount is
    no change."""
    made, changed = complaint.applied({'head': 1.0}, {'change': 'head', 'way': 'smaller', 'why': 'x'})
    assert changed == {'head': [1.0, 0.8]}
    made, changed = complaint.applied({'limbs': 0.8}, {'change': 'limbs', 'way': 'bigger', 'why': 'x'})
    assert changed == {'limbs': [0.8, 1.0]}
    made, changed = complaint.applied({}, {'change': 'head', 'way': 'smaller', 'times': 0.7, 'why': 'x'})
    assert changed == {'head': [1.0, 0.7]}, 'an amount the person named is theirs'
    assert ai._check({'change': 'trunk', 'way': 'bigger', 'why': 'x'}, complaint.SCHEMA) is None
    assert ai._check({'change': 'trunk', 'way': 'wider', 'why': 'x'}, complaint.SCHEMA)


def test_a_look_naming_a_piece_leaves_a_request_beside_the_record_for_the_build_to_settle():
    """Diddy Kong on Fox (2026-11-10): one import kept his barrel jetpack, and "take the jetpack off" was answered
    "nothing": no change left a piece out. Which of the model's pieces the words mean is settled by the build, which
    has the model open (pieces.py); the look says the way and the person's words for it."""
    record = {'drop': ['Diddy_Gun'], 'size': 1.0}
    out, changed = complaint.applied(record, {'change': 'leave out', 'piece': ' the barrel  jetpack on his back ',
                                              'why': 'x'})
    assert out['pieces'] == {'way': 'out', 'piece': 'the barrel jetpack on his back'}
    assert out['drop'] == ['Diddy_Gun'] and 'pieces' not in record
    assert changed == {'pieces': [None, 'leave out: the barrel jetpack on his back']}
    out, _ = complaint.applied(record, {'change': 'bring back', 'piece': 'his gun', 'why': 'x'})
    assert out['pieces'] == {'way': 'back', 'piece': 'his gun'}
    with pytest.raises(complaint.Unfixable, match='names no piece'):
        complaint.applied(record, {'change': 'leave out', 'why': 'x'})
    with pytest.raises(complaint.Unfixable, match='no list of what it leaves out'):
        complaint.applied({'size': 1.0}, {'change': 'leave out', 'piece': 'his hat', 'why': 'x'})
    assert ai._check({'change': 'leave out', 'piece': 'the jetpack', 'why': 'x'}, complaint.SCHEMA) is None
    assert ai._check({'change': 'bring back', 'piece': 'his hat', 'why': 'x'}, complaint.SCHEMA) is None


def test_a_look_naming_a_part_off_the_body_changes_what_it_and_all_below_it_are_called():
    """What swings is what the record calls hair, cloth or a tail; a part held stiff is called none of them. The
    look names the joint where the part leaves the body and what it should be; it and all that hangs from it take
    that kind, and nothing else of the record changes."""
    rows = [{'n': 1, 'parent': None, 'names': ['Hips']}, {'n': 2, 'parent': 1, 'names': ['Tail1']},
            {'n': 3, 'parent': 2, 'names': ['Tail2', 'Tail2_end']}, {'n': 4, 'parent': 1, 'names': ['Hair1']}]
    record = {'drop': [], 'parts': {'Tail1': {'kind': 'tail', 'held_out': False},
                                    'Tail2': {'kind': 'tail', 'held_out': False},
                                    'Hair1': {'kind': 'body', 'held_out': True}}}
    out, changed = complaint.applied(record, {'change': 'swing', 'joint': 2, 'as': 'stiff', 'why': 'x'}, rows)
    assert out['parts'] == {'Tail1': {'kind': 'other', 'held_out': False}, 'Tail2': {'kind': 'other', 'held_out': False},
                            'Hair1': {'kind': 'body', 'held_out': True}}
    assert changed == {'parts from Tail1': ['tail', 'other']} and record['parts']['Tail1']['kind'] == 'tail'
    out, changed = complaint.applied(record, {'change': 'swing', 'joint': 4, 'as': 'hair', 'why': 'x'}, rows)
    assert out['parts']['Hair1'] == {'kind': 'hair', 'held_out': True} and changed == {'parts from Hair1': ['body', 'hair']}
    for answer, why in (({'change': 'swing', 'joint': 1, 'as': 'hair'}, 'on the body itself'),
                        ({'change': 'swing', 'joint': 2, 'as': 'tail'}, 'already swinging as tail'),
                        ({'change': 'swing', 'joint': 9, 'as': 'hair'}, 'names no joint'),
                        ({'change': 'swing', 'joint': 2, 'as': 'wobbly'}, 'no way for the part to move')):
        with pytest.raises(complaint.Unfixable, match=why):
            complaint.applied(record, dict(answer, why='x'), rows)
    with pytest.raises(complaint.Unfixable, match='no numbered joints'):
        complaint.applied(record, {'change': 'swing', 'joint': 2, 'as': 'stiff', 'why': 'x'})
    assert ai._check({'change': 'swing', 'joint': 2, 'as': 'cloth', 'why': 'x'}, complaint.SCHEMA) is None
    assert ai._check({'change': 'swing', 'joint': 2, 'as': 'wobbly', 'why': 'x'}, complaint.SCHEMA)
