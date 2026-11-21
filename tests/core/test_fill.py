"""The look's answer becomes the body's entries: sides from where the limbs lie, and the checks code is sure of."""
import numpy as np
import pytest

from rigswap.core import body, fill
from tests.core.test_body import _bones, _figure

UP = np.array([0., 0., 1.])


@pytest.fixture(autouse=True)
def _no_record_left_behind():
    yield
    body.RECORD = None


def _answer(rows, **change):
    n = {name: j['n'] for j in rows for name in j['names']}
    answer = {'front_view': 'A', 'plan': 'biped', 'hips': n['hips'], 'chest': n['chest'], 'neck': n['neck'],
              'skull': n['head'],
              'legs': {s: {'hip': n[f'thigh.{x}'], 'knee': n[f'shin.{x}'], 'ankle': n[f'foot.{x}'], 'toe': n[f'toe.{x}']}
                       for s, x in (('left', 'L'), ('right', 'R'))},
              'arms': {s: {'clavicle': None, 'shoulder': n[f'arm.{x}'], 'elbow': n[f'fore.{x}'], 'hand': n[f'hand.{x}']}
                       for s, x in (('left', 'L'), ('right', 'R'))}}
    answer.update(change)
    return answer


def test_the_four_sides_of_a_file_standing_on_z_are_minus_y_plus_x_plus_y_minus_x():
    way = fill.sides(UP)
    assert [way[v].round(6).tolist() for v in 'ABCD'] == [[0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]


def test_the_table_numbers_every_joint_from_one_down_the_tree_with_its_parent():
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    assert [j['n'] for j in rows] == list(range(1, len(rows) + 1)) and rows[0]['parent'] is None
    by = {name: j for j in rows for name in j['names']}
    assert by['shin.L']['parent'] == by['thigh.L']['n'] and by['head']['skin'] == 40
    assert 'thigh.L' in fill.table(rows)


def test_an_answer_names_the_body_and_the_front_is_the_side_the_camera_stood_on():
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    made = fill.entries(rig, mass, rows, _answer(rows), UP)
    assert made['front'] == [0.0, -1.0, 0.0] and made['hips'] == 'hips' and made['skull'] == 'head'
    assert made['legs']['left'] == {'hip': 'thigh.L', 'knee': 'shin.L', 'ankle': 'foot.L', 'toe': 'toe.L'}
    body.RECORD = made
    assert [rig.names[b] for b in body.read(rig, UP, np.array(made['front'])).legs['left']][1] == 'thigh.L'


def test_left_and_right_are_read_from_where_the_limbs_lie_not_from_the_answer():
    """The lab: 3 answers of 60 had the sides swapped. Facing -y, the character's left is +x."""
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    swapped = _answer(rows)
    for limbs in ('legs', 'arms'):
        swapped[limbs] = {'left': swapped[limbs]['right'], 'right': swapped[limbs]['left']}
    made = fill.entries(rig, mass, rows, swapped, UP)
    assert made['legs']['left']['hip'] == 'thigh.L' and made['arms']['right']['hand'] == 'hand.R'


def test_facing_the_other_way_the_sides_turn_with_the_front():
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    made = fill.entries(rig, mass, rows, _answer(rows, front_view='C'), UP)
    assert made['front'] == [0.0, 1.0, 0.0] and made['legs']['left']['hip'] == 'thigh.R'


@pytest.mark.parametrize('change, why', [
    ({'hips': 999}, 'no joint of the table'), ({'front_view': 'E'}, 'not one of'),
    ({'skull': None}, 'no joint of the table')])
def test_an_answer_the_body_cannot_be_built_from_says_why(change, why):
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    with pytest.raises(fill.Unusable, match=why):
        fill.entries(rig, mass, rows, _answer(rows, **change), UP)


def test_a_limb_with_one_end_or_named_on_one_side_only_is_unusable():
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    one_end = _answer(rows)
    one_end['arms']['left']['hand'] = None
    with pytest.raises(fill.Unusable, match='names no hand'):
        fill.entries(rig, mass, rows, one_end, UP)
    one_side = _answer(rows)
    one_side['legs']['right'] = dict.fromkeys(fill.LEG)
    with pytest.raises(fill.Unusable, match='not the other'):
        fill.entries(rig, mass, rows, one_side, UP)


def test_a_ball_or_a_body_with_no_legs_is_left_to_the_reading():
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    assert fill.entries(rig, mass, rows, _answer(rows, plan='ball'), UP) is None
    legless = _answer(rows)
    legless['legs'] = {s: dict.fromkeys(fill.LEG) for s in ('left', 'right')}
    assert fill.entries(rig, mass, rows, legless, UP) is None


def test_a_limb_whose_joints_are_not_one_under_another_is_listed_joint_by_joint():
    """A foot hung on a floor control beside the hips (Swampert): the leg is its thigh, then that foot."""
    bones = _bones() + [('floor', None, [0, 0, 0], 0), ('pad.L', 'floor', [.3, 0, .05], 6),
                        ('pad.R', 'floor', [-.3, 0, .05], 6)]
    bones = [('root', None, [0, 0, 0], 0)] + [(n, p or 'root', h, c) for n, p, h, c in bones]
    rig = _figure(bones)
    mass, rows = fill.joints(rig, UP)
    n = {name: j['n'] for j in rows for name in j['names']}
    answer = _answer(rows)
    for side, x in (('left', 'L'), ('right', 'R')):
        answer['legs'][side] = {'hip': n[f'thigh.{x}'], 'knee': None, 'ankle': n[f'pad.{x}'], 'toe': None}
    made = fill.entries(rig, mass, rows, answer, UP)
    assert made['legs']['left']['bones'] == ['thigh.L', 'pad.L']


def test_a_fits_look_may_say_the_body_has_no_neck_and_the_fit_says_where_the_head_goes():
    """Crash (dots-1, dots-3): the look names where a head begins within a joint or two, and on a face that reaches
    the shoulders that margin left his grin on the trunk. A skeleton fitted inside the model is told a body with no
    neck is answered null; the fit (``settle``, onepiece.named) puts the head where the arms leave."""
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    given = []

    def settle(out):
        given.append(dict(out))
        return rig, {**out, 'neck': 'chest'}        # as onepiece.named: the Rig, and the entries as the fit holds them
    made = fill.entries(rig, mass, rows, _answer(rows, neck=None), UP, settle)
    assert given[0]['neck'] is None and given[0]['chest'] == 'chest' and given[0]['skull'] == 'head'
    assert made['neck'] == 'chest'
    assert 'give null for the neck' in fill.FITTED and 'null for the neck' not in fill.QUESTION
    with pytest.raises(fill.Unusable, match='neck is None, which is no joint of the table'):
        fill.entries(rig, mass, rows, _answer(rows, neck=None), UP)       # a file's own skeleton has no such answer


def test_a_files_own_skeleton_is_told_where_the_fighters_head_is_put_and_a_fits_question_is_as_it_was(monkeypatch):
    """The CarnEvil skeleton (the app's import of 2026-11-10): told only "the joint the head turns on", the look named
    the bone called neck, whose joint is at the neck's base; the fighter's head went there and the neck had no room."""
    rig = _figure(_bones())
    asked = []

    def ask(question, pictures, schema, default, route=None):
        asked.append(question)
        return {}, {}
    monkeypatch.setattr(fill, 'shown', lambda rig, up, out: (*fill.joints(rig, up), []))        # no pictures drawn
    fill.read(rig, UP, None, ask)
    fill.read(rig, UP, None, ask, settle=lambda out: (rig, out))
    own, fit = asked[0], asked[2]
    old = ("neck: the joint the head turns on. skull: the joint that carries the head's own surface. "
           "In a simple skeleton two of these can be the same joint.")
    assert "neck: the joint the fighter's head is put on" in own and "the head bone's joint, at the top of the neck" in own
    assert old not in own and fill.FITTED not in own
    assert old in fit and fill.FITTED in fit and "the head bone's joint" not in fit


def test_the_answer_the_question_asks_for_passes_the_asks_own_check():
    """fill-s1: a schema form ai._check did not know made every answer "no answer"; the look had answered."""
    from rigswap import ai
    from rigswap.plugins.melee import tripwire
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    assert ai._check(dict(_answer(rows), character='someone', notes=''), fill.SCHEMA) is None
    assert ai._check(dict(_answer(rows, neck=None), character='someone', notes=''), fill.SCHEMA) is None
    assert ai._check(dict(_answer(rows), hips='the hips'), fill.SCHEMA)
    assert ai._check({'upright': 'yes', 'attached': 'no', 'what': 'an arm floats'}, tripwire.SCHEMA) is None
    assert ai._check({'upright': 'maybe', 'attached': 'no'}, tripwire.SCHEMA)


def test_a_limbs_line_runs_from_the_spine_joint_it_leaves_out_to_its_tip():
    """What a fitted skeleton's limb is shown as, alone (fill.closer): the joint of the spine it leaves, every joint
    down to the last one the look named, and on to the limb's tip."""
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    n = {name: j['n'] for j in rows for name in j['names']}
    name = {j['n']: j['names'][0] for j in rows}
    answer = _answer(rows)
    dots, held = fill.line(rows, answer, 'arms', 'left')
    assert [name[d] for d in dots] == ['chest', 'arm.L', 'fore.L', 'hand.L'] and held == 4
    dots, held = fill.line(rows, answer, 'legs', 'right')
    assert [name[d] for d in dots] == ['hips', 'thigh.R', 'shin.R', 'foot.R', 'toe.R'] and held == 5
    answer['legs']['right'] = {'hip': n['thigh.R'], 'knee': None, 'ankle': n['shin.R'], 'toe': None}   # named short
    dots, held = fill.line(rows, answer, 'legs', 'right')
    assert [name[d] for d in dots] == ['hips', 'thigh.R', 'shin.R', 'foot.R', 'toe.R'] and held == 3
    answer['arms']['left'] = dict.fromkeys(fill.ARM)                                    # the body has no such limb
    assert fill.line(rows, answer, 'arms', 'left') == ([], 0)
    answer['arms']['right'] = {'clavicle': None, 'shoulder': n['spine'], 'elbow': None, 'hand': n['chest']}
    assert fill.line(rows, answer, 'arms', 'right') == ([], 0)                          # named on the spine: no limb


def test_a_line_says_how_far_along_each_dot_lies_and_how_far_it_turns_there():
    """Goku's bare right arm (size-2): the fit knew it bends 47 degrees at one dot; the look was never told."""
    places = [(0, 0), (10, 0), (20, 0), (30, 0), (30, -10), (30, -20), (30, -30)]
    rows = [{'n': i + 1, 'parent': i or None, 'across': x, 'deep': 0.0, 'high': z, 'skin': 5, 'below': 6 - i,
             'names': [f'slice{i}']} for i, (x, z) in enumerate(places)]
    said = fill.turns(rows, [1, 2, 3, 4, 5, 6, 7])
    assert [round(a) for a, _ in said] == [0, 17, 33, 50, 67, 83, 100]
    assert [round(t) for _, t in said] == [0, 0, 45, 90, 45, 0, 0]          # read two dots each way: the corner is 4
    assert fill.limb_table(rows, [1, 2, 3, 4]).splitlines()[0].split() == ['n', 'along', 'turn', 'skin']


def test_the_camera_looks_square_onto_a_limb_from_its_own_side_of_the_body():
    """A bent limb is seen along the way its steps reach least; a straight one across it, from the side given."""
    front = np.array([0., -1., 0.])
    bent = [[1, 0, 5], [1, 0, 3], [1, -2, 3], [1, -4, 3]]                    # down, then forward: its plane faces x
    out, top = fill.facing(bent, [0, 0, 3], UP, front)
    assert np.allclose(out, [1, 0, 0]) and np.allclose(top, UP)
    out, top = fill.facing([[-1, 0, 5], [-1, 0, 3], [-1, -2, 3]], [0, 0, 3], UP, front)
    assert np.allclose(out, [-1, 0, 0])                                     # the other side's limb, from its side
    out, top = fill.facing([[1, 0, 5], [1, 0, 3], [1, 0, 1]], [0, 0, 3], UP, front)
    assert abs(out[2]) < 1e-9 and np.isclose(abs(out[1]), 1)                # straight: across it, from the front
    # an arm held out level, a little off true: seen from the front, and the picture's up is still the model's
    # (the arm's own way lies exactly across the view and was taken for up: the arm stood on end, look-1)
    out, top = fill.facing([[1, 0, 5], [3, 0.04, 5.02], [5, 0.08, 5.04]], [0, 0, 3], UP, front)
    assert abs(out[1]) > 0.99 and top @ UP > 0.99
    # a leg that only slants and has a foot is seen from the side: the dots' places alone read it as flat (Goku)
    leg = [[1, 0, 9], [1.3, 0, 6], [1.6, 0, 3], [1.9, 0, 0.3], [1.9, -1.5, 0]]
    out, top = fill.facing(leg, [0, 0, 5], UP, front)
    assert abs(out[0]) > 0.95


def test_each_limb_of_a_fit_is_asked_again_on_its_own_picture_and_its_joints_taken(monkeypatch):
    """Goku on Mario (size-2): the look, on two whole-body pictures, named his right elbow one dot from the hand. Each
    limb of a fitted skeleton is asked again alone; no collar joint is asked; an answer out of order is asked once
    more, told why; a limb with no usable answer keeps the joints the first look gave. The arms alone are asked
    unless the legs are named too (fill.CLOSER: a fitted leg begins at the crotch, and its true knee can leave it no
    thigh)."""
    rig = _figure(_bones())
    mass, rows = fill.joints(rig, UP)
    n = {name: j['n'] for j in rows for name in j['names']}
    first = _answer(rows, character='someone')
    first['arms']['left'] = {'clavicle': n['arm.L'], 'shoulder': n['fore.L'], 'elbow': None, 'hand': n['hand.L']}
    monkeypatch.setattr(fill, 'closeups',
                        lambda rig, mass, rows, limbs, up, path: [f'limb-{i}.png' for i in range(len(limbs))])
    asked = []

    def ask(question, pictures, schema, default, route=None):
        asked.append((route, question, pictures, schema))
        if 'one arm' in question and f'dot {n["hand.L"]})' in question:           # the left arm
            return {'shoulder': n['arm.L'], 'elbow': n['fore.L'], 'hand': n['hand.L']}, {'source': 'ai', 'cost': 0.1}
        if 'one arm' in question:                                               # the right arm: out of order, twice
            return {'shoulder': n['fore.R'], 'elbow': n['arm.R'], 'hand': n['hand.R']}, {'source': 'ai'}
        if f'dot {n["toe.L"]})' in question:
            return {'hip': n['thigh.L'], 'knee': n['shin.L'], 'ankle': n['foot.L'], 'toe': None}, {'source': 'ai'}
        return {}, {'source': 'default', 'error': 'timed out'}                  # the right leg: no answer
    merged, report = fill.closer(rig, mass, rows, first, UP, 'out', ask)
    assert [(e['limb'], e['side']) for e in report] == [('arm', 'left'), ('arm', 'right')]
    assert merged['legs'] == first['legs'] and not any('one leg' in q for _, q, _, _ in asked)
    asked.clear()
    merged, report = fill.closer(rig, mass, rows, first, UP, 'out', ask, ('legs', 'arms'))
    assert merged['arms']['left'] == {'clavicle': None, 'shoulder': n['arm.L'], 'elbow': n['fore.L'],
                                      'hand': n['hand.L']}
    assert merged['arms']['right'] == first['arms']['right'] and merged['legs']['right'] == first['legs']['right']
    assert merged['legs']['left'] == {'hip': n['thigh.L'], 'knee': n['shin.L'], 'ankle': n['foot.L'], 'toe': None}
    assert merged['hips'] == first['hips'] and first['arms']['left']['clavicle'] == n['arm.L']      # the first stands
    by = {(e['limb'], e['side']): e for e in report}
    assert [a['route'] for a in by['arm', 'right']['asked']] == [fill.LIMB_ROUTE, fill.LIMB_AGAIN]
    assert 'does not lie past' in by['arm', 'right']['asked'][0]['unusable'] and 'joints' not in by['arm', 'right']
    again = next(q for r, q, _, _ in asked if r == fill.LIMB_AGAIN and 'one arm' in q)
    assert 'An earlier answer could not be used: the elbow' in again
    assert by['leg', 'right']['asked'][0]['unusable'] == 'no answer' and len(by['leg', 'right']['asked']) == 2
    arm = next(q for r, q, _, _ in asked if 'one arm' in q)
    assert 'clavicle' not in arm and 'collar' not in arm and '(someone)' in arm and 'turn' in arm
    assert all(s['required'] == ['shoulder', 'elbow', 'hand'] for _, q, _, s in asked if 'one arm' in q)
    assert by['arm', 'left']['line'][0] == n['chest'] and by['arm', 'left']['asked'][0]['cost'] == 0.1


def test_a_fits_look_goes_on_from_the_limbs_joints_and_falls_back_to_its_own(monkeypatch):
    """read(): the close-ups' joints build the body; where they cannot, or cannot be drawn, the first look's do. A
    file's own skeleton is asked nothing more."""
    rig = _figure(_bones())
    _, rows = fill.joints(rig, UP)
    n = {name: j['n'] for j in rows for name in j['names']}
    monkeypatch.setattr(fill, 'shown', lambda rig, up, out: (*fill.joints(rig, up), []))

    def ask(question, pictures, schema, default, route=None):
        return _answer(rows), {'source': 'ai'}
    closed = _answer(rows)
    closed['legs']['left'] = {'hip': n['thigh.L'], 'knee': None, 'ankle': n['foot.L'], 'toe': None}
    monkeypatch.setattr(fill, 'closer', lambda rig, mass, rows, answer, up, out, ask: (closed, [{'limb': 'leg'}]))
    made, report = fill.read(rig, UP, None, ask, settle=lambda out: (rig, out))
    assert made['legs']['left'] == {'hip': 'thigh.L', 'ankle': 'foot.L'}
    assert report['asked'][0]['limbs'] == [{'limb': 'leg'}]
    assert report['asked'][0]['answer']['legs']['left']['knee'] == n['shin.L']         # the first answer, as given
    broken = _answer(rows)
    broken['legs']['left'] = {'hip': n['thigh.L'], 'knee': None, 'ankle': n['thigh.L'], 'toe': None}
    monkeypatch.setattr(fill, 'closer', lambda rig, mass, rows, answer, up, out, ask: (broken, []))
    made, report = fill.read(rig, UP, None, ask, settle=lambda out: (rig, out))
    assert made['legs']['left']['knee'] == 'shin.L' and 'one joint' in report['asked'][0]['limbs_unusable']
    assert len(report['asked']) == 1                                                    # not asked again for it

    def fails(*a):
        raise RuntimeError('no camera')
    monkeypatch.setattr(fill, 'closer', fails)
    made, report = fill.read(rig, UP, None, ask, settle=lambda out: (rig, out))
    assert made['legs']['left']['knee'] == 'shin.L' and report['asked'][0]['limbs_failed'] == 'RuntimeError: no camera'
    made, report = fill.read(rig, UP, None, ask)                                        # its own skeleton
    assert 'limbs' not in report['asked'][0] and 'limbs_failed' not in report['asked'][0]


def test_the_limbs_ask_has_its_own_settings_and_its_answer_passes_the_asks_check():
    from rigswap import ai
    from rigswap.library import asks
    assert asks.ASKS[fill.LIMB_ROUTE].model == asks.ASKS[fill.ROUTE].model
    assert asks.ASKS[fill.LIMB_AGAIN].model == asks.ASKS[fill.AGAIN].model
    assert ai._check({'shoulder': 3, 'elbow': None, 'hand': 5}, fill._LIMB(fill.ARM[1:])) is None
    assert ai._check({'shoulder': 3, 'hand': 5}, fill._LIMB(fill.ARM[1:]))


def test_a_shoulder_seen_at_the_spine_joint_the_arm_leaves_is_kept_as_where_the_arm_begins(monkeypatch):
    """Goku's right arm (arm-1): held against his body, its own dots begin by the elbow, and on the close-up the look
    named his shoulder at the chest's joint, which is drawn where the shoulder is. The arm begins at its first dot
    and the record keeps where its shoulder was seen; a file's own skeleton has no such entry."""
    bones = _bones() + [('wrist.L', 'hand.L', [.55, 0, 1.5], 4), ('wrist.R', 'hand.R', [-.55, 0, 1.5], 4)]
    rig = _figure(bones)
    mass, rows = fill.joints(rig, UP)
    n = {name: j['n'] for j in rows for name in j['names']}
    first = _answer(rows)
    monkeypatch.setattr(fill, 'closeups',
                        lambda rig, mass, rows, limbs, up, path: [f'limb-{i}.png' for i in range(len(limbs))])

    def ask(question, pictures, schema, default, route=None):
        side = 'L' if f'dot {n["wrist.L"]})' in question else 'R'
        if side == 'L':     # seen at the chest: the arm's first dot is then its shoulder, the chest where it begins
            return {'shoulder': n['chest'], 'elbow': n['fore.L'], 'hand': n['hand.L']}, {'source': 'ai'}
        return {'shoulder': n['arm.R'], 'elbow': n['fore.R'], 'hand': n['hand.R']}, {'source': 'ai'}
    merged, report = fill.closer(rig, mass, rows, first, UP, 'out', ask)
    assert merged['arms']['left'] == {'clavicle': None, 'shoulder': n['arm.L'], 'elbow': n['fore.L'],
                                      'hand': n['hand.L'], 'begins': n['chest']}
    assert 'begins' not in merged['arms']['right']
    made = fill.entries(rig, mass, rows, merged, UP, settle=lambda out: (rig, out))
    assert made['arms']['left'] == {'shoulder': 'arm.L', 'elbow': 'fore.L', 'hand': 'hand.L', 'begins': 'chest'}
    assert 'begins' not in made['arms']['right']
    assert 'begins' not in fill.entries(rig, mass, rows, merged, UP)['arms']['left']       # a file's own skeleton
    # the elbow at the arm's first dot leaves no upper arm to begin there: the answer stands as it was given
    def tucked(question, pictures, schema, default, route=None):
        return {'shoulder': n['chest'], 'elbow': n['arm.L'], 'hand': n['hand.L']}, {'source': 'ai'}
    merged, _ = fill.closer(rig, mass, rows, first, UP, 'out', tucked)
    assert merged['arms']['left']['shoulder'] == n['chest'] and 'begins' not in merged['arms']['left']
