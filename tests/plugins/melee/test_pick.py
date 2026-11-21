import json
from types import SimpleNamespace as NS

import numpy as np

from rigswap.core import correspond
from rigswap.plugins.melee import pick

SAID = {'Mario': 'Mario', 'Kirby': 'Kirby', 'Link': 'Link', 'Young Link': 'YoungLink'}


def test_the_fighter_most_answers_give_wins_and_the_character_itself_counts_over_the_menu():
    answers = [{'is': 'none', 'fighter': 'Mario', 'why': 'a'}, {'is': 'none', 'fighter': 'Kirby', 'why': 'round'},
               {'is': 'none', 'fighter': 'Kirby', 'why': 'b'}]
    assert pick.tally(answers, SAID, {'Mario', 'Kirby'}) == ('Kirby', 'round', ['Mario', 'Kirby', 'Kirby'], False)
    # "is" names the character itself, offered or not; a fighter off the menu or unknown is no answer
    answers = [{'is': 'Young Link', 'fighter': 'Mario', 'why': 'it is him'}, None,
               {'is': 'none', 'fighter': 'Link', 'why': 'x'}]
    assert pick.tally(answers, SAID, {'Mario', 'Kirby'}) == ('YoungLink', 'it is him', ['YoungLink'], True)
    assert pick.tally([None, {'is': 'Waluigi', 'fighter': 'Waluigi'}], SAID, {'Mario'}) is None


def test_the_menu_is_the_fighters_it_comes_apart_on_least_then_the_least_stretched_first():
    rows = [dict(fighter='Link', apart=None, worst=.4), dict(fighter='Kirby', apart=None, worst=.7),
            dict(fighter='Samus', apart={'gap': .2, 'parts': ['foot', 'leg']}, worst=.1),
            dict(fighter='Mario', failed='ValueError: no arms')]
    # held together on two, those two: however far a part is stretched
    assert [r['fighter'] for r in pick.menu(rows)] == ['Link', 'Kirby']
    # apart on all (Kamek): those within APART of the least
    kamek = [dict(fighter='Bowser', apart={'gap': .32, 'parts': ['leg', 'trunk']}, worst=.1),
             dict(fighter='Mario', apart={'gap': .1, 'parts': ['leg', 'trunk']}, worst=.3),
             dict(fighter='Ness', apart={'gap': .08, 'parts': ['leg', 'trunk']}, worst=.1)]
    assert [r['fighter'] for r in pick.menu(kamek)] == ['Ness', 'Mario']


def test_a_skeleton_that_is_a_fighters_own_is_that_fighter():
    fighter = NS(names=['JOBJ_0', 'JOBJ_1', 'JOBJ_2'], parents=np.array([-1, 0, 1]))
    other = NS(names=['JOBJ_0', 'JOBJ_1', 'JOBJ_2'], parents=np.array([-1, 0, 0]))
    fighters = {'Mario': (fighter, 1.), 'Luigi': (other, 1.)}
    # hung off a node of its own, the root under anything
    rig = NS(names=['Armature', 'JOBJ_0', 'JOBJ_1', 'JOBJ_2'], parents=np.array([-1, 0, 1, 2]))
    assert pick.own(rig, fighters) == ['Mario']
    assert pick.own(NS(names=['hips', 'spine'], parents=np.array([-1, 0])), fighters) == []


def _chain(*bones):
    return NS(bones=list(bones), anchors=[0, len(bones) - 1, None])


def test_a_parts_stretch_is_over_the_bones_carrying_the_model_and_its_ends_keep_their_size():
    # bones: 0 hips, 1 chest, 2 head (an end); 3 upper arm, 4 hand (an end); 5 thigh, 6 foot (an end)
    rest = {0: ([0, 0, 0], [0, 0, 2]), 1: ([0, 0, 2], [0, 0, 4]), 2: ([0, 0, 4], [0, 0, 5]),
            3: ([0, 0, 4], [3, 0, 4]), 4: ([3, 0, 4], [4, 0, 4]), 5: ([0, 0, 0], [0, 0, -2]),
            6: ([0, 0, -2], [0, 0, -3])}
    embedded = {0: ([0, 0, 0], [0, 0, 1]), 1: ([0, 0, 1], [0, 0, 2]), 2: ([0, 0, 2], [0, 0, 3]),
                3: ([0, 0, 2], [1, 0, 2]), 4: ([1, 0, 2], [2, 0, 2]), 5: ([0, 0, 0], [0, 0, -1]),
                6: ([0, 0, -1], [0, 0, -2])}
    m = NS(embedded=embedded, rest=rest, ends={2, 4, 6}, placed={}, owner={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6})
    rig = NS(points=np.zeros((6, 3)), weights=np.eye(6))        # model bone n carries vertex n
    fch = {'spine': _chain(0, 1, 2), 'left arm': _chain(3, 4), 'right arm': _chain(3, 4), 'left leg': _chain(5, 6),
           'right leg': _chain(5, 6)}
    # the trunk twice its length, the arm three times; the thigh carries nothing, the foot (an end) the model's
    assert pick.stretched(rig, m, fch) == {'trunk': 2., 'arms': 3.}
    m.owner = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2}              # nothing on the leg at all
    assert pick.stretched(rig, m, fch) == {'trunk': 2., 'arms': 3., 'legs': None}
    assert abs(pick.worst({'trunk': 2., 'arms': .25, 'legs': None}) - np.log(4)) < 1e-9
    # a bone at a point (Fox's hips, nought long at rest and in the model) is a pivot: it says nothing
    m.rest[0], m.embedded[0] = ([0, 0, 0], [0, 0, 0]), ([0, 0, 0], [0, 0, 0])
    m.embedded[1] = ([0, 0, 0], [0, 0, 2])
    assert pick.stretched(rig, m, fch)['trunk'] == 1.


def test_how_tall_the_model_stands_on_a_fighter_is_said_where_it_is_far_off():
    model, fighter = NS(points=np.array([[0, 0, 0], [0, 0, 30.]])), NS(points=np.array([[0, 0, 1], [0, 0, 16.]]))
    assert pick.stands(model, fighter) == 2.
    row = dict(apart=None, stands=2., stretch={'arms': .9}, held='')
    assert pick._said(row) == ["it stands 2.0x the fighter's height", 'arms 0.9x']
    assert pick._said(dict(row, stands=1.1)) == ['arms 0.9x']
    # and orders the menu, as a stretch does
    rows = [dict(fighter='CaptainFalcon', apart=None, stands=2., worst=.1), dict(fighter='Fox', apart=None, worst=.2)]
    assert [r['fighter'] for r in pick.menu(rows)] == ['Fox', 'CaptainFalcon']


def test_what_becomes_of_a_held_prop_is_said_per_fighter(monkeypatch):
    sword = {'kind': 'sword', 'hand': 'left_hand'}
    shield = {'kind': 'shield', 'hand': 'right_hand'}
    monkeypatch.setattr(correspond, 'props', lambda rig, body: [(1, [1], None, 'left_hand', True)])
    m = NS(replaced=[dict(sword, item='spear')])
    assert pick.held(None, None, m, [sword]) == 'what it holds takes the place of its sword'
    m.replaced = []
    assert pick.held(None, None, m, [sword, shield]) == ('what it holds takes the place of its sword or shield, '
                                                         'whichever it is')
    assert pick.held(None, None, m, []) == 'what it holds is left off: it fights empty-handed'
    monkeypatch.setattr(correspond, 'props', lambda rig, body: [])
    assert pick.held(None, None, m, [sword, shield]) == 'it keeps its own sword and shield'
    assert pick.held(None, None, m, []) == ''


def _targets(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, body, has in (('Mario', 'biped', []), ('Kirby', 'ball', []), ('Bowser', 'biped', ['tail'])):
        (folder / f'{name}.description.json').write_text(json.dumps({'body_type': body, 'has': has}))
    return folder


def test_the_ask_sees_every_fighter_read_with_what_the_swap_does_there_and_its_answer_is_the_pick(monkeypatch,
                                                                                                  tmp_path):
    targets = _targets(tmp_path / 'targets')
    fighters = {n: (NS(names=[n], parents=np.array([-1])), 1.) for n in ('Mario', 'Kirby', 'Bowser')}
    rows = [dict(fighter='Mario', scale=1., apart={'gap': .3, 'parts': ['foot', 'trunk']},
                 stretch={'trunk': 1., 'arms': .9}, worst=.1, held=''),
            dict(fighter='Kirby', scale=1., apart=None, stretch={'arms': .5}, worst=.69,
                 held='what it holds is left off: it fights empty-handed'),
            dict(fighter='Bowser', failed='ValueError: no arms')]
    monkeypatch.setattr(pick, 'fit', lambda *args: rows)
    monkeypatch.setattr(pick, 'draw', lambda rig, up, front, path: path)
    monkeypatch.setattr(pick, 'sheet', lambda targets, names, path: path)
    asked = []

    def ask(question, pictures, schema, default, route):
        asked.append((question, schema))
        return {'is': 'none', 'fighter': 'Kirby', 'why': 'round'}, {'source': 'ai'}
    rig = NS(names=['hips'], parents=np.array([-1]))
    name, said = pick.fighter(rig, None, None, {}, {}, set(), fighters, targets, tmp_path, ask,
                              'Waddle Dee.zip', 'Bandana Waddle Dee')
    assert name == 'Kirby' and said['why'] == 'round' and said['votes'] == ['Kirby'] * 3
    assert [o['fighter'] for o in said['offered']] == ['Kirby'] and 'Bowser' in said['unread']
    assert [o['fighter'] for o in said['apart']] == ['Mario']
    question, schema = asked[0]
    assert 'Waddle Dee.zip' in question and 'Bandana Waddle Dee' in question
    assert 'Kirby (a round body): arms 0.5x, what it holds is left off: it fights empty-handed' in question
    assert 'Mario (two legs)' not in question
    assert schema['properties']['fighter']['enum'] == ['Kirby']
    assert set(schema['properties']['is']['enum']) == {'Mario', 'Kirby', 'Bowser', 'none'}
    # apart on every one, the one it comes apart on least is offered with what comes apart, and how far
    rows[1] = dict(rows[1], apart={'gap': .1, 'parts': ['arm', 'head']})
    asked.clear()
    pick.fighter(rig, None, None, {}, {}, set(), fighters, targets, tmp_path, ask)
    assert "Kirby (a round body): its arm and head come apart, 10% of the fighter's height, arms 0.5x" in asked[0][0]
    assert "Mario (two legs): its foot and trunk come apart, 30% of the fighter's height" not in asked[0][0]


def test_a_skeleton_its_own_goes_on_that_fighter_unasked(tmp_path):
    fighters = {'Mario': (NS(names=['JOBJ_0'], parents=np.array([-1])), 1.)}
    rig = NS(names=['JOBJ_0'], parents=np.array([-1]))

    def ask(*args, **kwargs):
        raise AssertionError('asked')
    name, said = pick.fighter(rig, None, None, {}, {}, set(), fighters, tmp_path, tmp_path, ask)
    assert name == 'Mario' and said['own'] is True
