"""facts/parts.py: the vanilla parts a costume keeps, and the weapons among them."""
import json

import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee.facts import fighters, parts as facts_parts


def test_the_weapons_are_the_parts_each_fighter_fights_with():
    fights = {name: {part for part, _, why in rows if why.startswith('the fighter fights with it')}
              for name, rows in facts_parts.PARTS.items()}
    assert {n: set(w) for n, w in facts_parts.WEAPONS.items()} == {n: p for n, p in fights.items() if p}


def test_every_weapon_names_its_other_forms_and_each_goes_with_it():
    """Every slot kind keeps to one rule: a weapon names its forms (none when it has none), each a part of the same
    fighter that goes with it ("it goes with the sword"): the weapon put away and the scabbard it is put away in; a
    fighter's part that goes with a weapon is a form of that weapon."""
    for name, weapons in facts_parts.WEAPONS.items():
        rows = {part: why for part, _, why in facts_parts.PARTS[name]}
        for part, weapon in weapons.items():
            assert set(weapon) == {'kind', 'forms', 'swung'}
            goes = {r for r, why in rows.items() if r != part and why.endswith(f'it goes with the {part}')}
            assert set(weapon['forms']) == goes
    assert 'scabbard' in facts_parts.WEAPONS['Marth']['sword']['forms']
    # the game swings each but the shield, which Link and Young Link carry on the arm
    assert {(n, part) for n, weapons in facts_parts.WEAPONS.items() for part, w in weapons.items()
            if not w['swung']} == {('Link', 'shield'), ('YoungLink', 'shield')}


def test_the_parts_triangles_add_up_to_the_fighters():
    assert set(facts_parts.PART_TRIANGLES) == {n for n, rows in facts_parts.PARTS.items() if rows}
    for name, rows in facts_parts.PARTS.items():
        if rows:
            assert set(facts_parts.PART_TRIANGLES[name]) == {part for part, _, _ in rows}
    # as measured whole
    assert facts_parts.KEPT_TRIANGLES == {'Peach': 1161, 'Bowser': 2070, 'Yoshi': 2224, 'Ganondorf': 282,
                                          'Kirby': 2064, 'Samus': 2966, 'Sheik': 102, 'Link': 917, 'YoungLink': 912,
                                          'GameAndWatch': 1752, 'Marth': 400, 'Roy': 501, 'Popo': 166, 'Nana': 166}


@pytest.mark.plugin
@pytest.mark.parametrize('fighter', [f for f in fighters.FIGHTERS if f.keep], ids=lambda f: f.name)
def test_a_kept_option_is_kept_whole_with_its_low_form(fighter):
    """An option is one object, and the game shows its low-detail form far off and in the magnifier (ftParts_80074B6C
    gives both sets one index): a kept option's places in both are kept. The body's own option 0 is the build's, though
    a kept place may sit in it. Read from the fighter's file."""
    import os
    from pathlib import Path
    from rigswap.plugins import melee
    from rigswap.plugins.melee.export import dobjs as melee_dobjs
    path = Path(os.environ.get(melee.FILES_VAR, '.')) / f'Pl{fighter.code}.dat'
    if not path.is_file():
        pytest.skip(f'needs {path}')
    options = melee_dobjs.fighter_tables(path).options
    keep = set(fighter.keep)
    for (detail, model, option), places in options.items():
        if detail in (melee_dobjs.HIGH_DETAIL, melee_dobjs.LOW_DETAIL) and (model, option) != (0, 0) \
                and keep.intersection(places):
            whole = {*options.get((melee_dobjs.HIGH_DETAIL, model, option), ()),
                     *options.get((melee_dobjs.LOW_DETAIL, model, option), ())}
            assert whole <= keep, (fighter.name, detail, model, option, sorted(whole - keep))


@pytest.mark.plugin
@pytest.mark.parametrize('name', ['Link', 'YoungLink', 'Marth', 'Roy', 'Peach', 'Ganondorf', 'GameAndWatch'])
def test_the_parts_triangles_are_measured_from_the_vanilla_costume(name):
    """Each part's triangles, against its places' in the vanilla costume exported with every DObj by place, which
    write_weapons leaves in the targets folder (vanilla/<Name>/every/<Name>.dae); skipped when it is not there."""
    import os
    from pathlib import Path
    from rigswap.plugins import melee
    from rigswap.plugins.melee.dae import PLACE, read_dae
    dae = Path(os.environ.get(melee.TARGETS_VAR, '.')) / 'vanilla' / name / 'every' / f'{name}.dae'
    if not dae.is_file():
        pytest.skip(f'needs {dae}')
    model = read_dae(dae)
    per = {}
    for o in model.face_objects:
        if (m := PLACE.fullmatch(o or '')):
            per[int(m[1])] = per.get(int(m[1]), 0) + 1
    measured = {part: sum(per.get(p, 0) for p in places) for part, places, _ in facts_parts.PARTS[name]}
    assert measured == facts_parts.PART_TRIANGLES[name]


LINK_SWORD = {'part': 'sword', 'kind': 'sword', 'hand': 'left_hand', 'places': [71, 72, 73, 74, 75], 'triangles': 287,
              'item': 'sword', 'pieces': [19, 21], 'item_hand': 'right_hand', 'hand_changed': True}


def test_a_weapon_whose_slot_a_held_item_took_is_not_kept_nor_its_other_forms(tmp_path):
    """The build put the Barbarian's sword into Link's sword slot: Link's own sword is not kept, nor his sword put
    away in its scabbard (places 93, 94), which the game shows in place of the one in his hand, nor the scabbard (83 to 85),
    which goes with it; his shield and the shield put away are, until a model's shield takes the shield's slot. A slot
    the fighter does not keep fails."""
    link = fighters.fighter('Link')
    assert facts_parts.kept_places(link, tmp_path) == (link.keep, [])
    (tmp_path / 'weapons_replaced.json').write_text(json.dumps([LINK_SWORD]), encoding='utf-8')
    keep, replaced = facts_parts.kept_places(link, tmp_path)
    assert replaced == [LINK_SWORD]
    assert keep == tuple(p for p in link.keep if p not in (*range(71, 76), 83, 84, 85, 93, 94))
    assert {76, 82, 89} <= set(keep)
    assert facts_parts.hidden_forms('Link', replaced) == ('sword_stowed', 'scabbard')
    assert facts_parts.not_kept('Link', replaced) == {'sword': {'places': [71, 72, 73, 74, 75], 'triangles': 287},
                                                      'sword_stowed': {'places': [93, 94], 'triangles': 217},
                                                      'scabbard': {'places': [83, 84, 85], 'triangles': 53}}
    # a model's own shield took his shield's slot as well: his shield goes, and the shield put away (86 to 92)
    shield = dict(part='shield', kind='shield', hand='right_hand', places=list(range(76, 83)), item='TateN')
    (tmp_path / 'weapons_replaced.json').write_text(json.dumps([LINK_SWORD, shield]), encoding='utf-8')
    keep, _ = facts_parts.kept_places(link, tmp_path)
    assert keep == tuple(p for p in link.keep if p not in range(71, 95))
    assert facts_parts.hidden_forms('Link', [shield]) == ('shield_stowed',)
    (tmp_path / 'weapons_replaced.json').write_text(json.dumps([dict(LINK_SWORD, places=[1, 2])]), encoding='utf-8')
    with pytest.raises(Failed, match='not among its kept places'):
        facts_parts.kept_places(link, tmp_path)
