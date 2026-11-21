"""targets/weapons.py: a fighter's weapon slots, measured on its vanilla costume."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee import dae as melee_dae
from rigswap.plugins.melee.facts import fighters
from rigswap.plugins.melee.targets import weapons as melee_weapons


def _sword_on_a_hand(hand_role='left_hand', off=0.0):
    """A vanilla costume as read_dae reads it, in the .dae's space (y up, facing +z): a hand joint JOBJ_1 and under it
    a sword joint JOBJ_2, the sword a thin box from just behind the hand (z -1.6) to the front (z 9), its faces in
    places 71 and 73 (Link's sword's), wholly on JOBJ_2; and the bone graph of the same skeleton in the target's frame,
    JOBJ_2 ``off`` from where the .dae has it."""
    hand, sword = np.array([6.5, 12.1, 0.0]), np.array([7.4, 11.8, 1.4])
    joints = {}
    for name, at in (('JOBJ_0', np.zeros(3)), ('JOBJ_1', hand), ('JOBJ_2', sword)):
        joints[name] = np.eye(4)
        joints[name][:3, 3] = at
    corners = np.array([(x, y, z) for z in (-1.6, 9.0) for y in (11.7, 11.9) for x in (7.2, 7.6)])
    faces = [[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
             [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    model = melee_dae.Model(joints=joints, parents={'JOBJ_0': None, 'JOBJ_1': 'JOBJ_0', 'JOBJ_2': 'JOBJ_1'},
                        points=corners, weights=[{'JOBJ_2': 1.0} for _ in corners], faces=faces, triangles=12,
                        face_objects=['Joint_2_Object_71_SINGLE'] * 6 + ['Joint_2_Object_73_SINGLE'] * 6)
    heads = {n: melee_weapons.TO_TARGET @ m[:3, 3] for n, m in joints.items()}
    heads['JOBJ_2'] = heads['JOBJ_2'] + [off, 0, 0]
    bones = [{'n': k + 1, 'name': n, 'parent': k or None, 'head': heads[n].tolist()} for k, n in enumerate(joints)]
    return model, bones, {1: 'none', 2: hand_role, 3: 'none'}


def test_a_weapon_slot_is_its_places_long_axis_from_the_end_by_its_hand(monkeypatch):
    """Link's sword as the vanilla costume has it, in little: the slot is in the hand its joint hangs under, its grip
    the middle of the box's end nearer that hand's bone, its tip straight out to the front (+z in the .dae is -y in
    the target's frame), its triangles those of its places. (The little costume draws no shield: Link's sword alone.)"""
    monkeypatch.setitem(melee_weapons.WEAPONS, 'Link', {'sword': melee_weapons.WEAPONS['Link']['sword']})
    model, bones, roles = _sword_on_a_hand()
    (slot,) = melee_weapons.weapon_slots(fighters.fighter('Link'), bones, roles, model)
    assert (slot['part'], slot['kind'], slot['hand'], slot['hand_bone'], slot['joints'], slot['bone']) == (
        'sword', 'sword', 'left_hand', 'JOBJ_1', ['JOBJ_2'], 'JOBJ_2')
    assert np.allclose(slot['grip'], [7.4, 1.6, 11.8]) and np.allclose(slot['tip_direction'], [0, -1, 0])
    assert np.allclose(slot['flat'], [1, 0, 0]) and slot['width'] == pytest.approx(0.4)
    assert slot['length'] == pytest.approx(10.6) and slot['places'] == [71, 72, 73, 74, 75]
    assert slot['triangles'] == 12 and 'nearer the head of the hand bone' in slot['method']
    # the roles as a description keeps them, keyed by the bone number as text: the same slot
    assert melee_weapons.weapon_slots(fighters.fighter('Link'), bones, {str(n): r for n, r in roles.items()},
                                      model) == [slot]
    _, bones, roles = _sword_on_a_hand(hand_role='none')
    with pytest.raises(Failed, match='not one hand'):
        melee_weapons.weapon_slots(fighters.fighter('Link'), bones, roles, model)
    _, bones, roles = _sword_on_a_hand(off=0.01)
    with pytest.raises(Failed, match='its bone graph has it'):
        melee_weapons.weapon_slots(fighters.fighter('Link'), bones, roles, model)


@pytest.mark.plugin
@pytest.mark.skipif(not ((Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlLkNr.dat').is_file()
                         and (Path(os.environ.get(melee.TARGETS_VAR, '.')) / 'Link.bones.json').is_file()),
                    reason=f'needs vanilla files in {melee.FILES_VAR} and the targets in {melee.TARGETS_VAR}')
def test_links_sword_slot_is_measured_on_the_vanilla_costume(tmp_path):
    """runs/pilot3/barbarian-Link: Melee's Link is left-handed. His sword (places 71-75, on joints 25 and 26 under
    LHandN, joint 23) is out to the front from his left fist, its grip end behind the hand; his shield (places 76-82,
    on joint 67) is on the right hand, a slot the game does not swing. Mario has none."""
    targets = Path(os.environ[melee.TARGETS_VAR])
    bones = json.loads((targets / 'Link.bones.json').read_text(encoding='utf-8'))['bones']
    roles = {int(n): r for n, r in json.loads((targets / 'Link.description.json').read_text(encoding='utf-8'))
             ['bones']['roles'].items()}
    record = melee_weapons.write_weapons(fighters.fighter('Link'), bones, roles, tmp_path / 'Link.glb')
    slot, shield = melee_weapons.read_weapons(tmp_path / 'Link.glb')
    assert (shield['part'], shield['kind'], shield['swung'], shield['hand'], shield['hand_bone'], shield['bone']) == (
        'shield', 'shield', False, 'right_hand', 'JOBJ_55', 'JOBJ_67')
    assert shield['triangles'] == 180 and slot['swung']
    assert record['weapons'] == [slot, shield] and record['from'] == 'PlLkNr.dat: every DObj, vanilla/Link/every/Link.dae'
    assert (slot['kind'], slot['hand'], slot['hand_bone'], slot['joints']) == (
        'sword', 'left_hand', 'JOBJ_23', ['JOBJ_25', 'JOBJ_26'])
    # most of the blade, hilt and guard is on joint 25, the root of the sword's two joints; the tip end on 26
    assert slot['bone'] == 'JOBJ_25' and slot['bone_weights'] == {'JOBJ_25': 236.0, 'JOBJ_26': 80.0}
    assert slot['triangles'] == 287 and slot['places'] == [71, 72, 73, 74, 75]
    assert np.allclose(slot['tip_direction'], [0, -1, 0], atol=0.03)               # out to the front
    assert np.allclose(slot['flat'], [1, 0, 0], atol=0.03)                          # its width across, its flat level
    hand = next(b['head'] for b in bones if b['name'] == 'JOBJ_23')
    assert slot['grip'][1] > hand[1] and slot['hand_to_grip'] < slot['length'] / 4   # just behind the fist
    assert melee_weapons.write_weapons(fighters.fighter('Mario'), [], {}, tmp_path / 'Mario.glb', files_dir=tmp_path) == {
        'fighter': 'Mario', 'frame': "the target's: +z up, facing -y, as Blender imports its .glb", 'weapons': []}
