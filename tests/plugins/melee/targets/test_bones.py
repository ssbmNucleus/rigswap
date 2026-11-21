"""targets/bones.py: the Melee fighters described from the game's own bone tables. The tests at the end need the game
files (RIGSWAP_MELEE_FILES) or the prepared targets (RIGSWAP_MELEE_TARGETS) and are skipped without them."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import types

import pytest

from rigswap.plugins.melee.targets import roles as describe_bones
from rigswap.plugins import melee
from rigswap.plugins.melee.facts import fighters
from rigswap.plugins.melee.targets import bones as melee_bones
from rigswap.vocabulary import BODY_TYPES, Description
import numpy as np

# Mario's joint tree as his vanilla .glb has it (2026-10-10): each joint's parent (-1: none), the joints the costume
# skins vertices to, and each joint's head, upright (+z up, facing -y), to two places.
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 13, 10, 15, 10, 17, 10, 19, 9, 5, 22, 23, 23, 23, 23, 23, 5,
           29, 30, 31, 32, 33, 34, 33, 36, 33, 38, 33, 40, 33, 33, 43, 32, 4, 4, 47, 48, 49, 50, 51, 4, 53, 54, 55, 56,
           57, 3, 0]
SKINNED = {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58}
HEADS = [[0, 0, 0], [0, 0, 0], [0, 0, 5.67], [0, 0, 5.67], [0, 0, 4.99], [0, -0.03, 5.97], [0.86, 0.18, 7.93],
         [1.93, 0.19, 7.92], [1.93, 0.19, 7.92], [3.3, 0.2, 7.92], [4.87, 0.17, 7.93], [6.21, -0.35, 8.24],
         [6.74, -0.36, 8.24], [6.27, 0.03, 8.34], [6.77, 0.03, 8.3], [6.21, 0.43, 8.32], [6.72, 0.42, 8.3],
         [6.15, 0.78, 8.24], [6.62, 0.78, 8.22], [5.72, -0.73, 7.68], [6.21, -0.89, 7.67], [4.97, 0.17, 7.93],
         [0, 0.22, 8.29], [0, 0.04, 9.31], [0, -0.34, 11.98], [-0.02, -1.88, 9.63], [-0.02, -0.1, 9.65],
         [-0.04, -1.8, 9.98], [-0.03, -1.88, 9.6], [-0.86, 0.18, 7.93], [-1.93, 0.19, 7.92], [-1.93, 0.19, 7.92],
         [-3.28, 0.21, 7.91], [-4.86, 0.2, 7.95], [-6.2, -0.35, 8.28], [-6.72, -0.36, 8.23], [-6.23, 0.02, 8.36],
         [-6.77, 0.03, 8.31], [-6.22, 0.43, 8.29], [-6.73, 0.42, 8.28], [-6.18, 0.78, 8.21], [-6.64, 0.79, 8.19],
         [-6.21, 0.19, 7.8], [-5.76, -0.72, 7.67], [-6.21, -0.9, 7.67], [-4.89, 0.2, 7.9], [0, -0.03, 5.97],
         [1.12, -0.03, 5.0], [1.12, -0.03, 5.0], [1.13, -0.01, 3.09], [1.15, 0.07, 0.25], [1.15, 0.07, 0.25],
         [1.15, -1.18, 0.72], [-1.12, -0.03, 5.0], [-1.12, -0.03, 5.0], [-1.14, 0.01, 3.09], [-1.16, 0.12, 0.25],
         [-1.16, 0.12, 0.25], [-1.17, -1.13, 0.72], [0, 0, 0], [0, 0, 0]]


def mario_bones():
    """Mario's bone graph as bones.json has it: bone n is JOBJ_(n-1)."""
    children = {j: [] for j in range(len(PARENTS))}
    for j, p in enumerate(PARENTS):
        if p >= 0:
            children[p].append(j + 1)
    return [{'n': j + 1, 'name': f'JOBJ_{j}', 'parent': p + 1 if p >= 0 else None, 'children': children[j],
             'head': HEADS[j], 'tail': None, 'vertices': 10 if j in SKINNED else 0} for j, p in enumerate(PARENTS)]


def by_joint(roles):
    """{joint number: role} from {bone number: role}."""
    return {int(n) - 1: r for n, r in roles.items()}


# ---- the tables ---------------------------------------------------------------------------------------------------

def test_every_slot_has_a_role_from_the_vocabulary_for_each_body_type():
    for body, table in melee_bones.ROLES.items():
        assert set(table) == set(melee_bones.SLOTS), body
        assert set(table.values()) <= set(describe_bones.roles(body)), body
    assert len(melee_bones.SLOTS) == 54 == len(set(melee_bones.SLOTS))


def test_the_biped_table_names_the_parts():
    t = melee_bones.ROLES['biped']
    assert [t[n] for n in ('HipN', 'WaistN', 'WaistNb', 'BustN', 'NeckN', 'HeadN')] == \
        ['hips', 'spine', 'spine', 'chest', 'neck', 'head']
    assert [t[n] for n in ('LShoulderN', 'LShoulderJ', 'LArmJ', 'LHandN', 'LLegJ', 'LKneeJ', 'LFootJ')] == \
        ['left_shoulder', 'left_upper_arm', 'left_forearm', 'left_hand', 'left_thigh', 'left_shin', 'left_foot']
    assert [t[n] for n in ('RShoulderN', 'RShoulderJ', 'RArmJ', 'RHandN', 'RLegJ', 'RKneeJ', 'RFootJ')] == \
        ['right_shoulder', 'right_upper_arm', 'right_forearm', 'right_hand', 'right_thigh', 'right_shin', 'right_foot']
    assert {t[n] for n in ('L1stNa', 'R4thNb', 'LThumbNb')} == {'finger'}
    assert {t[n] for n in ('TopN', 'TransN', 'XRotN', 'YRotN', 'ThrowN', 'TransN2', 'LLegJA', 'LFootJA',
                           'LShoulderJA', 'LHaveN', 'RHaveN')} == {'none'}
    b = melee_bones.ROLES['ball']
    assert [b[n] for n in ('WaistN', 'LShoulderJ', 'LArmJ', 'LFootJ', 'LKneeJ', 'HipN')] == \
        ['body', 'left_stub', 'left_stub', 'left_foot', 'none', 'none']


def test_every_target_fighter_has_its_table_and_a_body_type():
    names = [f.name for f in fighters.in_scope()]
    assert sorted(melee_bones.JOINT_NAMES) == sorted(names)
    for name, table in melee_bones.JOINT_NAMES.items():
        assert set(table.values()) <= set(melee_bones.SLOTS)
        assert len(set(table.values())) == len(table), name
        assert melee_bones.body_type(name) == ('ball' if name in ('Kirby', 'Jigglypuff') else 'biped')
    assert melee_bones.JOINT_NAMES['Mario'][4] == 'HipN' and melee_bones.JOINT_NAMES['Kirby'][36] == 'LFootJ'


# ---- Mario, from the tables ---------------------------------------------------------------------------------------

def test_marios_roles_give_the_body_and_both_full_arms_and_legs():
    roles = by_joint(melee_bones.joint_roles('Mario', mario_bones()))
    assert [roles[j] for j in (4, 5, 22, 23)] == ['hips', 'spine', 'neck', 'head']
    assert [roles[j] for j in (6, 8, 9, 10)] == ['left_shoulder', 'left_upper_arm', 'left_forearm', 'left_hand']
    assert [roles[j] for j in (29, 31, 32, 33)] == ['right_shoulder', 'right_upper_arm', 'right_forearm',
                                                    'right_hand']
    assert [roles[j] for j in (48, 49, 51)] == ['left_thigh', 'left_shin', 'left_foot']
    assert [roles[j] for j in (54, 55, 57)] == ['right_thigh', 'right_shin', 'right_foot']
    given = set(roles.values())
    # WaistN is the spine, and Mario has no BustN: no bone is the chest (see the module's docstring)
    assert [p for p in BODY_TYPES['biped'] if p not in given] == ['chest']
    # his own joints: the toes and the hat take their part; the helper beside WaistN under HipN is the hips
    assert (roles[52], roles[58], roles[24], roles[46]) == ('left_foot', 'right_foot', 'head', 'hips')
    # the adjust joints, the item joint and the root's are none
    assert {roles[j] for j in (0, 1, 2, 3, 7, 30, 42, 47, 50, 59, 60)} == {'none'}


def test_a_joint_the_table_does_not_name_is_its_parts_when_skinned_else_none():
    bones = mario_bones()
    for b in bones:
        if b['name'] == 'JOBJ_21':
            b['vertices'] = 0
    roles = by_joint(melee_bones.joint_roles('Mario', bones))
    assert roles[21] == 'none' and roles[45] == 'right_forearm'


def test_marios_description_has_round_3s_shape_and_loads_back(tmp_path):
    bones = mario_bones()
    desc = melee_bones.description('Mario', bones)
    roles = melee_bones.joint_roles('Mario', bones)
    assert len(desc.chains) == 10 and {c.kind for c in desc.chains} == {'finger'}
    assert {r: len(c) for r, c in desc.details['fingers'].items()} == {'left_hand': 5, 'right_hand': 5}
    assert (desc.pieces, desc.from_default, desc.up, desc.front, desc.model) == ([], [], '+z', '-y', 'Mario.glb')
    desc.save(tmp_path / 'Mario.description.json')
    assert asdict(Description.load(tmp_path / 'Mario.description.json')) == asdict(desc)


def test_marios_joints_are_at_his_bones_heads_and_his_toes_at_his_shoes_fronts():
    """The ankles at the foot bones' heads, as the tables say; each toe at the front of its foot's mesh (fit's rule,
    foot_fronts), not at the toe joint's head halfway along the shoe; no toes without the mesh."""
    bones = mario_bones()
    joints = melee_bones.description('Mario', bones).joints
    assert joints['hips'].pos == HEADS[4] and joints['neck'].pos == HEADS[22]
    assert joints['left_shoulder'].pos == HEADS[8] and joints['left_elbow'].pos == HEADS[9]
    assert joints['left_wrist'].pos == HEADS[10] and joints['right_knee'].pos == HEADS[55]
    assert joints['left_ankle'].pos == HEADS[51] and 'left_toe' not in joints
    assert 'chest' not in joints and all(j.sure for j in joints.values())
    # a heel and a shoe front on each foot joint, a toe cap on each toe joint, and a shin vertex reaching forward
    rows = [((1.15, 0.5, 0.2), 51), ((1.15, -2.4, 0.3), 51), ((1.15, -2.49, 0.25), 52), ((1.13, -3.0, 2.5), 49),
            ((-1.16, 0.5, 0.2), 57), ((-1.17, -2.45, 0.3), 58)]
    points = np.array([p for p, _ in rows], float)
    weights = np.zeros((len(rows), len(bones)))
    for i, (_, j) in enumerate(rows):
        weights[i, j] = 1.0
    joints = melee_bones.description('Mario', bones, mesh=(points, weights)).joints
    assert joints['left_toe'].pos == [1.15, -2.49, 0.25] and joints['right_toe'].pos == [-1.17, -2.45, 0.3]
    assert joints['left_ankle'].pos == HEADS[51]


def test_compare_counts_agreements_and_lists_the_rest(tmp_path):
    bones = mario_bones()
    (tmp_path / 'Mario.bones.json').write_text(json.dumps({'bones': bones}), encoding='utf-8')
    desc = melee_bones.description('Mario', bones)
    desc.save(tmp_path / 'Mario.description.json')
    desc.bones = dict(desc.bones, roles=dict(desc.bones['roles'], **{'5': 'chest', '24': "can't tell"}))
    desc.save(tmp_path / 'Mario.description.ai.json')
    got = melee_bones.compare(tmp_path / 'Mario.description.json', tmp_path / 'Mario.description.ai.json')
    assert (got['agree'], got['differ'], got['ai_cant_tell']) == (59, 1, 1)
    assert got['differences'] == [{'joint': 4, 'name': 'HipN', 'tables': 'hips', 'ai': 'chest'},
                                  {'joint': 23, 'name': 'HeadN', 'tables': 'head', 'ai': "can't tell"}]


# ---- against the game's files and the prepared targets ------------------------------------------------------------

FILES = Path(os.environ.get(melee.FILES_VAR, '.'))
TARGETS = Path(os.environ.get(melee.TARGETS_VAR, '.'))


@pytest.mark.plugin
@pytest.mark.skipif(not (FILES / 'PlCo.dat').is_file(), reason=f'needs the vanilla game files in {melee.FILES_VAR}')
def test_the_joint_names_are_plco_dats():
    tables = melee_bones.bone_tables(FILES / 'PlCo.dat')
    for name, names in melee_bones.JOINT_NAMES.items():
        assert tables[name]['names'] == names, name
    assert tables['Kirby']['skipped'] == [7, 8, 9, 10, 15, 16, 28, 29, 30, 31, 32, 33, 34]
    assert (tables['Kirby']['joints'], tables['Link']['joints'], tables['Mario']['joints']) == (46, 75, 61)


@pytest.mark.plugin
@pytest.mark.skipif(not (TARGETS / 'Mario.glb').is_file(), reason=f'needs the prepared targets in {melee.TARGETS_VAR}')
def test_describe_from_tables_writes_beside_the_glb_and_keeps_the_ais(tmp_path):
    shutil.copy(TARGETS / 'Mario.glb', tmp_path / 'Mario.glb')
    (tmp_path / 'Mario.description.json').write_text(json.dumps({'body_type': 'biped', 'up': '+z', 'front': '-y',
                                                                 'bones': {'graph': 'describe/x/bones.json'}}))
    desc = melee_bones.describe_from_tables('Mario', tmp_path)
    assert json.loads((tmp_path / 'Mario.description.ai.json').read_text())['bones']['graph'] == 'describe/x/bones.json'
    back = Description.load(tmp_path / 'Mario.description.json')
    assert asdict(back) == asdict(desc) and back.bones['graph'] == 'Mario.bones.json'
    graph = json.loads((tmp_path / 'Mario.bones.json').read_text())['bones']
    assert [b['name'] for b in graph] == [f'JOBJ_{j}' for j in range(61)]
    assert by_joint(back.bones['roles']) == by_joint(melee_bones.joint_roles('Mario', graph))
    # the ankle at the foot joint's head, the toe at the front of the shoe, well ahead of the toe joint's head
    toe, ankle = back.joints['left_toe'].pos, back.joints['left_ankle'].pos
    assert ankle == pytest.approx(HEADS[51], abs=0.01) and ankle[1] - toe[1] > 2.0 > ankle[1] - HEADS[52][1]
    melee_bones.describe_from_tables('Mario', tmp_path)         # again: the tables' own is not kept as the AI's
    assert json.loads((tmp_path / 'Mario.description.ai.json').read_text())['bones']['graph'] == 'describe/x/bones.json'


def test_each_hands_item_joint_is_named_by_the_fighters_bone_table():
    """The item joints (LHaveN, RHaveN), where the game carries what a hand holds, from JOINT_NAMES: Yoshi's right
    hand's is joint 65, under his hand joint 58; Ganondorf's is the joint his own sword is bound to (78); a ball's is
    its stub's; a hand the table names none for has none (Pikachu, Mario's left)."""
    assert melee_bones.item_joints('Yoshi') == {'right_hand': 'JOBJ_65'}
    assert melee_bones.item_joints('Ganondorf') == {'right_hand': 'JOBJ_78'}
    assert melee_bones.item_joints('Mario') == {'right_hand': 'JOBJ_42'}
    assert melee_bones.item_joints('Kirby') == {'left_stub': 'JOBJ_26', 'right_stub': 'JOBJ_31'}
    assert melee_bones.item_joints('Pikachu') == {}
