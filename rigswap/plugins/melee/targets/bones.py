"""The Melee fighters described from the game's own bone tables, not by the AI.

A target's description is made once and checked by hand (docs/pipeline.html, "What a game plugin is"). For a Melee
fighter the game already says what most joints are, so the description is written from that, and the AI's
description, where there is one, is kept beside it for comparison. Everything here is Melee data; the rest of rigswap
never reads it except through the Description it makes.

Where each fact comes from (game files: the vanilla v1.02 files, read 2026-10-10; PlCo.dat md5
be6e725873c4c5c0863e56c6c70f0d09, the same in three vanilla copies on this machine; decomp:
the doldecomp melee project):

  The joint names. PlCo.dat, root ftLoadCommonData, +0x10: one row per fighter kind, each row +0x00 a table joint ->
  slot, +0x04 a table slot -> joint (54 slots, 0xFF: none), +0x08 the number of the fighter's parts (decomp
  ft/fighter.c Fighter_LoadCommonData, "ftPartsTable = pData[4]"; ft/types.h struct FighterPartsTable;
  ft/ftparts.c ftParts_GetBoneIndex). The slot names, SLOTS, are in the order HSDLib stores the table
  (HSDLib's HSDRaw/Melee/SBM_BoneLookupTable.cs) with the joint names HSDLib's dummy fighter
  uses (HSDRawViewer/GUI/Plugins/Melee/DummyThrowModel.cs). The decomp's own enum (ft/forward.h Fighter_Part) is one
  slot short from slot 16 on and puts the thumbs before the item joint; Mario's joint tree says HSDLib's order is the
  right one (his joint 10 has the five finger chains under it, so it is LHandN, slot 22 in HSDLib's order and
  L1stNa in the decomp's). The old importer reads the same table the same way (randall
  model_import/native_rig/bone_lookup.py, "editor and decomp labels are not interchangeable").

  Parts with no joint. A fighter's parts are its costume's joints in tree order, except for the parts listed in
  PlCo.dat ftLoadCommonData +0x14 for that fighter kind, which the game leaves empty for joints it adds later (decomp
  ft/ftparts.c ftParts_SetupParts and ftParts_8007506C): Kirby's 7-10, 15, 16 and 28-34, Link's 68, Young Link's 72.
  The costume's joint n (JOBJ_n in the target's .glb) is the n-th part that is not one of these.

  The chains the game's physics moves. Pl<code>.dat, the ftData root, +0x2C: a count and a list of 0x18-byte entries,
  each +0x00 the chain's first part and +0x08 how many parts it runs (HSDLib HSDRaw/Melee/Pl/SBM_PhysicsGroup.cs;
  decomp ft/ftdynamics.c ftCo_8009CB40, which walks parts[first + i]). CHAINS (facts/chains.py) cites them.

  The tails: facts/chains.py says where they come from.

What is decided here, from the joint trees (positions, parents, and the vertices each joint skins in the vanilla
costume), not read from the game:

  WaistN is the spine, not the chest. Its head sits at the waist, about a unit above HipN on Mario, and everything
  above the hips hangs from it (the shoulders and the neck). The fighters with a WaistNb and a BustN instead (Fox,
  Zelda, Bowser and most others) have the same split the vocabulary has: WaistNb at the waist, the spine; BustN at the
  bottom of the rib cage, the chest. The weights step agrees: a part the target has no bone for takes the nearest
  part above it, so a piece on a Mario-like target's chest takes WaistN's weights; with WaistN the chest, a piece on
  its spine would take only the hips'.

  The adjust joints (LLegJA, LFootJA, LShoulderJA and the right side's) are none. Each sits exactly where the J joint
  under it is and skins no vertex in any of the 25 costumes; the J joint is the part's own bone.

  Body types. Kirby and Jigglypuff are balls: a round body with stubs for arms and feet below it, their leg joints
  above the feet inside the body and skinning nothing. Every other fighter is a biped, because the game animates it
  with the biped joint set (two legs from HipN, two arms with shoulders, a neck or a head): Yoshi, Bowser, Donkey Kong,
  Mewtwo, Pikachu, Pichu, Fox, Falco and Mr. Game & Watch included.

  A joint the table does not name is the fighter's own. When CHAINS or OWN_ROLES (facts/chains.py) list it, that says
  what it is. When they don't and the costume skins vertices to it, it is part of the part it hangs from (a toe, a
  jaw, a shoulder pad, a twist helper: "a bone for a smaller piece inside a part is that part", as round 3 tells the
  AI). When it skins nothing, it only marks a place (an effect point, an item point) and is none.

  Held things are none: swords, shields and scabbards on their own joints, and the item joints LHaveN and RHaveN,
  which carry what a fighter picks up or draws. Equipment is not body, even when drawn at spawn: Link shows models
  0, 1 and 2 (src/melee/ft/chara/ftLink/ftLk_Init.c:313-315); his supplied Link.bones.json has 227, 74, 175 and 38
  vertices on excluded JOBJ_25, 26, 67 and 69 (2026-10-10). Mr. Game & Watch's hands are drawn on his item joints,
  so his are his hands. So are the joints of the root (TopN, TransN,
  XRotN, YRotN, ThrowN, TransN2): in the meshes drawn at spawn they skin nothing (a modded project's costumes had
  Samus's morph ball and Yoshi's egg on them), and what hangs from the root without a part above it (Mr. Game &
  Watch's joint 1, 87 vertices) rides the root, not the body.

What describe_from_tables writes, beside the fighter's .glb in the targets folder:

  <Name>.bones.json         the bone graph, read from the .glb through Blender exactly as round 3 reads it
                            (targets/graph.py): +z up, facing -y, numbered from 1 depth first, so
                            bone n is JOBJ_(n-1)
  <Name>.description.json   the Description: body type, every bone's role, the chains as describe.bones groups them,
                            the joints at the heads of the parts' first bones, each toe at the far end of its foot's
                            mesh (the fit's rule, fit.foot_fronts), the fingers, and the eyes the game animates when
                            the game files are there. The AI's <Name>.description.json, when there was
                            one, is first kept as <Name>.description.ai.json.
  <Name>.weapons.json       the weapon slots (Target.weapons): each weapon the fighter fights with, its hand, grip and
                            tip direction, measured on the vanilla costume (weapons.weapon_slots; the costume with
                            every DObj is kept under vanilla/<Name>/every/)

The pieces list is empty on purpose: a target's original mesh is used whole, its weights copied by region, so its
pieces are never asked for (fit leaves a target's rigid pieces out of its head top; with none listed it leaves out
only what the chain bones move).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .roles import NONE, chains as join_chains, roles as roles_of
from ....library.body_types import part_ends
from ....vocabulary import BODY_TYPES, CANT_TELL, Description, Joint
from ... import Failed, description_path
from .. import FILES_VAR, TARGETS_VAR, _folder
from ..dat import Dat
from ..facts.chains import CHAINS, OWN_ROLES


# The 54 slots of a fighter's bone table, in the table's order (see the module's docstring).
SLOTS = (
    'TopN', 'TransN', 'XRotN', 'YRotN', 'HipN', 'WaistN',
    'LLegJA', 'LLegJ', 'LKneeJ', 'LFootJA', 'LFootJ',
    'RLegJA', 'RLegJ', 'RKneeJ', 'RFootJA', 'RFootJ',
    'WaistNb', 'BustN',
    'LShoulderN', 'LShoulderJA', 'LShoulderJ', 'LArmJ', 'LHandN',
    'L1stNa', 'L1stNb', 'L2ndNa', 'L2ndNb', 'L3rdNa', 'L3rdNb', 'L4thNa', 'L4thNb', 'LHaveN', 'LThumbNa', 'LThumbNb',
    'NeckN', 'HeadN',
    'RShoulderN', 'RShoulderJA', 'RShoulderJ', 'RArmJ', 'RHandN',
    'R1stNa', 'R1stNb', 'R2ndNa', 'R2ndNb', 'R3rdNa', 'R3rdNb', 'R4thNa', 'R4thNb', 'RHaveN', 'RThumbNa', 'RThumbNb',
    'ThrowN', 'TransN2',
)
NO_JOINT = 0xFF

# Each fighter's kind, the row of the game's tables that is its own (decomp ft/forward.h enum FighterKind).
KINDS = {
    'Mario': 0, 'Fox': 1, 'CaptainFalcon': 2, 'DonkeyKong': 3, 'Kirby': 4, 'Bowser': 5, 'Link': 6, 'Sheik': 7,
    'Ness': 8, 'Peach': 9, 'Popo': 10, 'Nana': 11, 'Pikachu': 12, 'Samus': 13, 'Yoshi': 14, 'Jigglypuff': 15,
    'Mewtwo': 16, 'Luigi': 17, 'Marth': 18, 'Zelda': 19, 'YoungLink': 20, 'DrMario': 21, 'Falco': 22, 'Pichu': 23,
    'GameAndWatch': 24, 'Ganondorf': 25, 'Roy': 26, 'MasterHand': 27, 'CrazyHand': 28, 'MaleWireframe': 29,
    'FemaleWireframe': 30, 'GigaBowser': 31, 'Sandbag': 32,
}

# What each named joint is, for each body type a fighter has: a part of the body type, a kind of chain, or none.
_SIDES = (('L', 'left'), ('R', 'right'))
_ROOT = {'TopN': NONE, 'TransN': NONE, 'XRotN': NONE, 'YRotN': NONE, 'ThrowN': NONE, 'TransN2': NONE}
_FINGERS = {f'{s}{finger}N{k}': 'finger' for s, _ in _SIDES for finger in ('1st', '2nd', '3rd', '4th', 'Thumb')
            for k in 'ab'}
ROLES = {
    'biped': {
        **_ROOT, **_FINGERS,
        'HipN': 'hips', 'WaistN': 'spine', 'WaistNb': 'spine', 'BustN': 'chest', 'NeckN': 'neck', 'HeadN': 'head',
        **{k: v for s, side in _SIDES for k, v in {
            f'{s}LegJA': NONE, f'{s}LegJ': f'{side}_thigh', f'{s}KneeJ': f'{side}_shin',
            f'{s}FootJA': NONE, f'{s}FootJ': f'{side}_foot',
            f'{s}ShoulderN': f'{side}_shoulder', f'{s}ShoulderJA': NONE, f'{s}ShoulderJ': f'{side}_upper_arm',
            f'{s}ArmJ': f'{side}_forearm', f'{s}HandN': f'{side}_hand', f'{s}HaveN': NONE}.items()},
    },
    # A ball's body is WaistN: HipN sits at the bottom of the ball and skins nothing. Its arms are stubs from the
    # shoulder joint out, its feet the foot joints; the leg joints above the feet are inside the body.
    'ball': {
        **_ROOT, **_FINGERS,
        'HipN': NONE, 'WaistN': 'body', 'WaistNb': 'body', 'BustN': 'body', 'NeckN': 'head', 'HeadN': 'head',
        **{k: v for s, side in _SIDES for k, v in {
            f'{s}LegJA': NONE, f'{s}LegJ': NONE, f'{s}KneeJ': NONE, f'{s}FootJA': NONE, f'{s}FootJ': f'{side}_foot',
            f'{s}ShoulderN': NONE, f'{s}ShoulderJA': NONE, f'{s}ShoulderJ': f'{side}_stub',
            f'{s}ArmJ': f'{side}_stub', f'{s}HandN': f'{side}_stub', f'{s}HaveN': NONE}.items()},
    },
}
BALLS = ('Kirby', 'Jigglypuff')


def body_type(name):
    return 'ball' if name in BALLS else 'biped'


# Each fighter's costume joints by number (JOBJ_n in its .glb) -> the slot that names it, read from PlCo.dat as the
# module's docstring says (bone_tables reads it again; tests/plugins/melee/test_bones.py checks the two agree when the
# game files are there). A joint not listed has no name in the game's table.
JOINT_NAMES = {
    # Mario: 61 joints
    'Mario': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 6: 'LShoulderN',
        7: 'LShoulderJA', 8: 'LShoulderJ', 9: 'LArmJ', 10: 'LHandN', 11: 'L1stNa', 12: 'L1stNb', 13: 'L2ndNa',
        14: 'L2ndNb', 15: 'L3rdNa', 16: 'L3rdNb', 17: 'L4thNa', 18: 'L4thNb', 19: 'LThumbNa', 20: 'LThumbNb',
        22: 'NeckN', 23: 'HeadN', 29: 'RShoulderN', 30: 'RShoulderJA', 31: 'RShoulderJ', 32: 'RArmJ', 33: 'RHandN',
        34: 'R1stNa', 35: 'R1stNb', 36: 'R2ndNa', 37: 'R2ndNb', 38: 'R3rdNa', 39: 'R3rdNb', 40: 'R4thNa',
        41: 'R4thNb', 42: 'RHaveN', 43: 'RThumbNa', 44: 'RThumbNb', 47: 'LLegJA', 48: 'LLegJ', 49: 'LKneeJ',
        50: 'LFootJA', 51: 'LFootJ', 53: 'RLegJA', 54: 'RLegJ', 55: 'RKneeJ', 56: 'RFootJA', 57: 'RFootJ',
        59: 'ThrowN', 60: 'TransN2'},
    # DrMario: 86 joints
    'DrMario': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 6: 'LShoulderN',
        7: 'LShoulderJA', 8: 'LShoulderJ', 9: 'LArmJ', 10: 'LHandN', 11: 'L1stNa', 12: 'L1stNb', 13: 'L2ndNa',
        14: 'L2ndNb', 15: 'L3rdNa', 16: 'L3rdNb', 17: 'L4thNa', 18: 'L4thNb', 19: 'LThumbNa', 20: 'LThumbNb',
        22: 'NeckN', 23: 'HeadN', 29: 'RShoulderN', 30: 'RShoulderJA', 31: 'RShoulderJ', 32: 'RArmJ', 33: 'RHandN',
        34: 'R1stNa', 35: 'R1stNb', 36: 'R2ndNa', 37: 'R2ndNb', 38: 'R3rdNa', 39: 'R3rdNb', 40: 'R4thNa',
        41: 'R4thNb', 42: 'RHaveN', 43: 'RThumbNa', 44: 'RThumbNb', 52: 'LLegJA', 53: 'LLegJ', 54: 'LKneeJ',
        55: 'LFootJA', 56: 'LFootJ', 68: 'RLegJA', 69: 'RLegJ', 70: 'RKneeJ', 71: 'RFootJA', 72: 'RFootJ',
        84: 'ThrowN', 85: 'TransN2'},
    # Luigi: 61 joints
    'Luigi': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 6: 'LShoulderN',
        7: 'LShoulderJA', 8: 'LShoulderJ', 9: 'LArmJ', 10: 'LHandN', 11: 'L1stNa', 12: 'L1stNb', 13: 'L2ndNa',
        14: 'L2ndNb', 15: 'L3rdNa', 16: 'L3rdNb', 17: 'L4thNa', 18: 'L4thNb', 19: 'LThumbNa', 20: 'LThumbNb',
        22: 'NeckN', 23: 'HeadN', 29: 'RShoulderN', 30: 'RShoulderJA', 31: 'RShoulderJ', 32: 'RArmJ', 33: 'RHandN',
        34: 'R1stNa', 35: 'R1stNb', 36: 'R2ndNa', 37: 'R2ndNb', 38: 'R3rdNa', 39: 'R3rdNb', 40: 'R4thNa',
        41: 'R4thNb', 42: 'RHaveN', 43: 'RThumbNa', 44: 'RThumbNb', 47: 'LLegJA', 48: 'LLegJ', 49: 'LKneeJ',
        50: 'LFootJA', 51: 'LFootJ', 53: 'RLegJA', 54: 'RLegJ', 55: 'RKneeJ', 56: 'RFootJA', 57: 'RFootJ',
        59: 'ThrowN', 60: 'TransN2'},
    # Peach: 114 joints
    'Peach': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        66: 'WaistNb', 67: 'BustN', 69: 'LShoulderN', 70: 'LShoulderJA', 71: 'LShoulderJ', 72: 'LArmJ', 74: 'LHandN',
        75: 'L1stNa', 76: 'L1stNb', 77: 'L2ndNa', 78: 'L2ndNb', 79: 'L3rdNa', 80: 'L3rdNb', 81: 'L4thNa',
        82: 'L4thNb', 83: 'LHaveN', 84: 'LThumbNa', 85: 'LThumbNb', 86: 'NeckN', 87: 'HeadN', 95: 'RShoulderN',
        96: 'RShoulderJA', 97: 'RShoulderJ', 98: 'RArmJ', 100: 'RHandN', 101: 'R1stNa', 102: 'R1stNb', 103: 'R2ndNa',
        104: 'R2ndNb', 105: 'R3rdNa', 106: 'R3rdNb', 107: 'R4thNa', 108: 'R4thNb', 109: 'RHaveN', 110: 'RThumbNa',
        111: 'RThumbNb', 112: 'ThrowN', 113: 'TransN2'},
    # Bowser: 76 joints
    'Bowser': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 14: 'RLegJA', 15: 'RLegJ', 16: 'RKneeJ', 17: 'RFootJA', 18: 'RFootJ',
        29: 'WaistNb', 30: 'BustN', 31: 'LShoulderN', 32: 'LShoulderJA', 33: 'LShoulderJ', 34: 'LArmJ', 35: 'LHandN',
        36: 'L1stNa', 37: 'L1stNb', 38: 'L2ndNa', 39: 'L2ndNb', 40: 'L3rdNa', 41: 'L3rdNb', 42: 'LThumbNa',
        43: 'LThumbNb', 47: 'NeckN', 48: 'HeadN', 58: 'RShoulderN', 59: 'RShoulderJA', 60: 'RShoulderJ', 61: 'RArmJ',
        62: 'RHandN', 63: 'R1stNa', 64: 'R1stNb', 65: 'R2ndNa', 66: 'R2ndNb', 67: 'R3rdNa', 68: 'R3rdNb',
        69: 'RHaveN', 70: 'RThumbNa', 71: 'RThumbNb', 74: 'ThrowN', 75: 'TransN2'},
    # Yoshi: 70 joints
    'Yoshi': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 12: 'RLegJA', 13: 'RLegJ', 14: 'RKneeJ', 15: 'RFootJA', 16: 'RFootJ',
        22: 'WaistNb', 23: 'BustN', 24: 'LShoulderN', 25: 'LShoulderJA', 26: 'LShoulderJ', 27: 'LArmJ', 28: 'LHandN',
        30: 'L1stNb', 32: 'L2ndNb', 34: 'L3rdNb', 36: 'LThumbNb', 38: 'HeadN', 54: 'RShoulderN', 55: 'RShoulderJA',
        56: 'RShoulderJ', 57: 'RArmJ', 58: 'RHandN', 60: 'R1stNb', 62: 'R2ndNb', 64: 'R3rdNb', 65: 'RHaveN',
        67: 'RThumbNb', 68: 'ThrowN', 69: 'TransN2'},
    # DonkeyKong: 75 joints
    'DonkeyKong': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 15: 'RLegJA', 16: 'RLegJ', 17: 'RKneeJ', 18: 'RFootJA', 19: 'RFootJ',
        25: 'WaistNb', 26: 'BustN', 27: 'LShoulderN', 28: 'LShoulderJA', 29: 'LShoulderJ', 30: 'LArmJ', 31: 'LHandN',
        32: 'L1stNa', 33: 'L1stNb', 34: 'L2ndNa', 35: 'L2ndNb', 36: 'L3rdNa', 37: 'L3rdNb', 38: 'L4thNa',
        39: 'L4thNb', 40: 'LThumbNa', 41: 'LThumbNb', 43: 'NeckN', 44: 'HeadN', 49: 'RShoulderN', 50: 'RShoulderJA',
        51: 'RShoulderJ', 52: 'RArmJ', 53: 'RHandN', 54: 'R1stNa', 55: 'R1stNb', 56: 'R2ndNa', 57: 'R2ndNb',
        58: 'R3rdNa', 59: 'R3rdNb', 60: 'R4thNa', 61: 'R4thNb', 62: 'RHaveN', 63: 'RThumbNa', 64: 'RThumbNb',
        73: 'ThrowN', 74: 'TransN2'},
    # CaptainFalcon: 63 joints
    'CaptainFalcon': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 6: 'LLegJA', 7: 'LLegJ', 8: 'LKneeJ',
        9: 'LFootJA', 10: 'LFootJ', 12: 'RLegJA', 13: 'RLegJ', 14: 'RKneeJ', 15: 'RFootJA', 16: 'RFootJ',
        18: 'WaistNb', 19: 'BustN', 21: 'LShoulderN', 22: 'LShoulderJA', 23: 'LShoulderJ', 24: 'LArmJ', 25: 'LHandN',
        26: 'L1stNa', 27: 'L1stNb', 28: 'L2ndNa', 29: 'L2ndNb', 30: 'L3rdNa', 31: 'L3rdNb', 32: 'L4thNa',
        33: 'L4thNb', 35: 'LThumbNa', 36: 'LThumbNb', 38: 'NeckN', 39: 'HeadN', 42: 'RShoulderN', 44: 'RShoulderJA',
        45: 'RShoulderJ', 46: 'RArmJ', 47: 'RHandN', 48: 'R1stNa', 49: 'R1stNb', 50: 'R2ndNa', 51: 'R2ndNb',
        52: 'R3rdNa', 53: 'R3rdNb', 54: 'R4thNa', 55: 'R4thNb', 57: 'RHaveN', 58: 'RThumbNa', 59: 'RThumbNb',
        61: 'ThrowN', 62: 'TransN2'},
    # Ganondorf: 84 joints
    'Ganondorf': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        17: 'WaistNb', 18: 'BustN', 20: 'LShoulderN', 21: 'LShoulderJA', 22: 'LShoulderJ', 23: 'LArmJ', 25: 'LHandN',
        26: 'L1stNa', 27: 'L1stNb', 28: 'L2ndNa', 29: 'L2ndNb', 30: 'L3rdNa', 31: 'L3rdNb', 32: 'L4thNa',
        33: 'L4thNb', 34: 'LThumbNa', 35: 'LThumbNb', 64: 'RShoulderN', 65: 'RShoulderJA', 66: 'RShoulderJ',
        67: 'RArmJ', 69: 'RHandN', 70: 'R1stNa', 71: 'R1stNb', 72: 'R2ndNa', 73: 'R2ndNb', 74: 'R3rdNa', 75: 'R3rdNb',
        76: 'R4thNa', 77: 'R4thNb', 78: 'RHaveN', 79: 'RThumbNa', 80: 'RThumbNb', 82: 'ThrowN', 83: 'TransN2'},
    # Falco: 67 joints
    'Falco': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        18: 'WaistNb', 19: 'BustN', 21: 'LShoulderN', 22: 'LShoulderJA', 23: 'LShoulderJ', 24: 'LArmJ', 26: 'LHandN',
        27: 'L1stNa', 28: 'L1stNb', 29: 'L2ndNa', 30: 'L2ndNb', 31: 'L3rdNa', 32: 'L3rdNb', 33: 'L4thNa',
        34: 'L4thNb', 35: 'LThumbNa', 36: 'LThumbNb', 38: 'NeckN', 39: 'HeadN', 47: 'RShoulderN', 48: 'RShoulderJA',
        49: 'RShoulderJ', 50: 'RArmJ', 52: 'RHandN', 53: 'R1stNa', 54: 'R1stNb', 55: 'R2ndNa', 56: 'R2ndNb',
        57: 'R3rdNa', 58: 'R3rdNb', 59: 'R4thNa', 60: 'R4thNb', 61: 'RHaveN', 62: 'RThumbNa', 63: 'RThumbNb',
        65: 'ThrowN', 66: 'TransN2'},
    # Fox: 73 joints
    'Fox': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        21: 'WaistNb', 22: 'BustN', 23: 'LShoulderN', 24: 'LShoulderJA', 25: 'LShoulderJ', 26: 'LArmJ', 27: 'LHandN',
        28: 'L1stNa', 29: 'L1stNb', 30: 'L2ndNa', 31: 'L2ndNb', 32: 'L3rdNa', 33: 'L3rdNb', 34: 'L4thNa',
        35: 'L4thNb', 37: 'LThumbNa', 38: 'LThumbNb', 40: 'NeckN', 41: 'HeadN', 53: 'RShoulderN', 54: 'RShoulderJA',
        55: 'RShoulderJ', 56: 'RArmJ', 57: 'RHandN', 58: 'R1stNa', 59: 'R1stNb', 60: 'R2ndNa', 61: 'R2ndNb',
        62: 'R3rdNa', 63: 'R3rdNb', 64: 'R4thNa', 65: 'R4thNb', 67: 'RHaveN', 68: 'RThumbNa', 69: 'RThumbNb',
        71: 'ThrowN', 72: 'TransN2'},
    # Ness: 63 joints
    'Ness': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 7: 'LShoulderN',
        8: 'LShoulderJA', 9: 'LShoulderJ', 10: 'LArmJ', 11: 'LHandN', 12: 'L1stNa', 13: 'L1stNb', 14: 'L2ndNa',
        15: 'L2ndNb', 16: 'L3rdNa', 17: 'L3rdNb', 18: 'L4thNa', 19: 'L4thNb', 20: 'LThumbNa', 21: 'LThumbNb',
        23: 'NeckN', 24: 'HeadN', 29: 'RShoulderN', 30: 'RShoulderJA', 31: 'RShoulderJ', 32: 'RArmJ', 33: 'RHandN',
        34: 'R1stNa', 35: 'R1stNb', 36: 'R2ndNa', 37: 'R2ndNb', 38: 'R3rdNa', 39: 'R3rdNb', 40: 'R4thNa',
        41: 'R4thNb', 42: 'RHaveN', 43: 'RThumbNa', 44: 'RThumbNb', 47: 'LLegJA', 48: 'LLegJ', 49: 'LKneeJ',
        50: 'LFootJA', 51: 'LFootJ', 54: 'RLegJA', 55: 'RLegJ', 56: 'RKneeJ', 57: 'RFootJA', 58: 'RFootJ',
        61: 'ThrowN', 62: 'TransN2'},
    # Kirby: 46 joints (59 parts; parts 7, 8, 9, 10, 15, 16, 28, 29, 30, 31, 32, 33, 34 have no costume joint)
    'Kirby': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 22: 'LShoulderN',
        23: 'LShoulderJA', 24: 'LShoulderJ', 25: 'LArmJ', 26: 'LHaveN', 27: 'RShoulderN', 28: 'RShoulderJA',
        29: 'RShoulderJ', 30: 'RArmJ', 31: 'RHaveN', 32: 'LLegJA', 33: 'LLegJ', 34: 'LKneeJ', 35: 'LFootJA',
        36: 'LFootJ', 38: 'RLegJA', 39: 'RLegJ', 40: 'RKneeJ', 41: 'RFootJA', 42: 'RFootJ', 44: 'ThrowN',
        45: 'TransN2'},
    # Samus: 60 joints
    'Samus': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 12: 'RLegJA', 13: 'RLegJ', 14: 'RKneeJ', 15: 'RFootJA', 16: 'RFootJ',
        19: 'WaistNb', 20: 'BustN', 21: 'LShoulderN', 25: 'LShoulderJA', 26: 'LShoulderJ', 27: 'LArmJ', 28: 'LHandN',
        29: 'L1stNa', 30: 'L1stNb', 31: 'L2ndNa', 32: 'L2ndNb', 33: 'L3rdNa', 34: 'L3rdNb', 35: 'L4thNa',
        36: 'L4thNb', 37: 'LHaveN', 38: 'LThumbNa', 39: 'LThumbNb', 41: 'NeckN', 42: 'HeadN', 43: 'RShoulderN',
        47: 'RShoulderJA', 48: 'RShoulderJ', 49: 'RArmJ', 58: 'ThrowN', 59: 'TransN2'},
    # Zelda: 118 joints
    'Zelda': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        66: 'WaistNb', 67: 'BustN', 68: 'LShoulderN', 71: 'LShoulderJA', 72: 'LShoulderJ', 73: 'LArmJ', 76: 'LHandN',
        77: 'L1stNa', 78: 'L1stNb', 79: 'L2ndNa', 80: 'L2ndNb', 81: 'L3rdNa', 82: 'L3rdNb', 83: 'L4thNa',
        84: 'L4thNb', 85: 'LHaveN', 86: 'LThumbNa', 87: 'LThumbNb', 88: 'NeckN', 89: 'HeadN', 96: 'RShoulderN',
        99: 'RShoulderJA', 100: 'RShoulderJ', 101: 'RArmJ', 104: 'RHandN', 105: 'R1stNa', 106: 'R1stNb',
        107: 'R2ndNa', 108: 'R2ndNb', 109: 'R3rdNa', 110: 'R3rdNb', 111: 'R4thNa', 112: 'R4thNb', 113: 'RHaveN',
        114: 'RThumbNa', 115: 'RThumbNb', 116: 'ThrowN', 117: 'TransN2'},
    # Sheik: 58 joints
    'Sheik': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        17: 'WaistNb', 18: 'BustN', 19: 'LShoulderN', 20: 'LShoulderJA', 21: 'LShoulderJ', 22: 'LArmJ', 23: 'LHandN',
        24: 'L1stNa', 25: 'L1stNb', 26: 'L2ndNa', 27: 'L2ndNb', 28: 'L3rdNa', 29: 'L3rdNb', 30: 'L4thNa',
        31: 'L4thNb', 32: 'LThumbNa', 33: 'LThumbNb', 35: 'NeckN', 36: 'HeadN', 39: 'RShoulderN', 40: 'RShoulderJA',
        41: 'RShoulderJ', 42: 'RArmJ', 43: 'RHandN', 44: 'R1stNa', 45: 'R1stNb', 46: 'R2ndNa', 47: 'R2ndNb',
        48: 'R3rdNa', 49: 'R3rdNb', 50: 'R4thNa', 51: 'R4thNb', 52: 'RHaveN', 53: 'RThumbNa', 54: 'RThumbNb',
        56: 'ThrowN', 57: 'TransN2'},
    # Link: 75 joints (76 parts; part 68 has no costume joint)
    'Link': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        17: 'WaistNb', 18: 'BustN', 19: 'LShoulderN', 20: 'LShoulderJA', 21: 'LShoulderJ', 22: 'LArmJ', 23: 'LHandN',
        27: 'L1stNa', 28: 'L1stNb', 29: 'L2ndNa', 30: 'L2ndNb', 31: 'L3rdNa', 32: 'L3rdNb', 33: 'L4thNa',
        34: 'L4thNb', 35: 'LHaveN', 36: 'LThumbNa', 37: 'LThumbNb', 39: 'NeckN', 40: 'HeadN', 51: 'RShoulderN',
        52: 'RShoulderJA', 53: 'RShoulderJ', 54: 'RArmJ', 55: 'RHandN', 56: 'R1stNa', 57: 'R1stNb', 58: 'R2ndNa',
        59: 'R2ndNb', 60: 'R3rdNa', 61: 'R3rdNb', 62: 'R4thNa', 63: 'R4thNb', 64: 'RHaveN', 65: 'RThumbNa',
        66: 'RThumbNb', 73: 'ThrowN', 74: 'TransN2'},
    # YoungLink: 79 joints (80 parts; part 72 has no costume joint)
    'YoungLink': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 11: 'RLegJA', 12: 'RLegJ', 13: 'RKneeJ', 14: 'RFootJA', 15: 'RFootJ',
        17: 'WaistNb', 18: 'BustN', 19: 'LShoulderN', 20: 'LShoulderJA', 21: 'LShoulderJ', 22: 'LArmJ', 25: 'LHandN',
        29: 'L1stNa', 30: 'L1stNb', 31: 'L2ndNa', 32: 'L2ndNb', 33: 'L3rdNa', 34: 'L3rdNb', 35: 'L4thNa',
        36: 'L4thNb', 37: 'LHaveN', 38: 'LThumbNa', 39: 'LThumbNb', 41: 'NeckN', 42: 'HeadN', 53: 'RShoulderN',
        54: 'RShoulderJA', 55: 'RShoulderJ', 56: 'RArmJ', 59: 'RHandN', 60: 'R1stNa', 61: 'R1stNb', 62: 'R2ndNa',
        63: 'R2ndNb', 64: 'R3rdNa', 65: 'R3rdNb', 66: 'R4thNa', 67: 'R4thNb', 68: 'RHaveN', 69: 'RThumbNa',
        70: 'RThumbNb', 77: 'ThrowN', 78: 'TransN2'},
    # Pichu: 46 joints
    'Pichu': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 7: 'LShoulderN',
        8: 'LShoulderJA', 9: 'LShoulderJ', 10: 'LArmJ', 11: 'LHandN', 15: 'NeckN', 23: 'RShoulderN',
        24: 'RShoulderJA', 25: 'RShoulderJ', 26: 'RArmJ', 27: 'RHandN', 31: 'LLegJA', 32: 'LLegJ', 33: 'LKneeJ',
        34: 'LFootJA', 35: 'LFootJ', 37: 'RLegJA', 38: 'RLegJ', 39: 'RKneeJ', 40: 'RFootJA', 41: 'RFootJ',
        44: 'ThrowN', 45: 'TransN2'},
    # Pikachu: 53 joints
    'Pikachu': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 6: 'LShoulderN',
        7: 'LShoulderJA', 8: 'LShoulderJ', 9: 'LArmJ', 10: 'LHandN', 14: 'NeckN', 22: 'RShoulderN', 23: 'RShoulderJA',
        24: 'RShoulderJ', 25: 'RArmJ', 26: 'RHandN', 31: 'LLegJA', 32: 'LLegJ', 33: 'LKneeJ', 34: 'LFootJA',
        35: 'LFootJ', 39: 'RLegJA', 40: 'RLegJ', 41: 'RKneeJ', 42: 'RFootJA', 43: 'RFootJ', 51: 'ThrowN',
        52: 'TransN2'},
    # Jigglypuff: 50 joints
    'Jigglypuff': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 27: 'LShoulderN',
        28: 'LShoulderJA', 29: 'LShoulderJ', 30: 'LArmJ', 31: 'RShoulderN', 32: 'RShoulderJA', 33: 'RShoulderJ',
        34: 'RArmJ', 36: 'LLegJA', 37: 'LLegJ', 38: 'LKneeJ', 39: 'LFootJA', 40: 'LFootJ', 42: 'RLegJA', 43: 'RLegJ',
        44: 'RKneeJ', 45: 'RFootJA', 46: 'RFootJ', 48: 'ThrowN', 49: 'TransN2'},
    # Mewtwo: 68 joints
    'Mewtwo': {0: 'TopN', 2: 'TransN', 3: 'XRotN', 4: 'YRotN', 5: 'HipN', 7: 'LLegJA', 8: 'LLegJ', 9: 'LKneeJ',
        10: 'LFootJA', 11: 'LFootJ', 13: 'RLegJA', 14: 'RLegJ', 15: 'RKneeJ', 16: 'RFootJA', 17: 'RFootJ',
        27: 'WaistNb', 28: 'BustN', 29: 'LShoulderN', 30: 'LShoulderJA', 31: 'LShoulderJ', 32: 'LArmJ', 35: 'LHandN',
        36: 'L1stNa', 37: 'L1stNb', 38: 'L2ndNa', 39: 'L2ndNb', 40: 'L3rdNa', 41: 'L3rdNb', 42: 'NeckN', 43: 'HeadN',
        52: 'RShoulderN', 53: 'RShoulderJA', 54: 'RShoulderJ', 55: 'RArmJ', 58: 'RHandN', 59: 'R1stNa', 60: 'R1stNb',
        61: 'R2ndNa', 62: 'R2ndNb', 63: 'R3rdNa', 64: 'R3rdNb', 65: 'RHaveN', 66: 'ThrowN', 67: 'TransN2'},
    # GameAndWatch: 53 joints
    'GameAndWatch': {0: 'TopN', 2: 'TransN', 3: 'XRotN', 4: 'YRotN', 5: 'HipN', 6: 'WaistN', 9: 'LShoulderN',
        10: 'LShoulderJA', 11: 'LShoulderJ', 12: 'LArmJ', 14: 'LHandN', 15: 'L2ndNa', 16: 'L2ndNb', 17: 'LHaveN',
        18: 'LThumbNa', 19: 'NeckN', 20: 'HeadN', 24: 'RShoulderN', 25: 'RShoulderJA', 26: 'RShoulderJ', 27: 'RArmJ',
        29: 'RHandN', 30: 'R2ndNa', 31: 'R2ndNb', 32: 'RHaveN', 34: 'RThumbNa', 35: 'LLegJA', 36: 'LLegJ',
        37: 'LKneeJ', 38: 'LFootJA', 39: 'LFootJ', 43: 'RLegJA', 44: 'RLegJ', 45: 'RKneeJ', 46: 'RFootJA',
        47: 'RFootJ', 51: 'ThrowN', 52: 'TransN2'},
    # Marth: 90 joints
    'Marth': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 12: 'RLegJA', 13: 'RLegJ', 14: 'RKneeJ', 15: 'RFootJA', 16: 'RFootJ',
        21: 'WaistNb', 22: 'BustN', 25: 'LShoulderN', 27: 'LShoulderJA', 28: 'LShoulderJ', 29: 'LArmJ', 31: 'LHandN',
        32: 'L1stNa', 33: 'L1stNb', 34: 'L2ndNa', 35: 'L2ndNb', 36: 'L3rdNa', 37: 'L3rdNb', 38: 'L4thNa',
        39: 'L4thNb', 40: 'LHaveN', 41: 'LThumbNa', 42: 'LThumbNb', 59: 'NeckN', 60: 'HeadN', 67: 'RShoulderN',
        69: 'RShoulderJA', 70: 'RShoulderJ', 71: 'RArmJ', 73: 'RHandN', 77: 'R1stNa', 78: 'R1stNb', 79: 'R2ndNa',
        80: 'R2ndNb', 81: 'R3rdNa', 82: 'R3rdNb', 83: 'R4thNa', 84: 'R4thNb', 85: 'RThumbNa', 86: 'RThumbNb',
        88: 'ThrowN', 89: 'TransN2'},
    # Roy: 92 joints
    'Roy': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'LLegJA', 6: 'LLegJ', 7: 'LKneeJ',
        8: 'LFootJA', 9: 'LFootJ', 13: 'RLegJA', 14: 'RLegJ', 15: 'RKneeJ', 16: 'RFootJA', 17: 'RFootJ',
        23: 'WaistNb', 24: 'BustN', 27: 'LShoulderN', 29: 'LShoulderJA', 30: 'LShoulderJ', 31: 'LArmJ', 33: 'LHandN',
        34: 'L1stNa', 35: 'L1stNb', 36: 'L2ndNa', 37: 'L2ndNb', 38: 'L3rdNa', 39: 'L3rdNb', 40: 'L4thNa',
        41: 'L4thNb', 42: 'LHaveN', 43: 'LThumbNa', 44: 'LThumbNb', 61: 'NeckN', 62: 'HeadN', 69: 'RShoulderN',
        71: 'RShoulderJA', 72: 'RShoulderJ', 73: 'RArmJ', 75: 'RHandN', 79: 'R1stNa', 80: 'R1stNb', 81: 'R2ndNa',
        82: 'R2ndNb', 83: 'R3rdNa', 84: 'R3rdNb', 85: 'R4thNa', 86: 'R4thNb', 87: 'RThumbNa', 88: 'RThumbNb',
        90: 'ThrowN', 91: 'TransN2'},
    # Popo: 49 joints
    'Popo': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 7: 'LShoulderN',
        8: 'LShoulderJA', 9: 'LShoulderJ', 10: 'LArmJ', 11: 'LHandN', 12: 'L3rdNa', 13: 'L3rdNb', 14: 'LHaveN',
        16: 'LThumbNa', 17: 'LThumbNb', 18: 'NeckN', 19: 'HeadN', 22: 'RShoulderN', 23: 'RShoulderJA',
        24: 'RShoulderJ', 25: 'RArmJ', 26: 'RHandN', 27: 'R3rdNa', 28: 'R3rdNb', 29: 'RHaveN', 30: 'RThumbNa',
        31: 'RThumbNb', 35: 'LLegJA', 36: 'LLegJ', 37: 'LKneeJ', 38: 'LFootJA', 39: 'LFootJ', 41: 'RLegJA',
        42: 'RLegJ', 43: 'RKneeJ', 44: 'RFootJA', 45: 'RFootJ', 47: 'ThrowN', 48: 'TransN2'},
    # Nana: 49 joints
    'Nana': {0: 'TopN', 1: 'TransN', 2: 'XRotN', 3: 'YRotN', 4: 'HipN', 5: 'WaistN', 7: 'LShoulderN',
        8: 'LShoulderJA', 9: 'LShoulderJ', 10: 'LArmJ', 11: 'LHandN', 12: 'L3rdNa', 13: 'L3rdNb', 14: 'LHaveN',
        16: 'LThumbNa', 17: 'LThumbNb', 18: 'NeckN', 19: 'HeadN', 22: 'RShoulderN', 23: 'RShoulderJA',
        24: 'RShoulderJ', 25: 'RArmJ', 26: 'RHandN', 27: 'R3rdNa', 28: 'R3rdNb', 29: 'RHaveN', 30: 'RThumbNa',
        31: 'RThumbNb', 35: 'LLegJA', 36: 'LLegJ', 37: 'LKneeJ', 38: 'LFootJA', 39: 'LFootJ', 41: 'RLegJA',
        42: 'RLegJ', 43: 'RKneeJ', 44: 'RFootJA', 45: 'RFootJ', 47: 'ThrowN', 48: 'TransN2'},
}

_JOBJ = re.compile(r'JOBJ_(\d+)$')


def item_joints(name):
    """{hand part: 'JOBJ_<n>'}: fighter ``name``'s item joints (LHaveN, RHaveN), each under its hand's bone, in the
    fist: the game carries what that hand holds on it (an item picked up; Ganondorf's own sword is bound to his), by
    the fighter's own bone table (JOINT_NAMES). A hand the table names no item joint for is left out
    (Target.item_joints)."""
    named = {joint: n for n, joint in JOINT_NAMES.get(name, {}).items()}
    hands = ROLES[body_type(name)]
    return {hands[f'{s}HandN']: f'JOBJ_{named[f"{s}HaveN"]}' for s, _ in _SIDES if f'{s}HaveN' in named}


def joint_number(bone_name):
    """The costume joint a bone of the target's .glb is: JOBJ_<n> -> n."""
    m = _JOBJ.match(bone_name)
    if not m:
        raise Failed(f'bone {bone_name!r} is not a costume joint (JOBJ_<n>)')
    return int(m.group(1))


def joint_roles(name, bones):
    """{bone number: role} for the bones of fighter ``name``'s graph (bones.json's list: n, name, parent, vertices).
    In order of precedence: OWN_ROLES, CHAINS, the joint's name (ROLES), then for a joint the table does not name, the
    role of the nearest bone above it that is a part when the costume skins vertices to it, else none."""
    body = body_type(name)
    names, table = JOINT_NAMES[name], ROLES[body]
    in_chain = {j: kind for kind, joints in CHAINS.get(name, ()) for j in joints}
    own = OWN_ROLES.get(name, {})
    parts = set(BODY_TYPES[body])
    by_n = {b['n']: b for b in bones}
    out = {}

    def role(n):
        if n in out:
            return out[n]
        b = by_n[n]
        j = joint_number(b['name'])
        if j in own:
            r = own[j]
        elif j in in_chain:
            r = in_chain[j]
        elif j in names:
            r = table[names[j]]
        elif b.get('vertices'):
            r, up = NONE, b['parent']
            while up is not None:
                if role(up) in parts:
                    r = role(up)
                    break
                up = by_n[up]['parent']
        else:
            r = NONE
        out[n] = r
        return r

    for n in sorted(by_n):
        role(n)
    return out


# ------------------------------------------------------------------------------------------------- the game's files

def bone_tables(plco):
    """For every fighter kind, what PlCo.dat says (see the module's docstring): {fighter: {'names': {costume joint:
    slot name}, 'joints': the costume's joint count, 'parts': the fighter's part count, 'skipped': the parts with no
    costume joint}}. JOINT_NAMES was written from this."""
    d = Dat.read(plco)
    root = d.root('ftLoadCommonData')
    if root is None:
        raise Failed(f'{Path(plco).name} has no ftLoadCommonData root')
    rows, empties = d.ptr(root + 0x10), d.ptr(root + 0x14)
    out = {}
    for name, kind in KINDS.items():
        row = d.ptr(rows + 4 * kind)
        count = d.u32(row + 8)
        slot_to_part = d.bytes_at(d.ptr(row + 4), len(SLOTS))
        listed = d.ptr(empties + 4 * kind) if empties is not None else None
        skipped = set()
        if listed is not None and d.u32(listed + 4):
            first = d.ptr(listed)
            skipped = {d.bytes_at(first + 4 * i, 1)[0] for i in range(d.u32(listed + 4))}
        costume = {}
        for part in range(count):
            if part not in skipped:
                costume[part] = len(costume)
        names = {}
        for slot, part in enumerate(slot_to_part):
            if part == NO_JOINT:
                continue
            if part not in costume:
                raise Failed(f'{name}: slot {SLOTS[slot]} names part {part}, which has no costume joint')
            joint = costume[part]
            if joint in names:
                raise Failed(f'{name}: joint {joint} is both {names[joint]} and {SLOTS[slot]}')
            names[joint] = SLOTS[slot]
        out[name] = {'names': dict(sorted(names.items())), 'joints': len(costume), 'parts': count,
                     'skipped': sorted(skipped)}
    return out


def dynamics(pl_dat, skipped=()):
    """The chains the game's physics moves for a fighter, from its Pl<code>.dat (see the module's docstring), as lists
    of costume joints. ``skipped``: the fighter's parts with no costume joint (bone_tables)."""
    d = Dat.read(pl_dat)
    root = next((off for name, off in d.roots if name.startswith('ftData')), None)
    group = d.ptr(root + 0x2C) if root is not None else None
    if group is None or not d.u32(group) or d.ptr(group + 4) is None:
        return []
    first = d.ptr(group + 4)
    out = []
    for i in range(d.u32(group)):
        start, count = d.u32(first + 0x18 * i), d.u32(first + 0x18 * i + 8)
        out.append([p - sum(s < p for s in skipped) for p in range(start, start + count)])
    return out


def _eyes(f, files_dir):
    """The textures the game animates on the fighter's costume (the eyes' blink), when the game files are there: how
    many, and where each is (DObj place, TObj place in that DObj). None without the files."""
    from ..export import dobjs as melee_dobjs
    folder = files_dir or os.environ.get(FILES_VAR)
    if not folder:
        return None
    fighter_dat, costume = Path(folder) / f'Pl{f.code}.dat', Path(folder) / f'Pl{f.code}Nr.dat'
    if not (fighter_dat.is_file() and costume.is_file()):
        return None
    animated = melee_dobjs.fighter_tables(fighter_dat).animated
    if not animated:
        return None
    at = melee_dobjs.texture_places(melee_dobjs.layout(costume, f.symbol), animated)
    return {'animated_textures': len(animated), 'at': [list(p) if p else None for p in at],
            'from': f'{fighter_dat.name}: the TObjs the game gives a texture animation (the blink), in {costume.name}'}


# ----------------------------------------------------------------------------------------------- the description

def bone_graph(glb, out):
    """The bone graph of the target's .glb, written to ``out`` as round 3 writes bones.json: imported the way load
    imports a .glb, read with describe.bones.graph (+z up, facing -y, as Blender's glTF importer leaves a y-up file).
    Returns the bones."""
    import bpy
    from . import graph as picture_bones
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb), disable_bone_shape=True)
    for ob in bpy.context.scene.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    bones, _ = picture_bones.graph()
    bpy.ops.wm.read_homefile(use_empty=True)
    picture_bones.write_graph(out, bones, glb=Path(glb).name, up_in_file='+z', front_in_file='-y',
                              frame='upright: +z up, facing -y, world units, rest pose',
                              roles='from the game\'s bone tables (rigswap.plugins.melee.bones), not asked')
    return bones


def _joints(bones, roles_, body, mesh=None):
    """{joint: Joint}: each joint the body type has (its part_ends, library.body_types) that is the start of a part
    with a bone, at the head of that part's first bone (correspond.part_starts), the ankles among them; and each foot's
    toe at the far end of the foot's mesh, as the fit finds every model's (fit.foot_fronts), from ``mesh``: (points,
    weights on ``bones``) of the costume as the fit reads it (fit.target_mesh); no toes without it. A joint two parts
    start at (the neck: the neck and the head) is the first part's in the body type's order. The head's top and the
    hands' ends are at no bone's head: left out."""
    from . import skeleton as correspond, geometry as fit
    skeleton = correspond.skeleton({'bones': bones}, roles_, body)
    starts = correspond.part_starts(skeleton, body)
    ends = part_ends(body)
    out = {}
    for part in BODY_TYPES[body]:
        if part in starts:
            out.setdefault(ends[part][0], Joint(pos=[round(float(v), 6) for v in starts[part]['start']]))
    for tip in (fit.foot_fronts(skeleton, body, *mesh).values() if mesh is not None else ()):
        out.setdefault(tip['name'], Joint(pos=[round(float(v), 6) for v in tip['tip']]))
    return out


def description(name, bones, eyes=None, graph=None, mesh=None):
    """The Description of fighter ``name`` from its bone graph ``bones`` (bones.json's list), ``graph`` its path
    relative to the description, ``mesh`` its costume's points and weights on ``bones`` (fit.target_mesh: the toes,
    _joints). Nothing in it is a default and nothing is unsure."""
    body = body_type(name)
    roles_ = joint_roles(name, bones)
    allowed = set(roles_of(body))
    wrong = {n: r for n, r in roles_.items() if r not in allowed}
    if wrong:
        raise Failed(f'{name}: roles a {body} has no place for: {wrong}')
    found = join_chains(bones, roles_, body)
    details = {}
    fingers = {}
    for c in found:
        if c.kind == 'finger':
            fingers.setdefault(c.root, []).append(c.bones)
    if fingers:
        details['fingers'] = fingers
    if eyes:
        details['eyes'] = eyes
    return Description(
        body_type=body, up='+z', front='-y',
        has=['tail'] if any(c.kind == 'tail' for c in found) else [],
        holds=[],
        pieces=[],                  # a target's original mesh is used whole: its pieces are never asked for
        joints=_joints(bones, roles_, body, mesh),
        bones={'graph': graph or f'{name}.bones.json', 'roles': {str(n): roles_[n] for n in sorted(roles_)}},
        chains=found, details=details, model=f'{name}.glb')


def ai_path(glb):
    """Where the AI's description of a target is kept once the tables' is written: <Name>.description.ai.json."""
    return Path(glb).with_suffix('.description.ai.json')


def made_from_tables(path):
    """Whether the description at ``path`` was written here (its bone graph is <Name>.bones.json beside it)."""
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return (d.get('bones') or {}).get('graph') == Path(path).name.split('.')[0] + '.bones.json'


def describe_from_tables(name, targets_dir=None, files_dir=None):
    """Write fighter ``name``'s description from the game's tables: <Name>.bones.json and <Name>.description.json
    beside its .glb in the targets folder (``targets_dir``, else RIGSWAP_MELEE_TARGETS). An AI-made
    <Name>.description.json is first kept as <Name>.description.ai.json. ``files_dir`` (else RIGSWAP_MELEE_FILES), when
    there, adds the eyes and measures the weapon slots, <Name>.weapons.json (weapons.write_weapons; a fighter with
    no weapon gets an empty list without the files). Returns the Description."""
    from . import geometry as fit
    from ..facts.fighters import fighter
    from .weapons import write_weapons
    f = fighter(name)
    folder = _folder(targets_dir, TARGETS_VAR, 'Melee targets')
    glb = folder / f'{f.name}.glb'
    if not glb.is_file():
        raise Failed(f'{f.name} has not been prepared: no {glb.name} in {folder}; run melee.prepare')
    graph = folder / f'{f.name}.bones.json'
    bones = bone_graph(glb, graph)
    names = JOINT_NAMES[f.name]
    if len(bones) <= max(names):
        raise Failed(f'{glb.name} has {len(bones)} joints; the game\'s table names joint {max(names)}')
    desc = description(f.name, bones, _eyes(f, files_dir), graph.name, fit.target_mesh(glb, bones))
    written = description_path(glb)
    if written.is_file() and not made_from_tables(written):
        written.replace(ai_path(glb))
    desc.save(written)
    write_weapons(f, bones, desc.bones['roles'], glb, files_dir)
    return desc


# ------------------------------------------------------------------------------------- the check against the AI

def _roles_by_joint(desc_path):
    """{joint number: role} from a description and its bone graph, the graph found relative to the description."""
    d = Description.load(desc_path)
    graph = json.loads((Path(desc_path).parent / d.bones['graph']).read_text(encoding='utf-8'))
    name_of = {b['n']: b['name'] for b in graph['bones']}
    return {joint_number(name_of[int(n)]): r for n, r in d.bones['roles'].items()}


def compare(tables_path, ai_path_):
    """The tables' roles against the AI's, joint by joint: {'agree', 'differ', 'ai_cant_tell' (counts), 'differences':
    [{'joint', 'name' (the table's, or None), 'tables', 'ai'}]} (the AI's "can't tell"s are among the differences)."""
    tables, ai = _roles_by_joint(tables_path), _roles_by_joint(ai_path_)
    name = Path(tables_path).name.split('.')[0]
    names = JOINT_NAMES.get(name, {})
    out = {'agree': 0, 'differ': 0, 'ai_cant_tell': 0, 'differences': []}
    for j in sorted(tables):
        t, a = tables[j], ai.get(j, CANT_TELL)
        if t == a:
            out['agree'] += 1
            continue
        out['ai_cant_tell' if a == CANT_TELL else 'differ'] += 1
        out['differences'].append({'joint': j, 'name': names.get(j), 'tables': t, 'ai': a})
    return out


def describe_targets(targets_dir=None, files_dir=None):
    """describe_from_tables for every prepared fighter in the targets folder, then each one's roles compared with the
    AI's description where there is one. Writes <folder>/tables-against-ai.json: per fighter, the comparison (or why
    it failed). Returns {<Name>.glb: 'done' | 'failed: <reason>'}, as run.describe_targets does."""
    from ..facts.fighters import in_scope
    folder = _folder(targets_dir, TARGETS_VAR, 'Melee targets')
    report, record = {}, {}
    for f in in_scope():
        glb = folder / f'{f.name}.glb'
        if not glb.is_file():
            continue
        try:
            describe_from_tables(f.name, folder, files_dir)
            report[glb.name] = 'done'
            ai = ai_path(glb)
            record[f.name] = compare(description_path(glb), ai) if ai.is_file() else {'ai': 'none'}
        except (Failed, OSError, KeyError, ValueError) as e:
            report[glb.name] = f'failed: {e}'
            record[f.name] = {'failed': str(e)}
        print(f'{glb.name}: {report[glb.name]}', flush=True)
    (folder / 'tables-against-ai.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    return report
