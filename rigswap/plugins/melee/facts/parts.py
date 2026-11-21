"""Kept vanilla places, measured in neutral v1.02 costumes on 2026-10-10.

Sources (read-only untangle/app/backend):
  randall/model_import/studio/keep_vanilla.py:plan
  shared/modellab/vanilla_parts.py:inventory, resolve
  shared/modellab/visibility_contract.py:body_state_kinds
  shared/modellab/mesh_edits/fighter_props.py:fighter_prop_groups

Only inventory defaults are kept: weapons and their stowed forms, sheaths,
move-only props and replacement body objects. Ordinary worn pieces (including
default hats) and same-body alternates are not defaults. A row keeps whole
options of the file's visibility tables: an option is one object, and the game
shows its low-detail form far off and in the magnifier, so each kept option's
low form is in its row (read again in tests/plugins/melee/facts/test_parts.py).

The weapons a fighter fights with are also its weapon slots (Target.weapons,
targets/weapons.py, which measures them; what each one is: WEAPONS): a held
item of the download that takes one in place.held replaces that weapon, which
is then not kept (kept_places), and neither are the weapon's other forms (the
sword put away and its scabbard, the shield put away: WEAPONS' forms), so the
fighter never shows its own weapon beside the item; every other kept part
stays. The same rule as Brawl's (plugins/brawl/facts/parts.py: WEAPONS,
hidden_forms), for every slot: a weapon names its forms, none when it has none.
"""
from dataclasses import replace
import json
from pathlib import Path

from ... import Failed
from ....vocabulary import CANT_TELL


# name, places, reason; empty rows mean no default equipment to retain.
PARTS = {
    'Mario': (
    ),
    'DrMario': (
    ),
    'Luigi': (
    ),
    'Peach': (
        ('weapon', (55, 56, 57, 58, 62, 63, 64, 65), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
        ('l_arm_piece_j83', (59, 60, 61, 66, 67), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j83', (68, 69, 74, 75, 76), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j83_2', (70, 71, 72, 73, 77, 78, 79, 80), 'the game only draws it during the move that uses it'),
    ),
    'Bowser': (
        ('body_state1', (65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99), 'the fighter swaps its body for this object in some moves (it does not follow the joints that drive the body; it is bound rigidly rather than enveloped like a body; it is not the size of the body; it does not stand where the body stands); without it nothing is drawn then'),
    ),
    'Yoshi': (
        ('body_state1', (34, 35), 'the fighter swaps its body for this object in some moves (it does not follow the joints that drive the body; it is bound rigidly rather than enveloped like a body; it does not stand where the body stands); without it nothing is drawn then'),
    ),
    'DonkeyKong': (
    ),
    'CaptainFalcon': (
    ),
    'Ganondorf': (
        ('weapon', (82, 83, 84, 85, 86, 87, 88, 89), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
    ),
    'Falco': (
    ),
    'Fox': (
    ),
    'Ness': (
    ),
    'Kirby': (
        ('body_state2', (20, 21, 22, 31, 32, 33), 'the fighter swaps its body for this object in some moves (it is bound rigidly rather than enveloped like a body); without it nothing is drawn then'),
        ('body_state4', (23, 24, 34, 35), 'the fighter swaps its body for this object in some moves (it is bound rigidly rather than enveloped like a body); without it nothing is drawn then'),
        ('body_state3', (25, 26, 36, 37), 'the fighter swaps its body for this object in some moves (it is bound rigidly rather than enveloped like a body); without it nothing is drawn then'),
        ('body_state5', (27, 28, 29, 38, 39, 40), 'the fighter swaps its body for this object in some moves (it is bound rigidly rather than enveloped like a body); without it nothing is drawn then'),
        ('body_state6', (30, 41), 'the fighter swaps its body for this object in some moves (it is bound rigidly rather than enveloped like a body); without it nothing is drawn then'),
    ),
    'Samus': (
        ('body_state1', (19, 26), 'the fighter swaps its body for this object in some moves (it does not follow the joints that drive the body; it is bound rigidly rather than enveloped like a body; it is not the size of the body; it does not stand where the body stands); without it nothing is drawn then'),
        ('body_state2', (38, 39), 'the fighter swaps its body for this object in some moves (it does not follow the joints that drive the body; it is bound rigidly rather than enveloped like a body; it is not the size of the body; it does not stand where the body stands); without it nothing is drawn then'),
    ),
    'Zelda': (
    ),
    'Sheik': (
        ('torso_prop_j4', (62,), 'the game only draws it during the move that uses it'),
    ),
    'Link': (
        ('sword', (71, 72, 73, 74, 75), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
        ('shield', (76, 77, 78, 79, 80, 81, 82), 'the fighter fights with it: it blocks shots with it on its arm'),
        ('scabbard', (83, 84, 85), 'the sword put away sits in it; it goes with the sword'),
        ('shield_stowed', (86, 87, 88, 89, 90, 91, 92), 'the shield put away; it goes with the shield'),
        ('sword_stowed', (93, 94), 'the sword put away; it goes with the sword'),
    ),
    'YoungLink': (
        ('sword', (57, 58, 59, 60, 61, 62), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
        ('shield', (63, 64, 65, 66, 67, 68), 'the fighter fights with it: it blocks shots with it on its arm'),
        ('scabbard', (69, 70, 71, 72), 'the sword put away sits in it; it goes with the sword'),
        ('shield_stowed', (73, 74, 75, 76, 77, 78), 'the shield put away; it goes with the shield'),
        ('sword_stowed', (79, 80, 81, 82), 'the sword put away; it goes with the sword'),
    ),
    'Pichu': (
    ),
    'Pikachu': (
    ),
    'Jigglypuff': (
    ),
    'Mewtwo': (
    ),
    'GameAndWatch': (
        ('util_prop_j1_7', (43, 44, 45, 46, 47, 48, 49, 50, 51), 'the game only draws it during the move that uses it'),
        ('util_prop_j1_4', (52, 55, 61, 67), 'the game only draws it during the move that uses it'),
        ('util_prop_j1_5', (53, 56, 62, 68), 'the game only draws it during the move that uses it'),
        ('util_prop_j1_6', (54, 57, 63, 69), 'the game only draws it during the move that uses it'),
        ('util_prop_j1', (58, 64), 'the game only draws it during the move that uses it'),
        ('util_prop_j1_2', (59, 65), 'the game only draws it during the move that uses it'),
        ('util_prop_j1_3', (60, 66), 'the game only draws it during the move that uses it'),
        ('torso_prop_j7', (70, 72), 'the game only draws it during the move that uses it'),
        ('torso_prop_j7_2', (71,), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17', (73, 91), 'the game only draws it during the move that uses it'),
        ('weapon', (74, 92), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
        ('l_arm_prop_j17_9', (75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17_2', (86, 93), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17_3', (87, 94), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17_4', (88, 95, 96), 'the game only draws it during the move that uses it'),
        ('weapon_2', (89, 97), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
        ('l_arm_prop_j17_5', (90, 98), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17_6', (99, 103), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17_7', (100, 104), 'the game only draws it during the move that uses it'),
        ('l_arm_prop_j17_8', (101, 105), 'the game only draws it during the move that uses it'),
        ('weapon_3', (102, 106), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
        ('head_prop_j21_2', (110,), 'the game only draws it during the move that uses it'),
        ('head_prop_j21', (111, 112), 'the game only draws it during the move that uses it'),
        ('r_arm_prop_j32_2', (113,), 'the game only draws it during the move that uses it'),
        ('r_arm_prop_j32', (114, 115), 'the game only draws it during the move that uses it'),
    ),
    'Marth': (
        ('scabbard', (63, 64, 65, 66, 67, 68, 69, 70, 71, 72), 'the sword put away sits in it; it goes with the sword'),
        ('sword_stowed', (73, 74), 'the sword put away; it goes with the sword'),
        ('sword', (78, 79, 80, 81, 82, 83), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
    ),
    'Roy': (
        ('scabbard', (36, 76, 77, 78, 80, 81), 'the sword put away sits in it; it goes with the sword'),
        ('sword_stowed', (79, 82, 83, 84), 'the sword put away; it goes with the sword'),
        ('sword', (89, 90, 91, 92, 93, 94), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
    ),
    'Popo': (
        ('weapon', (37, 38, 39, 40, 41, 42), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
    ),
    'Nana': (
        ('weapon', (37, 38, 39, 40, 41, 42), 'the fighter fights with it: without it the hand swings at air while the trails and hitboxes stay'),
    ),
}

KEPT = {name: tuple(sorted({p for _, places, _ in rows for p in places})) for name, rows in PARTS.items()}

# The triangles each part draws: its places' triangles in the vanilla costume exported with every DObj named by place
# (HSDRawViewer --model export --names place), from the neutral v1.02 files, 2026-10-12; each fighter's add up to the
# KEPT_TRIANGLES measured the same way on 2026-10-10. KEPT_TRIANGLES, the whole of a fighter's, reserves all copies
# and mutually exclusive states: conservative for the drawing budget, and counting every retained resource. A part the
# costume does not keep (a replaced weapon and its forms) gives its triangles back (not_kept).
PART_TRIANGLES = {
    'Peach': {'weapon': 360, 'l_arm_piece_j83': 82, 'l_arm_prop_j83': 344, 'l_arm_prop_j83_2': 375},
    'Bowser': {'body_state1': 2070},
    'Yoshi': {'body_state1': 2224},
    'Ganondorf': {'weapon': 282},
    'Kirby': {'body_state2': 376, 'body_state4': 88, 'body_state3': 1220, 'body_state5': 292, 'body_state6': 88},
    'Samus': {'body_state1': 918, 'body_state2': 2048},
    'Sheik': {'torso_prop_j4': 102},
    'Link': {'sword': 287, 'shield': 180, 'scabbard': 53, 'shield_stowed': 180, 'sword_stowed': 217},
    'YoungLink': {'sword': 274, 'shield': 122, 'scabbard': 154, 'shield_stowed': 122, 'sword_stowed': 240},
    'GameAndWatch': {'util_prop_j1_7': 66, 'util_prop_j1_4': 32, 'util_prop_j1_5': 32, 'util_prop_j1_6': 32,
                     'util_prop_j1': 12, 'util_prop_j1_2': 12, 'util_prop_j1_3': 12, 'torso_prop_j7': 20,
                     'torso_prop_j7_2': 10, 'l_arm_prop_j17': 41, 'weapon': 8, 'l_arm_prop_j17_9': 342,
                     'l_arm_prop_j17_2': 10, 'l_arm_prop_j17_3': 72, 'l_arm_prop_j17_4': 76, 'weapon_2': 83,
                     'l_arm_prop_j17_5': 147, 'l_arm_prop_j17_6': 4, 'l_arm_prop_j17_7': 4, 'l_arm_prop_j17_8': 4,
                     'weapon_3': 64, 'head_prop_j21_2': 182,
                     'head_prop_j21': 220, 'r_arm_prop_j32_2': 120, 'r_arm_prop_j32': 147},
    'Marth': {'scabbard': 93, 'sword_stowed': 134, 'sword': 173},
    'Roy': {'scabbard': 143, 'sword_stowed': 152, 'sword': 206},
    'Popo': {'weapon': 166},
    'Nana': {'weapon': 166},
}
KEPT_TRIANGLES = {name: sum(parts.values()) for name, parts in PART_TRIANGLES.items()}


# Each part a fighter fights with (the rows of PARTS whose reason is that the fighter fights with it): what it is
# ('kind', in plain words, which place.held matches against what a download holds), and its other forms ('forms': the
# rows of PARTS that draw the same weapon elsewhere, which the game shows in place of the hand's; none when it has
# none). Looked at, 2026-10-10: each part's places drawn alone from the vanilla v1.02 costume with every DObj exported
# by place (HSDRawViewer --names place). Peach's four places are one golf club. Mr. Game & Watch's are flat and
# untextured: 89 a key, 106 a frying pan (a round pan on a short handle), and 74 two flat bars side by side, which could
# not be told. Link's, Young Link's, Marth's and Roy's swords have one other form, the sword put away in its scabbard
# (sword_stowed), and the scabbard goes with them (its reason: the sword put away sits in it), so a model's sword never
# leaves it empty on a model that has none (Cloud on Marth, 10-28); its bone stays, for a model's own scabbard to ride
# (correspond._free_bones). Link's and Young Link's shields have one, the shield put away (10-28: BB Link's own shield
# stayed in his hand beside Link's). Ganondorf's sword, drawn only in the moves that swing it, and the others have
# none. When a held item takes a slot, the weapon's forms are not kept either (hidden_forms): the fighter never shows
# its own weapon beside the item, as in Brawl (plugins/brawl/facts/parts.py). And whether the game swings it ('swung'):
# a swung weapon's trails and hitboxes are sized to it, so a prop taking its slot is fitted to it
# (correspond._into_slot); a shield is carried on the arm, and a prop taking its slot is held as the model holds it.
WEAPONS = {
    'Peach': {'weapon': {'kind': 'golf club', 'forms': (), 'swung': True}},
    'Ganondorf': {'weapon': {'kind': 'sword', 'forms': (), 'swung': True}},
    'Link': {'sword': {'kind': 'sword', 'forms': ('sword_stowed', 'scabbard'), 'swung': True},
             'shield': {'kind': 'shield', 'forms': ('shield_stowed',), 'swung': False}},
    'YoungLink': {'sword': {'kind': 'sword', 'forms': ('sword_stowed', 'scabbard'), 'swung': True},
                  'shield': {'kind': 'shield', 'forms': ('shield_stowed',), 'swung': False}},
    'GameAndWatch': {'weapon': {'kind': CANT_TELL, 'forms': (), 'swung': True},
                     'weapon_2': {'kind': 'key', 'forms': (), 'swung': True},
                     'weapon_3': {'kind': 'frying pan', 'forms': (), 'swung': True}},
    'Marth': {'sword': {'kind': 'sword', 'forms': ('sword_stowed', 'scabbard'), 'swung': True}},
    'Roy': {'sword': {'kind': 'sword', 'forms': ('sword_stowed', 'scabbard'), 'swung': True}},
    'Popo': {'weapon': {'kind': 'hammer', 'forms': (), 'swung': True}},
    'Nana': {'weapon': {'kind': 'hammer', 'forms': (), 'swung': True}},
}


def for_symbol(symbol):
    from .fighters import FIGHTERS
    return next((f.keep for f in FIGHTERS if f.symbol == symbol), ())


def reserve(tables, places):
    return replace(tables, reserved=frozenset(places))


def hidden_forms(name, replaced=()):
    """The parts that are the other forms of the weapons ``replaced`` (each a weapon slot a held item of the build
    took, as weapons_replaced.json has it: its 'part'), WEAPONS' forms: not kept, so fighter ``name`` never shows its
    own weapon beside the item."""
    return tuple(form for r in replaced for form in WEAPONS.get(name, {}).get(r.get('part'), {}).get('forms', ()))


def not_kept(name, replaced=()):
    """{part: {'places', 'triangles'}}: the vanilla parts fighter ``name``'s costume does not keep, the weapons
    ``replaced`` and their other forms (hidden_forms), with their places (PARTS) and triangles (PART_TRIANGLES)."""
    rows = {part: places for part, places, _ in PARTS.get(name, ())}
    parts = [r.get('part') for r in replaced] + list(hidden_forms(name, replaced))
    return {p: {'places': list(rows[p]), 'triangles': PART_TRIANGLES[name][p]} for p in parts if p in rows}


def kept_places(f, build_dir):
    """(the places of ``f.keep`` a build keeps, the weapons it replaced). A weapon whose slot a held item of the build
    took (place.held wrote ``<build_dir>/weapons_replaced.json``, each entry the slot as write_weapons wrote it) is not
    kept, nor are its other forms (hidden_forms): their places are blank, as any vanilla part the build replaces.
    Failed when an entry's places are not among the fighter's kept places."""
    path = Path(build_dir) / 'weapons_replaced.json'
    replaced = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
    gone = set()
    for r in replaced:
        places = set(r.get('places') or ())
        if not places or not places <= set(f.keep):
            raise Failed(f"{path.name}: {f.name}'s {r.get('part')} has the places {sorted(places)}, which are not "
                         f"among its kept places {list(f.keep)}")
        gone |= places
    gone |= {p for part in not_kept(f.name, replaced).values() for p in part['places']}
    return tuple(p for p in f.keep if p not in gone), replaced
