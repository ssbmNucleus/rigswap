"""The chains of each fighter's joints, and the roles a joint's name would give wrongly: tables that targets/bones.py
reads to give each joint of a fighter's target its role (joint_roles). Where the game's physics strands are read
from, and what is decided about the rest: targets/bones.py's docstring.

  The tails. The old importer's table (randall model_import/fighter_body.py TAILS), checked against the joint trees.
"""
from ..targets.roles import NONE


# The fighters' chains, by joint number, each from the root of the chain out: (kind, joints). A chain the game's
# physics moves (Pl<code>.dat ftData +0x2C, see targets/bones.py: dynamics) is listed with the joints above it that
# skin vertices up to the part it hangs from, and its tip. A strand's root that skins nothing stays none.
CHAINS = {
    # The coat's tails: dynamics move 60-61, 63-64, 66-67, 76-77, 79-80 and 82-83; 59, 62, 65, 75, 78 and 81 hang
    # from carriers on the thighs (58, 74).
    'DrMario': [('other', [59, 60, 61]), ('other', [62, 63, 64]), ('other', [65, 66, 67]),
                ('other', [75, 76, 77]), ('other', [78, 79, 80]), ('other', [81, 82, 83])],
    # The dress: eight strands, dynamics 19-23, 25-29, ... 61-65, from carriers (18, 24, ...) that skin nothing, on
    # 17, which skins the waist. The hair: dynamics 88-92 (Peach), 90-93 (Zelda).
    'Peach': [('other', list(range(s, s + 5))) for s in (19, 25, 31, 37, 43, 49, 55, 61)]
             + [('hair', [88, 89, 90, 91, 92])],
    'Zelda': [('other', list(range(s, s + 5))) for s in (19, 25, 31, 37, 43, 49, 55, 61)]
             + [('hair', [90, 91, 92, 93])],
    # Tail: the old importer's TAILS. Hair: dynamics 49-50.
    'Bowser': [('tail', [24, 25, 26, 27, 28]), ('hair', [49, 50])],
    # The tie: dynamics 66-71.
    'DonkeyKong': [('other', [66, 67, 68, 69, 70, 71])],
    # The cape: dynamics 38-41, 43-46, 48-51, from 37, 42 and 47 on the chest.
    'Ganondorf': [('cape', [37, 38, 39, 40, 41]), ('cape', [42, 43, 44, 45, 46]), ('cape', [47, 48, 49, 50, 51])],
    'Marth': [('cape', [44, 45, 46, 47, 48]), ('cape', [49, 50, 51, 52, 53]), ('cape', [54, 55, 56, 57, 58])],
    'Roy': [('cape', [46, 47, 48, 49, 50]), ('cape', [51, 52, 53, 54, 55]), ('cape', [56, 57, 58, 59, 60])],
    # Tail: dynamics 17-20 (and the old importer's TAILS). Ears: one joint each on top of the head.
    'Fox': [('tail', [17, 18, 19, 20]), ('ear', [50]), ('ear', [52])],
    # Tail: one joint behind the hips where Fox's starts.
    'Falco': [('tail', [17])],
    # Tail: dynamics 19-26. The tube from the back of the chest: 48-51 (47, its root, skins nothing).
    'Mewtwo': [('tail', [19, 20, 21, 22, 23, 24, 25, 26]), ('other', [48, 49, 50, 51])],
    # Tail: the old importer's TAILS. Fingers: the table names only each finger's second joint (L1stNb ...); the
    # first is the unnamed joint above it.
    'Yoshi': [('tail', [19, 20, 21])] + [('finger', [a, a + 1]) for a in (29, 31, 33, 35, 59, 61, 63, 66)],
    # Tails: the old importer's TAILS. Ears: two joints each on top of the head.
    'Pikachu': [('tail', [47, 48, 49, 50]), ('ear', [17, 18]), ('ear', [20, 21])],
    'Pichu': [('tail', [43]), ('ear', [18, 19]), ('ear', [21, 22])],
    # The curl on top: dynamics 7-9, and 10, its tip. Ears: one joint each.
    'Jigglypuff': [('hair', [7, 8, 9, 10]), ('ear', [12]), ('ear', [26])],
    # The cap's long tip: dynamics 41-44 (Link), 43-46 (Young Link).
    'Link': [('other', [41, 42, 43, 44])],
    'YoungLink': [('other', [43, 44, 45, 46])],
}


# Roles the rules would give wrongly (CHAINS above; the joint's name, targets/bones.py ROLES), by joint number,
# each with why.
OWN_ROLES = {
    # NeckN carries the whole head (796 and 452 vertices in the supplied Pikachu/Pichu.bones.json, 2026-10-10),
    # and the PlCo.dat table has no HeadN. These counts describe the prepared meshes, not a game limit.
    'Pikachu': {14: 'head'},
    'Pichu': {15: 'head'},
    # The table names no neck; HeadN hangs from 37, which skins the neck.
    'Yoshi': {37: 'neck'},
    # The table names no neck and no head: 53 and 54 are the joints under 52 (a carrier on the chest) that skin them,
    # where Captain Falcon's NeckN and HeadN are on the same skeleton.
    'Ganondorf': {53: 'neck', 54: 'head'},
    # The arm cannon: the table has no RHandN; 50 is at the wrist, carrying the cannon's pieces (52-56).
    'Samus': {50: 'right_hand'},
    # His flat hands are drawn on the item joints (611 and 169 vertices shown at spawn, when he holds nothing).
    'GameAndWatch': {17: 'left_hand', 32: 'right_hand'},
    # Held things and scabbards: the sword and shield in the hands, the sword and shield on the back (Link, Young
    # Link), the sword in the right hand and the scabbard on the hip (Marth, Roy).
    'Link': {24: NONE, 25: NONE, 26: NONE, 67: NONE, 69: NONE, 70: NONE, 71: NONE},
    'YoungLink': {26: NONE, 27: NONE, 28: NONE, 71: NONE, 73: NONE, 74: NONE, 75: NONE},
    'Marth': {19: NONE, 20: NONE, 74: NONE, 75: NONE, 76: NONE},
    'Roy': {21: NONE, 22: NONE, 76: NONE, 77: NONE, 78: NONE},
    # The hammer: 15, under the left hand's item joint, skins it and nothing else.
    'Popo': {15: NONE},
    'Nana': {15: NONE},
}
