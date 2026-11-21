"""The vocabulary: what the AI fills in for a download and what a plugin supplies for each target. Same shape both sides.

Code only ever reads this. The AI decides what things are; code never re-decides. An unsure answer keeps its
alternatives, and each alternative a build can take alone (a joint's: Description.taking) becomes one more build. See
docs/pipeline.html, "The vocabulary".

Frames. ``up`` and ``front`` are axes of the loaded file, as load wrote it: they say how to turn it upright. Every
position in a description is in the oriented frame, the one orient leaves 2-describe/oriented.blend in: +z up, facing
-y, the character's left on +x, in the file's units. That is the joints, the bone heads and tails in the bone graph
(bones.json) and the pieces' boxes and centres (pieces.json), because every round after round 1, and every build
step, works on oriented.blend or a model made from it. A target's description is the same; the build reads the
target's .glb as Blender imports it, so a target comes in upright (a Melee fighter's .glb does)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

from .library import moves_as as _moves_as, pieces as _pieces
from .library.body_types import TYPES as _TYPES

# Body types are a library (rigswap.library.body_types): each lists the parts it has, from the hips outward. These are
# their names and parts.
BODY_TYPES = {name: t.parts for name, t in _TYPES.items()}

AXES = ('+x', '-x', '+y', '-y', '+z', '-z')
HAS = ('tail', 'wings', 'extra_head')
HANDS = ('left', 'right', 'both')      # the character's own hands, as it sees them
# How the character stands in its file: its left and right sides mirror each other (neutral: a T-pose, an A-pose)
# or they don't (posed: an action stance). A joint is mirrored from the other side only on a neutral model.
POSES = ('neutral', 'posed')
# What another model file in the download is, beside the character's file.
OTHER_FILES = ('same_character_other_format', 'low_detail_copy', 'alternate_form', 'part_of_character', 'effect',
               'prop', 'collision', 'physics', 'cutscene_variant', 'other_character')
CANT_TELL = "can't tell"               # where the question allows it; that answer takes the code default

MOVES_AS = ('body', 'rigid', 'hangs', 'held', 'layer')
BELONGS = tuple(_pieces.BELONGS)       # rigswap.library.pieces: one entry each
CHAIN_KINDS = ('tail', 'wing', 'ear', 'hair', 'cape', 'finger', 'other')


@dataclass
class Piece:
    id: int
    moves_as: str                       # one of MOVES_AS
    anchored_to: list[str] | str        # the parts of the body type it moves with (several when it spans parts),
                                        # or CANT_TELL
    belongs: str = 'character'          # one of BELONGS
    layer_of: int | None = None         # when belongs == 'layer_of': the piece it is a layer over
    copy_of: int | None = None          # when belongs == 'copy' (a lower-detail copy): the piece it copies, when
                                        # the AI could tell
    alternate_of: int | None = None     # when belongs == 'alternate_of': the piece seen in play that this piece is an
                                        # alternate form of (the same surface with another texture); from round 2's
                                        # twins answer (Description.twins)
    sure: bool = True

    chain_kind: str | None = None       # round 2's kind for a hanging piece; absent on older descriptions
    continues: int | None = None        # a hanging piece: the hanging piece it continues, whose bottom end its top
                                        # lies on (describe.pieces.continuing); it hangs from that piece

    # What the answers make a piece, defined once in rigswap.library.moves_as and asked by every step.
    def is_layer(self):
        """Solved as a layer: it moves as a layer, or it is a rendering layer of a piece it names."""
        return _moves_as.is_layer(self)

    def is_rendering_layer(self):
        """A rendering layer of another piece (belongs layer_of: an edge-line or shading pass over it)."""
        return _moves_as.is_rendering_layer(self)

    def is_held(self):
        """A held item: it belongs held or moves as held."""
        return _moves_as.is_held(self)


@dataclass
class Joint:
    pos: list[float]                    # in the oriented frame, as every position in a description
    sure: bool = True
    alt: list[float] | None = None      # an unsure joint's other position; one more build (Description.taking)


@dataclass
class Chain:
    kind: str                           # one of CHAIN_KINDS
    bones: list[int]                    # the bones' numbers in the bone graph, from the root of the chain out
    root: list[str] | str               # the part it hangs from: the role of the nearest bone above it that covers a
                                        # part, so a list of parts when that bone covers several (see
                                        # Description.bones), or CANT_TELL


@dataclass
class Description:
    body_type: str
    up: str                             # '+y', '-z', ...: an axis of the loaded file
    front: str                          # the same
    pose: str = CANT_TELL               # one of POSES, or CANT_TELL; round 1's answer about the picked file
    has: list[str] = field(default_factory=list)                 # from HAS
    holds: list[dict] = field(default_factory=list)              # {'what': str, 'hand': one of HANDS,
                                                                 # 'placed_apart': bool: the file places it away
                                                                 # from the hand} from round 1; the held ask adds
                                                                 # 'handle' (a point at the item's end that is its
                                                                 # handle, or None: can't tell) once round 2 has said
                                                                 # which pieces are the item
    pieces: list[Piece] = field(default_factory=list)
    # The twins ask's answer for each twin set (pieces with the same triangles at the same place): which piece is the
    # surface seen in play, and what each other one is: {'pieces': [numbers], 'seen': a number of the set or
    # CANT_TELL, 'others': {number as text: 'shell' | 'alternate' | 'copy'}}. Drop keeps the piece seen in play, or,
    # where the ask could not tell, the one that draws the surface best.
    twins: list[dict] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)             # the pieces drop took out: effects, stands,
                                                                 # copies and alternate forms
    joints: dict[str, Joint] = field(default_factory=dict)       # joint name -> joint, in the oriented frame
                                                                 # (see the module's docstring)
    bones: dict = field(default_factory=dict)                    # {'graph': path of bones.json, relative to
                                                                 # this file, 'roles': {bone number as text:
                                                                 # part | [part, part, ...] | chain kind |
                                                                 # 'none' | CANT_TELL}}
    # A role that is a list is one bone that covers several parts, kept as the AI gave it: two or more parts, in
    # order from the body outward, each once. One bone from hip to ankle is ['left_thigh', 'left_shin']; a single
    # spine bone from the hips to the neck is ['spine', 'chest']. The first part starts at the bone's head; the build
    # reads the bone as one source segment over those parts, in that order. A list of one part is kept as that part's
    # text, so a role that is a list always has two parts or more. A list across groups of the body (the trunk with
    # an arm, both legs) is a block bone that covers none of them: its role is 'none', and 'blocks': {bone number as
    # text: {'said': the list, 'groups': the groups it spans, 'why'}} keeps the answer (describe.bones.blocks).
    chains: list[Chain] = field(default_factory=list)
    details: dict = field(default_factory=dict)                  # fingers, eyes, mouth; only when present
    model: str | None = None                                     # which file in the download, when there were several
    others: dict[str, str] = field(default_factory=dict)         # every other file in the download -> OTHER_FILES
    alt: dict[str, str] = field(default_factory=dict)            # up, front or body_type -> the other plausible
                                                                 # answer; body_type gets a second description;
                                                                 # up/front stay recorded (Description.taking)
    from_default: list[str] = field(default_factory=list)        # the answers the AI couldn't give, so the code
                                                                 # default stands ('front', 'others.3', ...)
    # The facts whose ask failed even when sent again (rigswap.ai: no usable answer twice), so the code default stands
    # in for an answer: 'pieces.12' (every field of piece 12), 'twins.6,7', 'holds.0.handle'. Each is in from_default
    # too, beside the AI's own "can't tell"s. A defaulted fact is no reading: a defaulted piece is unsure (sure False),
    # every reader takes the fact as unsure, and the summary lists it as defaulted.
    defaulted: list[str] = field(default_factory=list)
    joint_mirrors: dict[str, str] = field(default_factory=dict)  # reflected joint -> answered joint; one fact

    def parts(self):
        return BODY_TYPES[self.body_type]

    def unsure(self):
        """Every unsure fact, each with its alternative: (kind, which, alternative), kept for the record: step 3
        builds once, with every answer as given."""
        out = [('field', f, value) for f, value in self.alt.items()]
        for part, j in self.joints.items():
            if not j.sure and j.alt and part not in self.joint_mirrors:
                out.append(('joint', part, j.alt))
        for p in self.pieces:
            if not p.sure:
                out.append(('piece', p.id, None))
        return out

    def taking(self, fact):
        """A copy of this description with ``fact`` (one of unsure()) at its alternative and every other fact as it
        is: a joint at its other position, still unsure, with the position it had as its alternative.

        Only a joint can be taken alone. Up and front set the frame every position in the description is in (orient
        turns the model by them before any picture of round 2 on is drawn), and every part, anchor and role is one of
        the body type's, so an alternative up, front or body type leaves no other answer true: it needs describe again
        from orient on. A piece's unsure answer keeps no alternative. Those raise ValueError saying so."""
        kind, which, alternative = fact
        if kind == 'joint':
            out = copy.deepcopy(self)
            out.joints[which] = Joint(pos=list(alternative), sure=False, alt=list(self.joints[which].pos))
            for reflected, source in self.joint_mirrors.items():
                partner = reflected if which == source else source if which == reflected else None
                if partner in self.joints and self.joints[partner].alt is not None:
                    joint = self.joints[partner]
                    out.joints[partner] = Joint(pos=list(joint.alt), sure=False, alt=list(joint.pos))
            return out
        if kind == 'field' and which in ('up', 'front'):
            raise ValueError(f'{which} {alternative} instead of {getattr(self, which)}: every position in the '
                             f'description is in the frame {self.up} up, {self.front} front made, so it needs '
                             f'describe again from orient on, not one more build')
        if kind == 'field':
            raise ValueError(f'{which} {alternative} instead of {getattr(self, which)}: every part, anchor and role '
                             f'in the description is one of {self.body_type}\'s, so it needs describe again from '
                             f'round 2 on, not one more build')
        raise ValueError(f'{kind} {which}: its unsure answer keeps no alternative to build')

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding='utf-8')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        d.pop('hands', None)        # round 4's open or closed hands, which nothing read: kept by older descriptions
        d['pieces'] = [Piece(**p) for p in d.get('pieces', [])]
        d['joints'] = {k: Joint(**v) for k, v in d.get('joints', {}).items()}
        d['chains'] = [Chain(**c) for c in d.get('chains', [])]
        return cls(**d)


def roles_by_number(roles):
    """A bone graph's roles, {bone number: role}, with every key a number: given by number (as round 3 and a plugin's
    tables make them) or by number as text (as a description stores them, JSON's keys being text), the same roles.
    Every reader of roles reads them through this, once, so either form finds the same bones; None is {}."""
    return {int(n): role for n, role in (roles or {}).items()}


def role_parts(role):
    """What a role names, as a list, in its order: a role that is a list (one bone across several parts, see
    Description.bones) as given, any other (a part, a chain kind, 'none', CANT_TELL) as a list of one. A chain's root
    is a role too (Chain.root). A role that may be a list is no key and no member of a set: every reader that keys by
    a role, or asks whether a role is among some parts, reads it through this, and a reader that needs one part reads
    its first, where the bone starts (correspond.skeleton's 'role')."""
    return list(role) if isinstance(role, (list, tuple)) else [role]


@dataclass
class Target:
    """What a plugin supplies for one character: its skeleton in rest pose, its original mesh skinned to it, and its
    description in the same vocabulary, made once and checked by hand."""
    name: str
    skeleton: Path                      # a .glb holding the armature in rest pose
    original: Path                      # a .glb holding the skinned original mesh (may be the same file)
    description: Description
    poses: list[Path] = field(default_factory=list)   # frames chosen to exercise its joints
    camera: dict = field(default_factory=dict)   # how the result's sheet looks at it (rigswap.plugins); {}: front, side
    max_influences: int | None = None   # the most bones one vertex may move with, the game's limit; None: no limit
    # Its weapon slots: one for each weapon the character fights with in a hand, as the plugin measured it on its own
    # model. Each {'kind': what it is in plain words ('sword'), or CANT_TELL; 'hand': the part of the body type it is
    # held in ('left_hand'); 'bone': the bone it is bound to, which may be one under the hand's, and which an item
    # taking the slot is pinned to; 'grip': the middle of its grip end, the end in the hand, and 'tip_direction': from
    # there toward its tip (a unit vector), and 'flat': across it, along its width (a unit vector, either way along
    # it), all in the target's frame, as a description's positions (+z up, facing -y); 'length'; and anything else the plugin keeps of it (where it is, how it was measured), passed back to the plugin
    # when a held item takes the slot (place.held, weapons_replaced.json)}. Empty when the plugin gives none.
    weapons: list[dict] = field(default_factory=list)
    # Its item joints: {a hand part of its body type ('right_hand'): the bone the game carries what that hand holds on,
    # a joint under the hand's bone, in its fist}, as the plugin knows them from the game. A held item no weapon slot
    # takes is placed with its grip there and pinned to it (place.held); a hand with none, on its hand bone's head.
    # Empty when the plugin names none.
    item_joints: dict = field(default_factory=dict)
