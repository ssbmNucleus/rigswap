"""Body types: one entry per body type, holding everything the steps know about it (docs/pipeline.html, "Body types are
a library: each one lists its parts and joints. The biped row is hips, spine, chest, neck, head, and per side shoulder,
upper arm, forearm, hand, thigh, shin, foot. A ball is body and head, with optional stubs. A new body type is a new
row."). Ball stubs have no elbow or knee bend; a round trunk with articulated limbs is a biped or quadruped.
Code reads the entry; no step names a part of one body type in its own code.

An entry (BodyType):
    parts       its parts, from the hips outward: the body's order (vocabulary.BODY_TYPES is these)
    chains      how the parts hang together: (the part a chain hangs from, the chain's parts from root to tip). The
                first chain is the body's own line and hangs from nothing. A part runs from its start to the start of
                the next part in its chain; the last part of a chain has no next.
    joints      its joints: (name, the joint it hangs from, what it is in plain words). "{side}" stands for left and
                for right: one joint each. The first joint is the root.
    part_ends   each part's two joints, "{side}" as in joints. Where the body type has two parts on one stretch (a
                biped's hips and spine, its neck and head), both name that stretch; the build divides it.
    narrowed    the parts that start at the narrowing of the stretch they share with the part before them, measured on
                the model (bind.narrowing): a biped's or a quadruped's head starts where its neck is narrowest, or
                at the neck joint when the model has no neck (no section narrower than both below and above it). A
                shared stretch with no part here is one bone across its parts (bind.bones_from_joints)
    groups      round 4's asks: (name, its joints, the closer grid that goes with it and the joints that may be
                answered there, or None)
    has_chains  what round 1 may say a model has (vocabulary.HAS) that hangs from its body as a chain, with no part of
                its own: the chain's kind, the part it hangs from, its joints from root to tip
    top_joints  the historical top-view subset, by name suffix, kept for reproducing older lab questions;
                current round 4 asks every joint in every view and allows hidden answers
    required    the parts a model can't be built without: the fit fails when the source has no bone or joint for one,
                and none is folded into a neighbour as a missing connector is
    fit         how each part is fitted (fit.segments): its rule in FIT, which says whether it stretches within its
                span (a link doesn't) and how it is fitted when nothing further lands it, the last of its chain. A
                part not named here is 'span'. Naming a part's rule also says what it is: the head, its neck, a hand,
                a foot, a link.
    height      how its height is measured (fit.height_points): the joints that are its ankles, the joint at the top
                of its head
    poses       its plain test poses (bend), for a target with none of its own: (name, what it does, [(parts, turned
                toward, degrees)])
    pose_moves  {part: another part a plain pose turns with it, when the other's bone doesn't hang below its own}

FIT, the rules: a part that is fitted within its span (both ends land) is turned and stretched along its length so
both its ends land on the target's, unless it is a link:
    link    a short link between joints that carries the next part out from the body (a biped's shoulders, its hips
            up to its spine): turned the way the target's link runs and moved whole, at the model's scale, so its
            child joint lands on the target's; never stretched along its length (a shoulder bone a hundredth of the
            target's would stretch its skin a hundredfold). A segment is a link only when every part in it is one:
            a shoulder that took in an upper arm with no start of its own is a limb span, and stretches
    neck    the part the head hangs from: stretched as any span, and the head after it moves whole with its stretch
            where the head's own skin meets it (fit.head_seam), so the neck's skin between the trunk and the head
            stretches and the head stays joined to the trunk by it; a head whose skin reaches down to the neck's
            start (a chin resting on the collar) moves with the trunk's end
The rules say what happens to a part that is the last of its chain on both sides, which has nothing further to land:
it lands its start, its pivot, on the target's and is never stretched: it keeps the model's one scale, a foot
scaled whole to the target's foot and moved onto the floor as its rule says, and no band of the fit's scales it either (fit.stretch's
pins: the head, the hands and the feet), and
    span    turns with the part it hangs from (a trunk or limb part that ends a chain: a ball's stub; a link too)
    hand    turns with the part it hangs from, the forearm. Its fingertip, the far end of its mesh beyond the wrist,
            and its breadth across that line are measured on every model and the target alike (fit.fingertips,
            fit.hand_breadths) and recorded against the target's, never applied
    head    keeps its orientation in the world: an upright head stays upright whatever the neck did; after a neck
            (rule 'neck') it lands where the neck's stretch puts its edge, not its start (fit.head_seam); with no
            neck span of its own it lands the target's head joint, front to back, where that sits in the
            fighter's own head (fit.head_place)
    foot    keeps its orientation in the world, so it stays flat on the floor, turned only about the vertical at its
            ankle so its ankle-to-toe line on the ground points the way the target's does (a foot turned out in a
            fighting stance faces where the target's faces; fit.facing: its toe, the far end of its mesh on the
            ground), then laid as the target's sole lies at rest (a fighter whose rest pose tips its shoe
            toe-down, levelled in its game poses, tips the model's with it; each sole's plane the one it rests on
            pressed down at its middle, fit.level); it is scaled whole about its ankle to the length of the target's foot (its heel-to-toe
            mesh extent on the ground along that line, measured on every model and the target alike; two feet
            that mirror each other at one scale, the mean of theirs), so a pose that tips the target's foot and
            toe bones carries no extra length through the floor, and moved straight down or up so its sole lies on
            the target's, the shin stretching or shortening to it; the shin's lowest stretch moves with
            it (fit.stand). Its toe, the far end of its mesh on the ground (the vertex farthest from the ankle
            across the ground, as a hand's fingertip is the far end of its mesh beyond the wrist), is measured on
            every model and the target alike (fit.foot_fronts)
"""
from __future__ import annotations

from dataclasses import dataclass, field

SIDES = ('left', 'right')
# Directions in a description's frame (+z up, facing -y, the character's left on +x), for the plain poses.
UP, DOWN, FRONT, BACK = (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0)
LEFT, RIGHT = (1, 0, 0), (-1, 0, 0)


@dataclass(frozen=True)
class FitRule:
    says: str                   # in plain words, for the fit's record
    world: bool                 # keeps its orientation in the world; else turns with the part it hangs from
    to_tip: str | None = None   # measured to its tip against the target's when both sides have one, recorded and
    #                             never applied: 'ground' (a foot: along its forward line laid flat; a limb whose last
    #                             part is one stands on the ground), 'whole' (a hand: wrist to fingertip, and across);
    #                             None: not measured
    link: bool = False          # within its span: moved whole so its child joint lands, never stretched (a link)


FIT = {
    'span': FitRule('turns with the part it hangs from, not stretched', world=False),
    'link': FitRule("within its span, turned the way the target's link runs and moved whole so its child joint lands "
                    "on the target's, never stretched along its length; the last of its chain, it turns with the "
                    "part it hangs from, not stretched", world=False, link=True),
    'hand': FitRule("turns with the forearm, its wrist on the target's, at the model's one scale: never stretched "
                    "or scaled, by the fit or its bands; its length and breadth against the target's hand are "
                    "recorded, not applied", world=False, to_tip='whole'),
    'head': FitRule("keeps its orientation in the world, at the model's one scale: never stretched or scaled, by the "
                    "fit or its bands; moved whole with its neck's stretch where its own skin meets the neck (its "
                    "start on the target's when its skin does not reach down the neck)", world=True),
    'neck': FitRule("within its span, turned and stretched so both its ends land, the head after it moving whole with "
                    "its stretch where the head's own skin meets it; the last of its chain, it turns with the part it "
                    "hangs from, not stretched", world=False),
    'foot': FitRule("keeps its orientation in the world, turned in one turn about its ankle so its sole lies as the "
                    "target's lies at rest and it points the way the target's foot does, scaled whole about its ankle "
                    "to the heel-to-toe length of the target's foot (two feet that mirror each other at one scale, "
                    "the mean of theirs), and moved straight down or up so its sole lies on the target's, the shin "
                    "stretching or shortening to it: never stretched, nor scaled by the bands", world=True,
                    to_tip='ground'),
}


@dataclass(frozen=True)
class Height:
    ankles: tuple = ()          # the joints that are its ankles: height is from their mean to the head top
    top: str = 'head_top'       # the joint at the top of its head (a model with no bones); none of a body type's
    #                             joints for a quadruped, a fish or a serpent, which have no ankles either: their height
    #                             is the mesh's z extent (fit.scale_by)


@dataclass(frozen=True)
class BodyType:
    name: str
    parts: list
    chains: list
    joints: list
    part_ends: dict
    groups: list
    narrowed: tuple = ()
    has_chains: dict = field(default_factory=dict)
    top_joints: tuple = ()
    required: tuple = ('head',)
    fit: dict = field(default_factory=dict)
    height: Height = Height()
    poses: list = field(default_factory=list)
    pose_moves: dict = field(default_factory=dict)

    def rule(self, part):
        """The fit rule (FIT) of ``part``: its own, else 'span'."""
        return FIT[self.fit.get(part, 'span')]

    @property
    def head(self):
        """The part that is the head: the one whose rule is 'head'."""
        return next(p for p, r in self.fit.items() if r == 'head')

    @property
    def hands(self):
        """The parts that are hands (rule 'hand'), left before right."""
        return [p for p in self.parts if self.fit.get(p) == 'hand']

    @property
    def feet(self):
        """The parts that are feet (rule 'foot'), in the body's order."""
        return [p for p in self.parts if self.fit.get(p) == 'foot']

    @property
    def tipped(self):
        """The parts scaled to their tip when both sides have it (a rule with to_tip: the feet, the hands), in the
        body's order."""
        return [p for p in self.parts if self.rule(p).to_tip]


# A tail, the chain a biped may have with no part of its own (has_chains).
_TAIL = {'tail': ('tail', 'hips', [
    ('tail_base', 'hips', 'where the tail joins the body, at the root of the tail'),
    ('tail_tip', 'tail_base', 'the tip of the tail, its far end')])}
# The plain test poses of a biped (bend, for a target with no poses of its own): each part's first bone turns that many
# degrees toward the direction given, about its head; 'twist' turns it about its own line. A quadruped shares the
# spine, neck and head moves; the biped's limb parts are absent and left out.
PLAIN = [
    ('squat', 'thighs forward 60°, shins back 90°, spine forward 20°',
     [(['left_thigh', 'right_thigh'], FRONT, 60), (['left_shin', 'right_shin'], BACK, 90), (['spine'], FRONT, 20)]),
    ('arms down', 'upper arms down 80° at the sides',
     [(['left_upper_arm', 'right_upper_arm'], DOWN, 80)]),
    ('elbows forward', 'forearms forward 90° at both elbows',
     [(['left_forearm', 'right_forearm'], FRONT, 90)]),
    ('wrists forward', 'hands forward 45° at both wrists',
     [(['left_hand', 'right_hand'], FRONT, 45)]),
    ('ankles up', 'feet up 30° at both ankles',
     [(['left_foot', 'right_foot'], UP, 30)]),
    ('neck forward', 'neck forward 20° at its base',
     [(['neck'], FRONT, 20)]),
    ('head nod', 'head forward 30° above the neck',
     [(['head'], FRONT, 30)]),
    ('turn', 'spine twisted 40° about its line',
     [(['spine'], 'twist', 40)]),
    # the legs spread apart: what the inner sides of the legs do (an inseam), which the squat, moving both thighs the
    # same way, never shows
    ('legs apart', 'thighs out 45° to the sides',
     [(['left_thigh'], LEFT, 45), (['right_thigh'], RIGHT, 45)]),
]

# A ball's (bend, likewise): its stubs and feet are its limbs, its body turns as a biped's spine does.
BALL_PLAIN = [
    ('stubs up', 'stubs up 60°', [(['left_stub', 'right_stub'], UP, 60)]),
    ('stubs forward', 'stubs forward 60°', [(['left_stub', 'right_stub'], FRONT, 60)]),
    ('step', 'left foot forward 40°, right foot back 40°', [(['left_foot'], FRONT, 40), (['right_foot'], BACK, 40)]),
    ('turn', 'body twisted 30° about its line', [(['body'], 'twist', 30)]),
]

TYPES = {
    'biped': BodyType(
        name='biped',
        parts=['hips', 'spine', 'chest', 'neck', 'head',
               'left_shoulder', 'left_upper_arm', 'left_forearm', 'left_hand',
               'right_shoulder', 'right_upper_arm', 'right_forearm', 'right_hand',
               'left_thigh', 'left_shin', 'left_foot',
               'right_thigh', 'right_shin', 'right_foot'],
        chains=[(None, ['hips', 'spine', 'chest', 'neck', 'head']),
                ('chest', ['left_shoulder', 'left_upper_arm', 'left_forearm', 'left_hand']),
                ('chest', ['right_shoulder', 'right_upper_arm', 'right_forearm', 'right_hand']),
                ('hips', ['left_thigh', 'left_shin', 'left_foot']),
                ('hips', ['right_thigh', 'right_shin', 'right_foot'])],
        joints=[
            ('hips', None, 'the middle of the pelvis, between the two hip joints, inside the body'),
            ('chest', 'hips', 'the middle of the body at the bottom of the rib cage, where the chest starts above the '
                              'belly, inside the body'),
            ('neck', 'chest', 'the bottom of the neck, where it meets the chest between the shoulders, inside the '
                              'body'),
            ('head_top', 'neck', 'the top of the head itself: the top of the skull, under any hair, hat or helmet'),
            ('{side}_shoulder', 'neck', 'the {side} shoulder joint, where the {side} upper arm joins the body and '
                                        'turns'),
            ('{side}_elbow', '{side}_shoulder', 'where the {side} arm bends, between the upper arm and the forearm'),
            ('{side}_wrist', '{side}_elbow', 'where the {side} hand meets the forearm'),
            ('{side}_hand_end', '{side}_wrist', 'the tip of the {side} hand: its fingertip or knuckle farthest from '
                                                'the wrist'),
            ('{side}_hip', 'hips', 'the {side} hip joint, where the {side} thigh joins the pelvis, inside the top of '
                                   'the leg'),
            ('{side}_knee', '{side}_hip', 'where the {side} leg bends, between the thigh and the shin'),
            ('{side}_ankle', '{side}_knee', 'where the {side} foot meets the shin'),
            ('{side}_toe', '{side}_ankle', 'the front tip of the {side} foot or shoe'),
        ],
        part_ends={'hips': ('hips', 'chest'), 'spine': ('hips', 'chest'), 'chest': ('chest', 'neck'),
                   'neck': ('neck', 'head_top'), 'head': ('neck', 'head_top'),
                   '{side}_shoulder': ('neck', '{side}_shoulder'),
                   '{side}_upper_arm': ('{side}_shoulder', '{side}_elbow'),
                   '{side}_forearm': ('{side}_elbow', '{side}_wrist'),
                   '{side}_hand': ('{side}_wrist', '{side}_hand_end'),
                   '{side}_thigh': ('{side}_hip', '{side}_knee'), '{side}_shin': ('{side}_knee', '{side}_ankle'),
                   '{side}_foot': ('{side}_ankle', '{side}_toe')},
        groups=[('middle', ['hips', 'chest', 'neck', 'head_top'], None),
                ('{side} arm', ['{side}_shoulder', '{side}_elbow', '{side}_wrist', '{side}_hand_end'],
                 ('{side}', ['{side}_elbow', '{side}_wrist', '{side}_hand_end'])),
                ('{side} leg', ['{side}_hip', '{side}_knee', '{side}_ankle', '{side}_toe'],
                 ('low', ['{side}_knee', '{side}_ankle', '{side}_toe']))],
        # Round 4 places the neck joint (the bottom of the neck) and the head top, and no joint between the neck and
        # the head. Cut in equal shares, the head started halfway up, at eye level on a model with no neck,
        # where every rig puts it far lower: on the 21 Melee fighters' rigs with a neck and a head bone the head bone
        # starts 10 to 27% of the way from the neck joint to the head top (median 18%). See bind.narrowing.
        narrowed=('head',),
        has_chains=_TAIL,
        # Not the hips: seen from above a leg's hip is under the belly, and the top picture put it about two cells in
        # front of the side picture's in 43 to 49 of 70 answers (never behind), which made 53 to 67% of hips unsure
        # (two candidates each). Asked in the front and side only, a hip's depth comes from the side picture alone:
        # in the lab (round 4, 2026-10-11, 15 hips with truth from two answerers) it was as close to the truth or
        # closer (median 8.3 to 10.1% of the height, against 11.1 to 11.7%) and unsure in 0 to 7% of answers.
        top_joints=('_shoulder',),
        required=('hips', 'head'),
        fit={'head': 'head', 'left_hand': 'hand', 'right_hand': 'hand', 'left_foot': 'foot', 'right_foot': 'foot',
             'hips': 'link', 'neck': 'neck', 'left_shoulder': 'link', 'right_shoulder': 'link'},
        height=Height(ankles=('left_ankle', 'right_ankle')),
        poses=PLAIN,
        pose_moves={'spine': 'chest'}),
    'quadruped': BodyType(
        name='quadruped',
        parts=['hips', 'spine', 'chest', 'neck', 'head',
               'left_foreleg_upper', 'left_foreleg_lower', 'left_forefoot',
               'right_foreleg_upper', 'right_foreleg_lower', 'right_forefoot',
               'left_hindleg_upper', 'left_hindleg_lower', 'left_hindfoot',
               'right_hindleg_upper', 'right_hindleg_lower', 'right_hindfoot'],
        chains=[(None, ['hips', 'spine', 'chest', 'neck', 'head']),
                ('chest', ['left_foreleg_upper', 'left_foreleg_lower', 'left_forefoot']),
                ('chest', ['right_foreleg_upper', 'right_foreleg_lower', 'right_forefoot']),
                ('hips', ['left_hindleg_upper', 'left_hindleg_lower', 'left_hindfoot']),
                ('hips', ['right_hindleg_upper', 'right_hindleg_lower', 'right_hindfoot'])],
        joints=[
            ('hips', None, 'the middle of the pelvis, between the two hind hip joints, inside the body'),
            ('chest', 'hips', 'the middle of the chest between the two shoulders, inside the body'),
            ('neck', 'chest', 'where the neck leaves the body, inside it'),
            ('head_end', 'neck', 'the far end of the head from the neck: the tip of the snout or nose'),
            ('{side}_shoulder', 'chest', 'where the {side} front leg joins the body'),
            ('{side}_elbow', '{side}_shoulder', 'the first bend of the {side} front leg below the shoulder'),
            ('{side}_wrist', '{side}_elbow', 'where the {side} front foot meets the front leg'),
            ('{side}_forefoot_end', '{side}_wrist', 'the tip of the {side} front foot'),
            ('{side}_hip', 'hips', 'where the {side} hind leg joins the pelvis'),
            ('{side}_knee', '{side}_hip', 'the first bend of the {side} hind leg below the hip'),
            ('{side}_hock', '{side}_knee', 'the next bend of the {side} hind leg, the heel above the hind foot'),
            ('{side}_hindfoot_end', '{side}_hock', 'the tip of the {side} hind foot'),
        ],
        part_ends={'hips': ('hips', 'chest'), 'spine': ('hips', 'chest'), 'chest': ('chest', 'neck'),
                   'neck': ('neck', 'head_end'), 'head': ('neck', 'head_end'),
                   '{side}_foreleg_upper': ('{side}_shoulder', '{side}_elbow'),
                   '{side}_foreleg_lower': ('{side}_elbow', '{side}_wrist'),
                   '{side}_forefoot': ('{side}_wrist', '{side}_forefoot_end'),
                   '{side}_hindleg_upper': ('{side}_hip', '{side}_knee'),
                   '{side}_hindleg_lower': ('{side}_knee', '{side}_hock'),
                   '{side}_hindfoot': ('{side}_hock', '{side}_hindfoot_end')},
        groups=[('middle', ['hips', 'chest', 'neck', 'head_end'], None),
                ('{side} front leg', ['{side}_shoulder', '{side}_elbow', '{side}_wrist', '{side}_forefoot_end'],
                 ('low', ['{side}_elbow', '{side}_wrist', '{side}_forefoot_end'])),
                ('{side} hind leg', ['{side}_hip', '{side}_knee', '{side}_hock', '{side}_hindfoot_end'],
                 ('low', ['{side}_knee', '{side}_hock', '{side}_hindfoot_end']))],
        has_chains={'tail': ('tail', 'hips', [
            ('tail_base', 'hips', 'where the tail joins the body, at the root of the tail'),
            ('tail_tip', 'tail_base', 'the tip of the tail, its far end')])},
        # Its neck and head share one stretch, neck -> head_end, as a biped's do: the head starts where the neck is
        # narrowest, as a biped's does (bind.narrowing). Not named, the stretch is one bone and nothing starts the head
        narrowed=('head',),
        top_joints=('_shoulder', '_hip'),
        required=('hips', 'head'),
        fit={'head': 'head', 'neck': 'neck', 'left_forefoot': 'foot', 'right_forefoot': 'foot', 'left_hindfoot': 'foot',
             'right_hindfoot': 'foot'},
        poses=PLAIN,
        pose_moves={'spine': 'chest'}),
    'ball': BodyType(
        name='ball',
        parts=['body', 'head', 'left_stub', 'right_stub', 'left_foot', 'right_foot'],
        chains=[(None, ['body', 'head']),
                ('body', ['left_stub']), ('body', ['right_stub']),
                ('body', ['left_foot']), ('body', ['right_foot'])],
        joints=[
            ('body', None, 'the middle of the round body'),
            ('head_top', 'body', 'the top of the head; when the body is the head, the top of the ball'),
            ('{side}_stub_root', 'body', 'where the {side} stub (a small arm or fin) joins the body'),
            ('{side}_stub_end', '{side}_stub_root', 'the tip of the {side} stub'),
            ('{side}_foot_root', 'body', 'where the {side} foot joins the body'),
            ('{side}_foot_end', '{side}_foot_root', 'the front tip of the {side} foot'),
        ],
        part_ends={'body': ('body', 'head_top'), 'head': ('body', 'head_top'),
                   '{side}_stub': ('{side}_stub_root', '{side}_stub_end'),
                   '{side}_foot': ('{side}_foot_root', '{side}_foot_end')},
        groups=[('middle', ['body', 'head_top'], None),
                ('{side} stub', ['{side}_stub_root', '{side}_stub_end'], ('{side}', ['{side}_stub_end'])),
                ('{side} foot', ['{side}_foot_root', '{side}_foot_end'], ('low', ['{side}_foot_end']))],
        poses=BALL_PLAIN,
        has_chains={'tail': ('tail', 'body', [
            ('tail_base', 'body', 'where the tail joins the body, at the root of the tail'),
            ('tail_tip', 'tail_base', 'the tip of the tail, its far end')])},
        fit={'head': 'head', 'left_foot': 'foot', 'right_foot': 'foot'}),
    'fish': BodyType(
        name='fish',
        parts=['head', 'body', 'tail_base', 'tail_fin', 'left_fin', 'right_fin'],
        chains=[(None, ['body', 'tail_base', 'tail_fin']),
                ('body', ['head']), ('body', ['left_fin']), ('body', ['right_fin'])],
        joints=[
            ('head_back', None, 'where the head meets the body, behind the gills, inside the body'),
            ('snout', 'head_back', 'the front tip of the head'),
            ('tail_base', 'head_back', 'where the body narrows into the tail'),
            ('tail_fin_root', 'tail_base', 'where the tail fin starts'),
            ('tail_fin_end', 'tail_fin_root', 'the far edge of the tail fin'),
            ('{side}_fin_root', 'head_back', 'where the {side} side fin joins the body'),
            ('{side}_fin_end', '{side}_fin_root', 'the tip of the {side} side fin'),
        ],
        part_ends={'head': ('head_back', 'snout'), 'body': ('head_back', 'tail_base'),
                   'tail_base': ('tail_base', 'tail_fin_root'), 'tail_fin': ('tail_fin_root', 'tail_fin_end'),
                   '{side}_fin': ('{side}_fin_root', '{side}_fin_end')},
        groups=[('middle', ['head_back', 'snout', 'tail_base', 'tail_fin_root', 'tail_fin_end'], None),
                ('{side} fin', ['{side}_fin_root', '{side}_fin_end'], None)],
        fit={'head': 'head'}),
    'serpent': BodyType(
        name='serpent',
        parts=['head', 'neck', 'body_1', 'body_2', 'body_3', 'tail'],
        chains=[(None, ['neck', 'body_1', 'body_2', 'body_3', 'tail']),
                ('neck', ['head'])],
        joints=[
            ('head_back', None, 'where the head meets the neck'),
            ('snout', 'head_back', 'the front tip of the head'),
            ('neck_end', 'head_back', 'where the neck ends and the thick of the body starts'),
            ('body_1_end', 'neck_end', 'a third of the way along the body, from the neck\'s end to where the tail '
                                       'starts'),
            ('body_2_end', 'body_1_end', 'two thirds of the way along the body, from the neck\'s end to where the tail '
                                         'starts'),
            ('tail_base', 'body_2_end', 'where the body narrows into the tail'),
            ('tail_tip', 'tail_base', 'the tip of the tail'),
        ],
        part_ends={'head': ('head_back', 'snout'), 'neck': ('head_back', 'neck_end'),
                   'body_1': ('neck_end', 'body_1_end'), 'body_2': ('body_1_end', 'body_2_end'),
                   'body_3': ('body_2_end', 'tail_base'), 'tail': ('tail_base', 'tail_tip')},
        groups=[('body', ['head_back', 'snout', 'neck_end', 'body_1_end', 'body_2_end', 'tail_base', 'tail_tip'],
                 None)],
        fit={'head': 'head'}),
}


def of(body_type):
    """The entry for ``body_type``."""
    return TYPES[body_type]


# --------------------------------------------------------------------------------------------- read off an entry

def joints_of(body_type, has=()):
    """[(name, parent, what it is)] for the body type, left before right; then the joints of each chain ``has`` (round
    1's) names that the body type has a row for (has_chains)."""
    out = []
    for name, parent, what in of(body_type).joints:
        if '{side}' in name:
            out += [(name.format(side=s), parent.format(side=s) if parent else None, what.format(side=s))
                    for s in SIDES]
        else:
            out.append((name, parent, what))
    return out + [j for _, _, joints in has_chains(body_type, has).values() for j in joints]


def has_chains(body_type, has):
    """{group: (chain kind, the part it hangs from, [(joint, parent, what it is)])} for each thing ``has`` (round 1's)
    names that the body type's has_chains has a row for. The group is its round-4 ask's name."""
    rows = of(body_type).has_chains
    return {h: rows[h] for h in has if h in rows}


def part_ends(body_type):
    """{part: (from joint, to joint)} for every part of the body type, in its order."""
    out = {}
    for part, (a, b) in of(body_type).part_ends.items():
        for s in (SIDES if '{side}' in part else (None,)):
            out[part.format(side=s)] = (a.format(side=s), b.format(side=s))
    return {p: out[p] for p in of(body_type).parts}


def groups_of(body_type):
    """[(group, [joint names], closer grid or None, [joints that may be answered in it])] for the body type, a
    "{side}" group once for the left and once for the right."""
    out = []
    for group, names, zoom in of(body_type).groups:
        for s in (SIDES if '{side}' in group else (None,)):
            zoom_name, zoom_joints = (zoom[0].format(side=s), [n.format(side=s) for n in zoom[1]]) if zoom else \
                (None, [])
            out.append((group.format(side=s), [n.format(side=s) for n in names], zoom_name, zoom_joints))
    return out


def chain_groups_of(body_type, has):
    """[(group, [joint names], None, [])] in groups_of's shape, one for each chain ``has`` names (has_chains). A
    chain's joints are asked with no closer grid."""
    return [(group, [n for n, _, _ in joints], None, [])
            for group, (_, _, joints) in has_chains(body_type, has).items()]


def zooms_of(body_type):
    """The closer grids the body type's groups use, in the order first used."""
    return list(dict.fromkeys(z for _, _, z, _ in groups_of(body_type) if z))


def limb_groups(body_type):
    """Each limb's question and parts, from the body chains and their end joints."""
    ends = part_ends(body_type)
    return {group: parts for group, names, _, _ in groups_of(body_type)
            for parent, parts in of(body_type).chains
            if parent is not None and all(ends[p][1] in names for p in parts)}


def connector(body_type, parts):
    """A chain's connector: its first part when that starts at a joint of the body's line (a biped's shoulder starts at
    the neck), else None (a leg: the thigh starts at its own hip joint)."""
    ends = part_ends(body_type)
    joints = {j for p in of(body_type).chains[0][1] for j in ends[p]}
    return parts[0] if ends[parts[0]][0] in joints else None


def stands(body_type, parts):
    """Whether a chain stands on the ground: its last part is fitted along the ground (its FIT rule's to_tip is
    'ground': a foot)."""
    return of(body_type).rule(parts[-1]).to_tip == 'ground'


def part_groups(body_type):
    """Each round-4 question's parts, including the body's middle."""
    ends = part_ends(body_type)
    return {group: [p for p, (_, end) in ends.items() if end in names]
            for group, names, _, _ in groups_of(body_type)}


def groups_of_parts(parts, body_type):
    """The groups of part_groups that ``parts`` lie in, in the groups' order."""
    parts = set(parts)
    return [g for g, ps in part_groups(body_type).items() if parts.intersection(ps)]


def limbs(body_type):
    """The parts that are on a limb: every chain but the body's own line."""
    return [p for parent, parts in of(body_type).chains if parent is not None for p in parts]


def limb_tips(body_type):
    """The joints at the tips of the limbs: the end joint of each limb's last part (part_ends), a biped's toes and hand
    ends. They are on the mesh's surface by nature, not inside it."""
    ends = part_ends(body_type)
    return [ends[parts[-1]][1] for parent, parts in of(body_type).chains if parent is not None and parts[-1] in ends]


def world_part(body_type, parts):
    """The part of a segment that is the last of its chain (``parts``, fit.segments) whose rule keeps its orientation in
    the world, or None: the head anywhere in it (a part folded in with the head rides upright with it), else its first
    part when that part's rule is a world one (a foot)."""
    t = of(body_type)
    if t.head in parts:
        return t.head
    return parts[0] if t.rule(parts[0]).world else None


def side_of(name):
    return 'left' if name.startswith('left_') else 'right' if name.startswith('right_') else 'middle'


def mirror_of(name):
    """The same joint on the other side ("left_ankle" -> "right_ankle"); None for a joint in the middle."""
    for a, b in (('left_', 'right_'), ('right_', 'left_')):
        if name.startswith(a):
            return b + name[len(a):]
    return None


def child_of(body_type, name):
    """The next joint along the line ``name`` is on: the end of the first part, in the body type's order, that starts
    at it (part_ends); None at the tip of a limb or of the head."""
    return next((b for a, b in part_ends(body_type).values() if a == name and b != name), None)
