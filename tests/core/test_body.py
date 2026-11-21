"""Synthetic bodies exercise the ordered facts, independently of downloaded rigs."""
import numpy as np
import pytest

from rigswap.core import body
from rigswap.core.rig import Rig


def _figure(bones):
    """Each bone carries count vertices, independently of mesh topology."""
    names = [b[0] for b in bones]
    parents = np.array([names.index(p) if p else -1 for _, p, _, _ in bones])
    heads = np.array([h for _, _, h, _ in bones], float)
    owners = np.repeat(np.arange(len(bones)), [b[3] for b in bones])
    return Rig('synthetic', names, parents, heads, heads + [0, 0, .05], ['body'],
               heads[owners].copy(), np.eye(len(bones))[owners])


def _bones():
    bones = [('hips', None, [0, 0, 1], 10), ('spine', 'hips', [0, 0, 1.25], 10),
             ('chest', 'spine', [0, 0, 1.5], 12), ('neck', 'chest', [0, 0, 1.8], 4),
             ('head', 'neck', [0, 0, 2.1], 40)]
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'thigh.{side}', 'hips', [sign * .3, 0, .9], 8),
                  (f'shin.{side}', f'thigh.{side}', [sign * .3, 0, .5], 8),
                  (f'foot.{side}', f'shin.{side}', [sign * .3, 0, .08], 6),
                  (f'toe.{side}', f'foot.{side}', [sign * .3, -.12, 0], 4),
                  (f'arm.{side}', 'chest', [sign * .4, 0, 1.5], 8),
                  (f'fore.{side}', f'arm.{side}', [sign * .7, 0, 1.5], 8),
                  (f'hand.{side}', f'fore.{side}', [sign, 0, 1.5], 8)]
    return bones


# What the labels ask (labels.py) would answer for these synthetic bodies, read off their pair roots' names: in a test
# the names stand in for the AI, and what is tested is the reading given that answer.
ROLE_NAMES = {'thigh': 'legs', 'hind': 'legs', 'foot': 'legs', 'leg': 'legs', 'fore': 'front legs', 'arm': 'arms'}


def _roles(rig):
    return {n: ROLE_NAMES[n.split('.')[0]] for n in rig.names if n.split('.')[0] in ROLE_NAMES}


def _read(rig, up=(0, 0, 1), front=(0, -1, 0)):
    return body.read(rig, up, front, _roles(rig))


def _names(rig, bones):
    return [rig.names[b] for b in bones]


def _stages(rig):
    mass = body._Mass(rig)
    axis = body._mirror(mass, [0, 0, 1])
    twins = body._twins(mass, axis)
    trunk = body._trunk(mass, twins)
    chains = body._limbs(mass, trunk)
    return mass, axis, trunk, chains, body._pairs(mass, twins, chains)


def _pair_names(rig):
    return {frozenset(_names(rig, pair)) for pair in _stages(rig)[-1]}


def test_two_limbs_and_a_trunk_name_a_biped():
    rig = _figure(_bones())
    result = _read(rig)
    assert result.plan == 'biped'
    assert _names(rig, result.spine) == ['hips', 'spine', 'chest', 'neck', 'head']
    assert _names(rig, result.legs['left']) == ['hips', 'thigh.L', 'shin.L', 'foot.L', 'toe.L']
    assert _names(rig, result.arms['right']) == ['arm.R', 'fore.R', 'hand.R']
    assert result.chest in result.spine


def test_a_limb_in_pieces_hung_side_by_side_off_the_trunk_is_joined_down_it():
    # a flat rip (Boomerang Bro): its thigh, shin and foot each hang off the hips, a pair each, all named legs
    bones = [b for b in _bones() if b[0].split('.')[0] not in ('thigh', 'shin', 'foot', 'toe')]
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'foot.{side}', 'hips', [sign * .3, 0, .08], 6), (f'thigh.{side}', 'hips', [sign * .3, 0, .9], 8),
                  (f'leg.{side}', 'hips', [sign * .3, 0, .5], 8)]
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'biped'
    assert _names(rig, result.legs['left']) == ['hips', 'thigh.L', 'leg.L', 'foot.L']
    assert _names(rig, result.legs['right']) == ['hips', 'thigh.R', 'leg.R', 'foot.R']


def test_step_1_uses_the_heaviest_tree_and_ignores_empty_controls():
    rig = _figure(_bones() + [('prop', None, [20, 10, -20], 1),
                             ('control', 'chest', [30, 10, -10], 0)])
    result = _read(rig)
    assert rig.names[result.root] == 'hips'
    assert rig.names[result.skull] == 'head'
    assert result.floor == 0
    assert 'control' not in _names(rig, body._Mass(rig).bones)


def test_step_1_coincident_joints_use_the_highest_bone_and_combine_skin():
    bones = _bones()
    bones[0] = ('hips', 'waist', [0, 0, 1], 10)
    bones[1] = ('spine', 'waist', [0, 0, 1.25], 10)
    rig = _figure(bones + [('waist', None, [0, 0, 1], 3)])
    mass = body._Mass(rig)
    hips, waist = (rig.names.index(n) for n in ('hips', 'waist'))
    assert hips not in mass.bones
    assert mass.own[waist] == 13
    result = _read(rig)
    assert result.root == result.spine[0] == waist
    assert result.legs['left'][:2] == [waist, rig.names.index('thigh.L')]
    assert rig.parents[hips] == waist  # the input rig is untouched


def test_joints_within_file_precision_combine_skin_and_leave_no_extra_spine_bone():
    bones = [(n, p, [0, 0, 1.00005] if n == 'spine' else h, w) for n, p, h, w in _bones()]
    rig = _figure(bones)
    mass = body._Mass(rig)
    distance = np.linalg.norm(rig.heads[1] - rig.heads[0])
    assert np.finfo(np.float32).eps * mass.size < distance < body.EXACT * mass.size
    assert mass.groups[0] == [0, 1]
    assert mass.own[0] == 20
    assert np.array_equal(mass.weights([0]), rig.weights[:, :2].sum(axis=1))
    assert _names(rig, _read(rig).spine) == ['hips', 'chest', 'neck', 'head']
    assert rig.parents[1] == 0


def test_step_2_mirror_precedes_up_and_survives_axis_permutation():
    rig = _figure(_bones())
    assert np.array_equal(body.symmetry_axis(rig), [1, 0, 0])
    for attr in ('heads', 'tails', 'points'):
        setattr(rig, attr, getattr(rig, attr)[:, [2, 0, 1]])
    assert np.array_equal(body.symmetry_axis(rig), [0, 1, 0])
    assert np.array_equal(body.mirror_axis(rig, [1, 0, 0]), [0, 1, 0])
    result = _read(rig, [1, 0, 0], [0, 0, -1])
    assert result.plan == 'biped'
    assert rig.names[result.skull] == 'head'


def test_step_2_exact_near_plane_twins_vote_without_a_distance_band():
    rig = _figure(_bones() + [('near.a', 'chest', [0, .00001, 1.6], 1000),
                             ('near.b', 'chest', [0, -.00001, 1.6], 1000),
                             ('end.a', 'near.a', [0, .00001, 1.9], 1),
                             ('end.b', 'near.b', [0, -.00001, 1.9], 1)])
    assert np.array_equal(body.symmetry_axis(rig), [0, 1, 0])
    assert np.array_equal(body.mirror_axis(rig, [0, 0, 1]), [0, 1, 0])


def test_copies_on_one_side_of_the_middle_cast_no_mirror_vote():
    # Bart: his upper and lower eyelids hang from one joint, their meshes a step apart up and down, both above his
    # middle; heavier than his limbs, they outvoted them.
    rig = _figure(_bones() + [('lid.up', 'head', [0, -.15, 2.1], 1500), ('lid.down', 'head', [0, -.15, 2.1], 1500)])
    up, down = rig.names.index('lid.up'), rig.names.index('lid.down')
    owner = np.argmax(rig.weights, axis=1)
    rig.points[owner == up, 2] += .05
    rig.points[owner == down, 2] -= .05
    mass = body._Mass(rig)
    assert (up, down) in body._copies(mass)
    assert np.array_equal(body.symmetry_axis(rig), [1, 0, 0])


def test_the_middle_is_a_joint_carrying_skin_past_an_empty_branch_beside_the_body():
    # Meta Knight: his hip carries no skin and lies below his feet, to one side of every twin pair.
    rig = _figure([('top', None, [0, 0, 0], 0), ('hip', 'top', [-.3, .2, -.2], 0), ('body', 'hip', [0, 0, .5], 30),
                   ('wing.L', 'body', [.1, 0, .6], 5), ('tip.L', 'wing.L', [.25, 0, .8], 5),
                   ('wing.R', 'body', [-.1, 0, .6], 5), ('tip.R', 'wing.R', [-.25, 0, .8], 5),
                   ('foot.L', 'hip', [.08, 0, .05], 6), ('toe.L', 'foot.L', [.08, -.1, 0], 3),
                   ('foot.R', 'hip', [-.08, 0, .05], 6), ('toe.R', 'foot.R', [-.08, -.1, 0], 3)])
    mass = body._Mass(rig)
    assert np.array_equal(mass.middle, [0, 0, .5])
    assert body._copy_steps(mass) @ [1, 0, 0] > 0
    pairs = {tuple(_names(rig, p)) for p in body._copies(mass, np.array([1., 0, 0]))}
    assert pairs == {('wing.L', 'wing.R'), ('foot.L', 'foot.R')}


@pytest.mark.parametrize('carried', [0, 2])
def test_step_3_branching_midline_keeps_a_shared_arm_junction(carried):
    bones = [(n, 'junction' if n in ('arm.L', 'arm.R') else p, h, w) for n, p, h, w in _bones()]
    rig = _figure(bones + [('junction', 'chest', [0, 0, 1.6], carried),
                          ('jaw', 'head', [0, -.2, 2.0], 5), ('crest', 'head', [0, .1, 2.3], 2)])
    _, _, trunk, _, pairs = _stages(rig)
    assert {'hips', 'chest', 'head', 'junction', 'jaw', 'crest'} <= set(_names(rig, trunk))
    assert frozenset({'arm.L', 'arm.R'}) in {frozenset(_names(rig, p)) for p in pairs}
    result = _read(rig)
    assert rig.names[result.skull] == 'head'
    assert _names(rig, result.arms['left']) == ['arm.L', 'fore.L', 'hand.L']
    assert rig.names[result.chest] == 'chest'
    assert 'junction' not in _names(rig, result.spine)


def test_step_3_unpaired_mass_and_its_connecting_joints_form_the_trunk():
    rig = _figure(_bones() + [('branch', 'chest', [.4, .2, 1.6], 4),
                             ('empty', 'chest', [0, .2, 1.65], 0),
                             ('tip', 'empty', [0, .2, 1.7], 2)])
    branch = rig.names.index('branch')
    rig.points[rig.weights[:, branch] > 0, 0] = 0
    assert {'branch', 'empty', 'tip'} <= set(_names(rig, _stages(rig)[2]))
    rig.points[rig.weights[:, branch] > 0, 0] = .5
    assert 'branch' in _names(rig, _stages(rig)[2])


def test_step_3_never_connects_unpaired_pieces_through_a_twin_bone():
    rig = _figure(_bones() + [('return', 'arm.L', [0, .3, 1.7], 4)])
    trunk = _names(rig, _stages(rig)[2])
    assert 'arm.L' not in trunk
    assert 'return' not in trunk


def test_step_3_spine_runs_from_the_joint_the_legs_part_from_to_the_head_whichever_way_the_parents_point():
    # rooted above the hips, the hips hanging from it (a segmented rip rooted at its chest: Eevee's)
    bones = _bones()
    bones[0] = ('hips', 'waist', [0, 0, 1], 10)
    bones[1] = ('spine', 'waist', [0, 0, 1.25], 10)
    rig = _figure(bones + [('waist', None, [0, 0, 1.1], 3)])
    result = _read(rig)
    assert _names(rig, result.spine) == ['hips', 'waist', 'spine', 'chest', 'neck', 'head']
    assert rig.names[result.root] == 'waist'
    assert _names(rig, result.legs['left'])[:2] == ['hips', 'thigh.L']


def test_step_4_follows_mass_at_each_fork_before_comparing_lengths():
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        for i in range(5):
            bones.append((f'strap{i}.{side}', f'strap{i - 1}.{side}' if i else f'arm.{side}',
                          [sign * .4, .1 * (i + 1), 1.4 - .1 * i], 1))
    rig = _figure(bones)
    result = _read(rig)
    assert _names(rig, result.arms['left']) == ['arm.L', 'fore.L', 'hand.L']


def test_step_5_unmatched_branches_do_not_decide_pairing():
    rig = _figure(_bones() + [('strap', 'arm.L', [.5, .3, 1.3], 1),
                             ('strap.end', 'strap', [.5, .3, 1.1], 1)])
    assert _pair_names(rig) == {frozenset({'arm.L', 'arm.R'}), frozenset({'thigh.L', 'thigh.R'})}


def test_step_5_shared_edges_keep_unequal_tips_and_survive_internal_disagreement():
    bones = _bones() + [('extra', 'hand.L', [1.2, 0, 1.5], 1)]
    rig = _figure(bones)
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(rig)
    result = _read(rig)
    assert rig.names[result.arms['left'][-1]] == 'extra'
    changed = [(n, p, [-.7, 0, 2.0] if n == 'fore.R' else h, w) for n, p, h, w in bones]
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(_figure(changed))  # their root edge still agrees


def test_step_5_posed_copy_roots_need_not_leave_at_mirrored_places():
    bones = [(n, p, [h[0], h[1] + .9, h[2]] if n in ('arm.R', 'fore.R', 'hand.R') else h, w)
             for n, p, h, w in _bones()]
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(_figure(bones))


def test_step_5_midline_branches_are_never_twins():
    rig = _figure(_bones() + [('mid.a', 'chest', [0, .2, 1.6], 2),
                             ('mid.b', 'chest', [0, -.2, 1.6], 2)])
    assert all(not any(n.startswith('mid.') for n in _names(rig, p)) for p in _stages(rig)[-1])


def test_step_6_the_pair_labelled_legs_wins_over_heavier_floor_panels():
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'panel.{side}', 'hips', [sign * .4, -.2, .9], 40),
                  (f'panel.end.{side}', f'panel.{side}', [sign * .5, -.2, 0], 40)]
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'biped'
    assert rig.names[result.legs['left'][-1]] == 'toe.L'


def test_step_6_floor_contact_uses_chain_mesh_and_excludes_attachments():
    rig = _figure(_bones() + [('hanging', 'arm.L', [.4, .4, 0], 1)])
    mass, _, _, chains, _ = _stages(rig)
    _, _, contact = body._support(mass, chains, np.array([0, 0, 1]))
    assert contact[rig.names.index('arm.L')] == 0
    assert contact[rig.names.index('thigh.L')] > 0
    assert _read(rig).plan == 'biped'


def test_step_6_one_posed_leg_can_support_its_twin_pair():
    rig = _figure(_bones())
    start = rig.names.index('thigh.R')
    bones = [start] + body._below_all(rig.children(), start)
    pivot = rig.heads[start].copy()
    angle = np.pi / 3
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    selected = rig.weights[:, bones].sum(axis=1) > 0
    rig.points[selected] = (rig.points[selected] - pivot) @ rotation.T + pivot
    rig.heads[bones] = (rig.heads[bones] - pivot) @ rotation.T + pivot
    result = _read(rig)
    assert result.plan == 'biped'
    assert rig.names[result.legs['right'][-1]] == 'toe.R'


@pytest.mark.parametrize('permuted', [False, True])
def test_step_6_the_pair_labelled_arms_wins_over_heavier_farther_reaching_hip_swords(permuted):
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'sword.{side}', 'hips', [sign * .6, .2, .9], 80),
                  (f'sword.end.{side}', f'sword.{side}', [sign * 1.4, .2, .9], 80)]
    rig = _figure(bones)
    up, front = np.array([0, 0, 1]), np.array([0, -1, 0])
    if permuted:
        for attr in ('heads', 'tails', 'points'):
            setattr(rig, attr, getattr(rig, attr)[:, [2, 0, 1]] + [4, -3, 2])
        up, front = up[[2, 0, 1]], front[[2, 0, 1]]
    result = _read(rig, up, front)
    assert rig.names[result.arms['left'][0]] == 'arm.L'
    assert rig.names[result.legs['left'][-1]] == 'toe.L'


def test_step_6_wings_not_labelled_arms_stay_attachments_however_heavy_or_far_reaching():
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'wing.{side}', 'chest', [sign * .4, -.3, 1.5], 80),
                  (f'wing.end.{side}', f'wing.{side}', [sign * 1.5, -.3, 1.5], 80)]
    rig = _figure(bones)
    assert rig.names[_read(rig).arms['left'][0]] == 'arm.L'


def test_step_7_head_is_above_the_arm_departure_even_when_the_lower_torso_is_heavier():
    rig = _figure([(n, p, h, 100 if n == 'spine' else w) for n, p, h, w in _bones()]
                  + [('hair', 'head', [0, .1, 2.4], 2)])
    assert rig.names[_read(rig).skull] == 'head'


def test_step_8_a_pair_labelled_front_legs_names_a_quadruped():
    bones = [('hips', None, [0, .5, 1], 10), ('spine', 'hips', [0, 0, 1], 10),
             ('chest', 'spine', [0, -.5, 1], 12), ('head', 'chest', [0, -.7, 1.5], 40)]
    for end, y, parent in [('hind', .5, 'hips'), ('fore', -.5, 'chest')]:
        for side, sign in [('L', 1), ('R', -1)]:
            bones += [(f'{end}.{side}', parent, [sign * .3, y, .9], 8),
                      (f'{end}.knee.{side}', f'{end}.{side}', [sign * .3, y, .5], 8),
                      (f'{end}.foot.{side}', f'{end}.knee.{side}', [sign * .3, y, 0], 8)]
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'quadruped'
    assert rig.names[result.legs['left'][-1]] == 'hind.foot.L'
    assert rig.names[result.arms['left'][0]] == 'fore.L'
    assert _names(rig, result.spine) == ['hips', 'spine', 'chest', 'head']


def test_step_8_feet_without_an_intervening_joint_name_a_ball():
    rig = _figure([('root', None, [0, 0, .5], 0), ('body', 'root', [0, 0, 1], 50),
                   ('junction', 'body', [0, 0, .75], 0), ('crest', 'body', [0, .1, 1.4], 2),
                   ('foot.L', 'root', [.3, 0, 0], 8), ('foot.R', 'root', [-.3, 0, 0], 8),
                   ('arm.L', 'junction', [.4, 0, .75], 8), ('arm.R', 'junction', [-.4, 0, .75], 8)])
    result = _read(rig)
    assert result.plan == 'ball'
    assert rig.names[result.skull] == 'body'
    assert _names(rig, result.spine) == ['root', 'body']
    assert result.root == result.spine[0]
    assert result.chest in result.spine
    assert _names(rig, result.legs['left']) == ['foot.L']
    assert _names(rig, result.arms['left']) == ['arm.L']


def test_step_8_feet_under_thighs_that_only_blend_the_body_name_a_ball():
    # the thighs start inside the body and carry none of its skin over half: the leg one sees is the foot
    rig = _figure([('root', None, [0, 0, .5], 0), ('body', 'root', [0, 0, 1], 50),
                   ('junction', 'body', [0, 0, .75], 0), ('crest', 'body', [0, .1, 1.4], 2),
                   ('thigh.L', 'root', [.3, 0, .6], 0), ('thigh.R', 'root', [-.3, 0, .6], 0),
                   ('foot.L', 'thigh.L', [.3, 0, 0], 8), ('foot.R', 'thigh.R', [-.3, 0, 0], 8),
                   ('arm.L', 'junction', [.4, 0, .75], 8), ('arm.R', 'junction', [-.4, 0, .75], 8)])
    blend = np.zeros((6, len(rig.names)))
    for k, side in enumerate(('L', 'R')):
        blend[3 * k:3 * k + 3, rig.names.index(f'thigh.{side}')] = .3
        blend[3 * k:3 * k + 3, rig.names.index('body')] = .7
    points = np.array([[s * .3, 0, z] for s in (1, -1) for z in (.5, .6, .7)], float)
    rig = Rig('synthetic', rig.names, rig.parents, rig.heads, rig.tails, rig.meshes,
              np.vstack([rig.points, points]), np.vstack([rig.weights, blend]))
    result = _read(rig)
    assert result.plan == 'ball'
    assert _names(rig, result.legs['left']) == ['foot.L']
    # thighs carrying their own skin are a biped's
    assert _read(_figure(_bones())).plan == 'biped'


def test_a_smooth_skins_tail_at_the_soles_is_no_floor_the_thigh_carries():
    # Richard: his one-piece skin reaches his soles from the thigh at 0.0055; the toes carry them
    rig = _figure(_bones())
    for side in ('L', 'R'):
        toe = rig.weights[:, rig.names.index(f'toe.{side}')] == 1
        rig.weights[toe, rig.names.index(f'toe.{side}')] = .99
        rig.weights[toe, rig.names.index(f'thigh.{side}')] = .01
    assert _read(rig).plan == 'biped'


def test_step_8_no_pair_labelled_legs_names_a_legless_body():
    bones = [b for b in _bones() if not b[0].startswith(('thigh', 'shin', 'foot', 'toe'))]
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'legless'
    assert result.legs == {}
    assert rig.names[result.skull] == 'head'
    assert rig.names[result.arms['left'][0]] == 'arm.L'


def test_without_roles_no_pair_is_a_limb():
    rig = _figure(_bones())
    result = body.read(rig, [0, 0, 1], [0, -1, 0])
    assert result.plan == 'legless'
    assert result.legs == {}
    assert result.arms == {'left': [], 'right': []}


def test_a_role_on_either_twin_names_the_pair():
    rig = _figure(_bones())
    result = body.read(rig, [0, 0, 1], [0, -1, 0], {'thigh.R': 'legs', 'arm.L': 'arms'})
    assert rig.names[result.legs['left'][-1]] == 'toe.L'
    assert rig.names[result.arms['right'][0]] == 'arm.R'


def test_twins_before_trunk_preserves_narrow_thighs_and_shins():
    bones = []
    for name, parent, head, weight in _bones():
        head = list(head)
        if name.startswith(('thigh.', 'shin.')):
            head[0] = .08 if name.endswith('L') else -.08
        bones.append((name, parent, head, weight))
    rig = _figure(bones)
    mass, _, trunk, _, _ = _stages(rig)
    assert .08 < .05 * mass.size        # the thighs leave nearer the midline than a twentieth of the body
    assert not {'thigh.L', 'thigh.R', 'shin.L', 'shin.R'} & set(_names(rig, trunk))
    result = _read(rig)
    assert result.plan == 'biped'
    assert _names(rig, result.legs['left']) == ['hips', 'thigh.L', 'shin.L', 'foot.L', 'toe.L']


def test_descendant_twins_follow_structure_when_a_pose_crosses_the_plane():
    rig = _figure(_bones())
    start = rig.names.index('arm.R')
    descendants = body._below_all(rig.children(), start)
    # Turn this arm back across the body without moving its mirrored root.
    rig.heads[descendants, 0] = 2 * rig.heads[start, 0] - rig.heads[descendants, 0]
    mass = body._Mass(rig)
    twins = body._twins(mass, np.array([1., 0., 0.]))
    for part in ('arm', 'fore', 'hand'):
        assert twins[rig.names.index(part + '.L')] == rig.names.index(part + '.R')
    assert rig.heads[rig.names.index('hand.R'), 0] > 0
    assert 'hand.R' not in _names(rig, body._trunk(mass, twins))


def test_root_twins_can_leave_nearby_joints_in_the_same_trunk_region():
    bones = [(n, 'socket.' + n[-1] if n.startswith('arm.') else p, h, w)
             for n, p, h, w in _bones()]
    rig = _figure(bones + [('socket.L', 'chest', [0, 0, 1.51], 1),
                          ('socket.R', 'chest', [0, 0, 1.49], 1)])
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(rig)


@pytest.mark.parametrize('extra_mass', [1, 100])
def test_equal_copies_pair_nearest_first_independently_of_mass(extra_mass):
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [1, 0, 0], 10), ('b', 'root', [1, .01, 0], 9),
                   ('c', 'root', [-1, 0, 0], 10), ('d', 'root', [-1, -.01, 0], extra_mass)])
    mass = body._Mass(rig)
    twins = body._twins(mass, np.array([1., 0., 0.]))
    assert twins == {1: 3, 3: 1, 2: 4, 4: 2}


def test_twins_are_distinct_and_one_shared_internal_edge_still_counts():
    rig = _figure(_bones())
    mass = body._Mass(rig)
    assert all(a != b for a, b in body._twins(mass, np.array([1., 0., 0.])).items())
    a, b = (rig.names.index('arm.' + s) for s in ('L', 'R'))
    rig.heads[b, 0] = -.6
    mass = body._Mass(rig)
    assert body._shared(mass)[0](a, b)[0] == 1
    assert body._twins(mass, np.array([1., 0., 0.]))[a] == b


def test_posed_roots_choose_their_copy_over_a_nearer_trunk_joint():
    bones = [(n, p, [h[0], h[1] + .7, h[2] + .2] if n in
              ('arm.R', 'fore.R', 'hand.R') else h, w) for n, p, h, w in _bones()]
    bones += [('decoy', 'chest', [-.4, 0, 1.5], 1),
              ('decoy.end', 'decoy', [-.4, 0, 1.6], 1)]
    rig = _figure(bones)
    mass = body._Mass(rig)
    a, b, decoy = (rig.names.index(n) for n in ('arm.L', 'arm.R', 'decoy'))
    reflected = rig.heads[a] * [-1, 1, 1]
    assert np.linalg.norm(reflected - rig.heads[decoy]) < np.linalg.norm(reflected - rig.heads[b])
    assert body._twins(mass, np.array([1., 0, 0]))[a] == b
    assert rig.names[_read(rig).arms['right'][0]] == 'arm.R'


def test_twin_feet_ignore_the_edge_from_an_off_centre_control():
    rig = _figure([('root', None, [0, 0, .5], 0),
                   ('control', 'root', [.3, 0, .6], 0),
                   ('foot.L', 'control', [.4, 0, 0], 8),
                   ('toe.L', 'foot.L', [.4, -.2, 0], 4),
                   ('foot.R', 'control', [-.4, 0, 0], 8),
                   ('toe.R', 'foot.R', [-.4, -.2, 0], 4),
                   ('body', 'root', [0, 0, 1], 40)])
    mass = body._Mass(rig)
    assert abs(np.linalg.norm(rig.heads[2] - rig.heads[1])
               - np.linalg.norm(rig.heads[4] - rig.heads[1])) > .05 * mass.size
    assert body._copies(mass) == [(2, 4)]
    assert body._twins(mass, np.array([1., 0, 0]))[2] == 4
    assert _names(rig, _read(rig).legs['left']) == ['foot.L', 'toe.L']


def test_more_shared_edges_beat_a_nearer_copy_for_pairing_and_axis():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [1, 0, 1], 10), ('a.end', 'a', [1, 0, 1.4], 10),
                   ('exact', 'root', [-np.sqrt(.99), .1, 1], 10),
                   ('exact.end', 'exact', [-np.sqrt(.99), .1, 1.4], 10),
                   ('loose', 'root', [-1, 0, 1], 10), ('loose.end', 'loose', [-1, 0, 1.41], 10)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](1, 3)[0] == 2
    assert body._shared(mass)[0](1, 5)[0] == 1
    assert body._copies(mass) == [(1, 3)]
    assert body._twins(mass, np.array([1., 0, 0]))[1] == 3


@pytest.mark.parametrize('tiny_skin', [0., 1e-9])
def test_empty_junction_above_ball_arms_cannot_block_the_body_head(tiny_skin):
    rig = _figure([('root', None, [0, 0, .5], 0), ('body', 'root', [0, 0, 1], 50),
                   ('foot.L', 'root', [.3, 0, 0], 8), ('foot.R', 'root', [-.3, 0, 0], 8),
                   ('arm.L', 'body', [.4, 0, 1], 8), ('arm.R', 'body', [-.4, 0, 1], 8),
                   ('knot', 'body', [0, 0, 1.5], 0), ('tail', 'knot', [0, .2, .8], 2)])
    rig.weights[0, rig.names.index('knot')] = tiny_skin
    result = _read(rig)
    assert rig.names[result.skull] == 'body'
    assert all(result.arms.values())


def test_ball_head_is_below_empty_shoulder_controls():
    rig = _figure([('root', None, [0, 0, .5], 0), ('body', 'root', [0, 0, 1], 50),
                   ('foot.L', 'root', [.3, 0, 0], 8), ('foot.R', 'root', [-.3, 0, 0], 8),
                   ('socket.L', 'body', [0, 0, 1.3], 0),
                   ('socket.R', 'body', [0, 0, 1.31], 0),
                   ('arm.L', 'socket.L', [.4, 0, 1.3], 8),
                   ('arm.R', 'socket.R', [-.4, 0, 1.3], 8)])
    result = _read(rig)
    assert rig.names[result.skull] == 'body'
    assert all(result.arms.values())


def test_forked_copies_keep_their_independently_heaviest_tips():
    bones = _bones()
    for side, sign in [('L', 1), ('R', -1)]:
        bones += [(f'finger.a.{side}', f'hand.{side}', [sign * 1.2, 0, 1.5], 4 if side == 'L' else 3),
                  (f'finger.b.{side}', f'hand.{side}', [sign * 1.3, 0, 1.5], 3 if side == 'L' else 4)]
    rig = _figure(bones)
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(rig)
    result = _read(rig)
    assert rig.names[result.arms['left'][-1]] == 'finger.a.L'
    assert rig.names[result.arms['right'][-1]] == 'finger.b.R'


def test_an_empty_child_does_not_end_the_chain_comparison():
    bones = _bones() + [('control', 'arm.L', [.4, .3, 1.5], 0)]
    bones = [(n, p, [-.7, 0, 2.0] if n == 'fore.R' else h, w) for n, p, h, w in bones]
    rig = _figure(bones)
    assert body._shared(body._Mass(rig))[0](rig.names.index('arm.L'), rig.names.index('arm.R'))[0] == 1


@pytest.mark.parametrize('middle', [False, True])
def test_without_arms_chest_is_below_head_unless_that_is_root(middle):
    bones = [('root', None, [0, 0, 0], 8)]
    if middle:
        bones.append(('torso', 'root', [0, 0, .5], 10))
    bones.append(('head', 'torso' if middle else 'root', [0, 0, 1], 40))
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'legless'
    assert result.chest in result.spine
    assert rig.names[result.chest] == ('torso' if middle else 'head')
    assert np.linalg.norm(rig.heads[result.chest] - rig.heads[result.root]) > 0


@pytest.mark.parametrize('scale', [.001, 1e6])
def test_the_reading_is_invariant_under_uniform_scale(scale):
    rig = _figure(_bones())
    expected = _read(rig)
    for attr in ('heads', 'tails', 'points'):
        setattr(rig, attr, getattr(rig, attr) * scale)
    result = _read(rig)
    for attr in ('root', 'spine', 'skull', 'legs', 'arms', 'chest', 'plan'):
        assert getattr(result, attr) == getattr(expected, attr)


def test_sub_precision_helpers_merge_before_shared_edges_are_counted():
    bones = _bones()
    for side, sign, length in [('L', 1, .0001), ('R', -1, .00011)]:
        bones += [(f'helper.{side}', f'hand.{side}', [sign * (1 + length), 0, 1.5], 1),
                  (f'finger.{side}', f'helper.{side}', [sign * 1.2, 0, 1.5], 2)]
    rig = _figure(bones)
    mass = body._Mass(rig)
    assert all(rig.names.index('helper.' + side) not in mass.bones for side in ('L', 'R'))
    assert body._shared(mass)[0](*(rig.names.index('hand.' + side) for side in ('L', 'R')))[0] == 2
    result = _read(rig)
    assert rig.names[result.arms['left'][-1]] == 'finger.L'
    assert rig.names[result.arms['right'][-1]] == 'finger.R'


def test_chest_is_above_root_even_when_arms_leave_root():
    bones = [(n, 'hips' if n.startswith('arm.') else p, h, w) for n, p, h, w in _bones()]
    rig = _figure(bones)
    result = _read(rig)
    assert rig.names[result.chest] == 'chest'
    assert result.chest in result.spine[1:]


def test_the_chest_is_the_spine_joint_the_arms_hang_from_not_the_neck_nearest_their_height():
    from rigswap.core import correspond
    # Kritter's arms hang from his waist and leave him well above it, nearer his head joint than his waist: the
    # shoulders stand where the arms leave, not at the head.
    bones = [(n, 'spine' if n.startswith('arm.') else p, [h[0], h[1], 1.7] if n.split('.')[0] in ('arm', 'fore', 'hand')
              else h, w) for n, p, h, w in _bones()]
    rig = _figure(bones)
    result = _read(rig)
    assert rig.names[result.chest] == 'spine'
    assert correspond.levels(correspond.model_chains(rig, result)[0])[1] == pytest.approx(1.7)


def test_a_ball_keeps_arms_leaving_its_head():
    rig = _figure([('root', None, [0, 0, .5], 0), ('body', 'root', [0, 0, 1], 50),
                   ('foot.L', 'root', [.3, 0, 0], 8), ('foot.R', 'root', [-.3, 0, 0], 8),
                   ('arm.L', 'body', [.4, 0, 1], 8), ('arm.R', 'body', [-.4, 0, 1], 8)])
    result = _read(rig)
    assert result.plan == 'ball'
    assert rig.names[result.skull] == 'body'
    assert result.chest == result.root  # no spine bone between root and head
    assert _names(rig, result.arms['left']) == ['arm.L']
    assert _names(rig, result.arms['right']) == ['arm.R']


@pytest.mark.parametrize('scale', [.001, 1., 1e6])
def test_twin_places_are_nearer_each_other_than_their_own_mirrors(scale):
    heads = np.array([[.05, 0, 1.5], [-.05, 0, 1.5], [-.000001, 0, 1.56],
                      [0, 0, 1.5], [.05, 0, 1.6]]) * scale
    distance = body._mirror_distances(heads, np.zeros(3), np.array([1., 0., 0.]))
    assert distance[0, 1] == distance[1, 0] == 0
    assert np.isinf(distance[0, 2]) and np.isinf(distance[2, 0])
    assert np.isinf(distance[3]).all()  # the midline is its own mirror
    assert np.isinf(np.diag(distance)).all()
    assert np.isinf(distance[0, 4])  # same-side heads cannot pair


def test_twin_place_inequality_is_strict_for_both_bones():
    heads = np.array([[1., 0, 0], [-1., 2, 0], [-.25, 0, 0]])
    distance = body._mirror_distances(heads, np.zeros(3), np.array([1., 0., 0.]))
    assert np.isinf(distance[0, 1])  # mismatch equals the self-mirror distance
    assert np.isinf(distance[0, 2])  # close enough for one bone, not for its twin


def test_a_heavy_near_midline_neck_cannot_replace_a_narrow_shoulder_twin():
    bones = [(n, p, [.05 if n.endswith('L') else -.05, 0, 1.5] if n.startswith('arm.') else
              [-.000001, 0, 1.56] if n == 'neck' else h, w) for n, p, h, w in _bones()]
    rig = _figure(bones)
    mass = body._Mass(rig)
    twins = body._twins(mass, np.array([1., 0., 0.]))
    left, right, neck = (rig.names.index(n) for n in ('arm.L', 'arm.R', 'neck'))
    assert mass.total[neck] > mass.total[right]
    assert twins[left] == right
    assert neck not in twins
    assert rig.names[_read(rig).skull] == 'head'


@pytest.mark.parametrize('feet_only', [False, True])
def test_a_heavy_root_cannot_be_head_when_the_trunk_continues_above_it(feet_only):
    bones = [(n, 'hips' if n.startswith('arm.') else p,
              [h[0], h[1], 1.] if n.startswith('arm.') else h,
              100 if n == 'hips' else w) for n, p, h, w in _bones()]
    if feet_only:
        bones = [(n, 'hips' if n.startswith('foot.') else p,
                  [h[0], h[1], 0] if n.startswith('foot.') else h, w)
                 for n, p, h, w in bones if not n.startswith(('thigh.', 'shin.', 'toe.'))]
    rig = _figure(bones)
    result = _read(rig)
    assert rig.names[result.root] == 'hips'
    assert rig.names[result.skull] == 'head'
    assert result.plan == ('ball' if feet_only else 'biped')
    assert result.chest in result.spine[1:]
    assert rig.heads[result.chest, 2] > rig.heads[result.root, 2]
    assert all(result.arms.values())


def test_an_armless_heavy_root_still_has_its_head_above_it():
    rig = _figure([('root', None, [0, 0, 0], 100), ('head', 'root', [0, 0, 1], 10)])
    result = _read(rig)
    assert rig.names[result.skull] == rig.names[result.chest] == 'head'


def test_only_a_trunk_with_no_bone_above_root_can_have_root_as_head():
    rig = _figure([('root', None, [0, 0, 1], 100), ('below', 'root', [0, 0, 0], 10)])
    result = _read(rig)
    assert result.skull == result.root == result.chest
    rig.heads[1, 2] = 2
    rig.weights[:, 1] = 0
    rig.weights[:, 0] = 1
    # An empty control is not part of the trunk.
    assert _read(rig).skull == result.root


def test_an_unskinned_trunk_bone_above_root_cannot_be_head():
    rig = _figure([('root', None, [0, 0, 1], 100), ('joint', 'root', [0, 0, 2], 0),
                   ('tip', 'joint', [0, 0, 1], 10)])
    assert rig.names[_read(rig).skull] == 'root'


def test_terminal_copies_need_their_attachment_edge():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('socket', 'root', [-1, 0, .9], 1),
                   ('left', 'root', [1, 0, 1], 10),
                   ('nearest', 'socket', [-1, 0, 1], 10),
                   ('farther', 'root', [-np.sqrt(.99), .1, 1], 100)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](2, 3)[0] == 0
    assert body._shared(mass)[0](2, 4)[0] == 1
    assert body._twins(mass, np.array([1., 0., 0.]))[2] == 4


def test_an_empty_control_cannot_steal_the_nearest_mirror():
    bones = [(n, p, [h[0], h[1] + .01, h[2]] if n in ('arm.R', 'fore.R', 'hand.R') else h, w) for n, p, h, w in _bones()]
    rig = _figure(bones + [('empty', 'chest', [-.4, 0, 1.5], 0)])
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(rig)


def test_limbs_pair_either_side_of_the_body_not_of_a_root_left_beside_it():
    # Tauros: the rip's root is an empty at the world origin, and his posed body stands to one side of it.
    bones = [('origin', None, [0, 0, 0], 0)] + [(n, p or 'origin', [h[0] + 2, h[1], h[2]], w)
                                                for n, p, h, w in _bones()]
    pairs = _pair_names(_figure(bones))
    assert frozenset({'arm.L', 'arm.R'}) in pairs
    assert frozenset({'thigh.L', 'thigh.R'}) in pairs


@pytest.mark.parametrize('separate_head', [False, True])
def test_one_piece_trunk_is_both_hips_and_chest_with_arms(separate_head):
    bones = [('body', None, [0, 0, .5], 50),
             ('foot.L', 'body', [.3, 0, 0], 8), ('foot.R', 'body', [-.3, 0, 0], 8),
             ('arm.L', 'body', [.4, 0, .5], 8), ('arm.R', 'body', [-.4, 0, .5], 8)]
    if separate_head:
        bones.append(('head', 'body', [0, 0, 1], 40))
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'ball'
    assert result.root == result.chest == 0
    assert rig.names[result.skull] == ('head' if separate_head else 'body')
    assert all(result.arms.values()) and all(result.legs.values())


@pytest.mark.parametrize('permuted', [False, True])
def test_copied_legs_choose_lateral_over_positional_fore_hind_symmetry(permuted):
    bones = [('root', None, [0, 0, 1], 10)]
    for end, y, knee in [('hind', .5, .7), ('fore', -.5, .3)]:
        for side, sign in [('L', 1), ('R', -1)]:
            bones += [(f'{end}.{side}', 'root', [sign * .2, y, 1], 10),
                      (f'{end}.knee.{side}', f'{end}.{side}', [sign * .2, y, knee], 10),
                      (f'{end}.foot.{side}', f'{end}.knee.{side}', [sign * .2, y, 0], 10)]
    # Unrelated, heavy front/back joints favour Y by position but are not copied limbs.
    bones += [('front', 'root', [0, -.4, 1.2], 100),
              ('back', 'root', [0, .4, 1.21], 100)]
    rig = _figure(bones)
    expected = np.array([1., 0, 0])
    if permuted:
        for attr in ('heads', 'tails', 'points'):
            setattr(rig, attr, getattr(rig, attr)[:, [2, 0, 1]] + [4, -3, 2])
        expected = expected[[2, 0, 1]]
    mass = body._Mass(rig)
    assert {frozenset(_names(rig, p)) for p in body._copies(mass)} == {
        frozenset({'hind.L', 'hind.R'}), frozenset({'fore.L', 'fore.R'})}
    assert np.array_equal(body.symmetry_axis(rig), expected)


@pytest.mark.parametrize('noise', [0, .00005])
def test_coincident_copy_roots_step_between_their_carried_mesh_centres(noise):
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [0, 0, 1], 10), ('end.a', 'a', [1, 0, 1], 10),
                   ('b', 'root', [0, 0, 1 + noise], 10), ('end.b', 'b', [-1, 0, 1], 10)])
    mass = body._Mass(rig)
    assert np.linalg.norm(rig.heads[1] - rig.heads[3]) < body.EXACT * mass.size
    step = np.array([1, 0, -noise / 2])
    assert np.allclose(body._copy_steps(mass), 40 * np.abs(step) / np.linalg.norm(step))
    assert np.array_equal(body.symmetry_axis(rig), [1, 0, 0])


def test_unequal_off_mirror_roots_vote_at_the_outermost_equal_edges():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [0, 0, 1], 10), ('b', 'root', [.1, 0, -1], 10),
                   ('end.a', 'a', [.55, 0, 0], 10), ('end.b', 'b', [-.45, 0, 0], 10)])
    mass = body._Mass(rig)
    assert body._copies(mass) == [(1, 2)]
    assert body._shared(mass)[0](1, 2)[0] == 1
    assert np.allclose(body._copy_steps(mass), [40, 0, 0])
    assert np.array_equal(body.symmetry_axis(rig), [1, 0, 0])


def test_equal_root_edges_vote_before_posed_descendants_stepping_diagonally():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [.2, 0, 1], 10), ('b', 'root', [-.2, 0, 1], 10),
                   ('end.a', 'a', [.2, 2, 1], 100), ('end.b', 'b', [-.2, -2, 1], 100)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](1, 2)[0] == 2
    assert np.allclose(body._copy_steps(mass), [220, 0, 0])
    assert np.array_equal(body.symmetry_axis(rig), [1, 0, 0])


def test_mirrored_heads_with_unequal_parent_edges_are_twins_outside_the_trunk():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [0, 0, .1], 1), ('b', 'root', [.02, 0, -.1], 1),
                   ('end.a', 'a', [.4, 0, .3], 10), ('end.b', 'b', [-.4, 0, .30002], 10)])
    mass = body._Mass(rig)
    axis = np.array([1., 0, 0])
    assert body._copies(mass, axis) == []
    assert body._shared(mass, axis)[0](3, 4)[0] == 0
    twins = body._twins(mass, axis)
    assert twins == {3: 4, 4: 3}
    assert body._trunk(mass, twins) == {0, 1, 2}


def test_mirrored_heads_inside_the_plane_band_are_never_positional_twins():
    rig = _figure([('root', None, [0, 0, 0], 1), ('socket', 'root', [0, 0, .5], 1),
                   ('a', 'root', [-.00002, 0, 1], 10),
                   ('b', 'socket', [.00002, 0, 1.00003], 10)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](2, 3)[0] == 0
    assert body._twins(mass, np.array([1., 0, 0])) == {}


def test_forked_shoulders_share_arm_and_pad_edges_and_vote_for_the_axis():
    bones = [('root', None, [0, 0, 0], 1), ('a', 'root', [1, 0, 0], 10),
             ('b', 'root', [-1, 0, 0], 10), ('a.short', 'a', [1, 0, .2], 1),
             ('a.long', 'a', [1, 0, .8], 1), ('b.long', 'b', [-1, 0, .8], 1),
             ('b.short', 'b', [-1, 0, .2], 1)]
    rig = _figure(bones)
    mass = body._Mass(rig)
    assert body._copies(mass) == [(1, 2)]
    assert body._shared(mass)[0](1, 2)[0] == 3
    assert body._twins(mass, np.array([1., 0, 0])) == {1: 2, 2: 1, 3: 6, 6: 3, 4: 5, 5: 4}
    assert np.array_equal(body.symmetry_axis(rig), [1, 0, 0])


def test_bone_positions_only_break_ties_in_copied_limb_votes():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('left', 'root', [1, 1, 0], 10), ('left.end', 'left', [1, 1, .5], 10),
                   ('right', 'root', [-1, -1, 0], 10), ('right.end', 'right', [-1, -1, .5], 10),
                   ('front', 'root', [0, 2, 1], 100), ('back', 'root', [0, -2, 1.01], 100)])
    votes = body._copy_steps(body._Mass(rig))
    assert votes[0] == votes[1] > 0
    assert np.array_equal(body.symmetry_axis(rig), [0, 1, 0])


def test_head_strictly_above_arms_excludes_a_heavier_chest():
    rig = _figure([(n, p, h, 1000 if n == 'chest' else w) for n, p, h, w in _bones()])
    result = _read(rig)
    assert rig.names[result.skull] == 'head'
    assert rig.names[result.chest] == 'chest'


def test_ball_uses_heaviest_trunk_bone_when_no_skinned_bone_is_above_arms():
    bones = [('root', None, [0, 0, .5], 100), ('body', 'root', [0, 0, 1], 50),
             ('foot.L', 'root', [.3, 0, 0], 8), ('foot.R', 'root', [-.3, 0, 0], 8),
             ('arm.L', 'body', [.4, 0, 1], 8), ('arm.R', 'body', [-.4, 0, 1], 8)]
    rig = _figure(bones)
    result = _read(rig)
    assert result.plan == 'ball'
    assert rig.names[result.skull] == 'root'
    assert all(result.arms.values())
    rig = _figure(bones + [('head', 'body', [0, 0, 1.5], 1)])
    assert rig.names[_read(rig).skull] == 'head'


def test_a_root_fork_with_no_equal_edges_is_not_a_copy():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [1, 0, 0], 1), ('b', 'root', [-2, 0, 0], 1),
                   ('a.1', 'a', [1, 0, .2], 1), ('a.2', 'a', [1, 0, .4], 1),
                   ('b.1', 'b', [-2, 0, .6], 1), ('b.2', 'b', [-2, 0, .8], 1)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](1, 2) == (0, ())
    assert body._twins(mass, np.array([1., 0, 0])) == {}


def test_float_noise_on_the_plane_is_on_neither_side():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [-1e-15, 0, 1], 1), ('b', 'root', [1e-15, 0, -1], 1)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](1, 2)[0] == 1
    assert body._twins(mass, np.array([1., 0, 0])) == {}


def test_a_terminal_arm_and_its_twin_holding_a_prop_share_the_root_edge():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [1, 0, 0], 1), ('b', 'root', [-1, 0, 0], 1),
                   ('prop', 'b', [-1, 0, 2], 1)])
    assert body._twins(body._Mass(rig), np.array([1., 0, 0])) == {1: 2, 2: 1}


def test_a_near_copy_with_only_a_shared_hand_edge_pairs_outermost():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [1, 0, 0], 1), ('a.fore', 'a', [1.4, 0, 0], 1),
                   ('a.hand', 'a.fore', [1.6, 0, 0], 1),
                   ('b', 'root', [-.9, 0, 0], 1), ('b.fore', 'b', [-1.5, 0, 0], 1),
                   ('b.hand', 'b.fore', [-1.7, 0, 0], 1)])
    mass = body._Mass(rig)
    assert body._shared(mass)[0](1, 4)[0] == 1
    assert body._copies(mass, np.array([1., 0, 0])) == [(1, 4)]
    assert body._twins(mass, np.array([1., 0, 0])) == {1: 4, 4: 1, 2: 5, 5: 2, 3: 6, 6: 3}


def test_three_equal_copies_use_position_not_sub_precision_length_error():
    rig = _figure([('root', None, [0, 0, 0], 1),
                   ('a', 'root', [1, 0, 0], 1),
                   ('far', 'root', [-np.sqrt(.99), .1, 0], 1),
                   ('near', 'root', [-1.00001, 0, 0], 1)])
    assert body._twins(body._Mass(rig), np.array([1., 0, 0])) == {1: 3, 3: 1}


def test_stated_twins_and_axis_win_over_lengths_in_read_and_standing(monkeypatch):
    rig = _figure(_bones())
    rig.twins = {i: rig.names.index(n[:-1] + ('R' if n.endswith('L') else 'L'))
                 for i, n in enumerate(rig.names) if n.endswith(('.L', '.R'))}
    rig.mirror = np.array([1., 0, 0])
    rig.heads[[i for i, n in enumerate(rig.names) if n.endswith('.R')], 0] *= 1.3
    rig.points[:2, 1] = [-.5, .5]  # only the feet touch a candidate floor
    def infer_forbidden(*args):
        raise AssertionError('the builder already stated the axis')
    monkeypatch.setattr(body, '_mirror', infer_forbidden)
    mass = body._Mass(rig)
    assert body._twins(mass, np.array([0., 1, 0])) == rig.twins
    result = _read(rig)
    assert result.plan == 'biped'
    assert rig.names[result.arms['right'][0]] == 'arm.R'
    assert np.array_equal(body.read(rig, [0, 0, 1]).front, [0, -1, 0])
    votes = body.standing(rig, rig.mirror, list(_roles(rig)))
    assert max(votes, key=lambda v: v[1])[0].tolist() == [0, 0, 1]
    rig.twins = {}  # an explicitly empty statement is not a request to infer
    assert body._twins(body._Mass(rig), rig.mirror) == {}


def test_only_the_pairs_named_legs_stand_a_body():
    # Pidgeot's crest plumes reach the floor upside down, its curled foot misses it upright: the plumes held it on its
    # head. What a pair is, the labels ask says; a pair it doesn't name legs holds nothing up, however it touches
    bones = _bones() + [(f'plume.{s}', 'head', [sign * .1, 0, 2.6], 30) for s, sign in (('L', 1), ('R', -1))]
    rig = _figure(bones)
    votes = {tuple(v.astype(int)): n for v, n in body.standing(rig, np.array([1., 0, 0]), list(_roles(rig)))}
    assert votes[(0, 0, 1)] > 0 and votes[(0, 0, -1)] == 0
    votes = {tuple(v.astype(int)): n for v, n in body.standing(rig, np.array([1., 0, 0]), ['plume.L'])}
    assert votes[(0, 0, -1)] > 0 and votes[(0, 0, 1)] == 0


def test_a_file_stands_on_its_origin_only_when_it_is_under_its_footprint_at_its_bottom():
    # Boldore, a rock with no limbs, stands on z = 0, its origin under it: the picture had it on its head
    def solid():
        rig = _figure(_bones())
        rig.points[::2, 1] += .1     # some depth: the origin inside its footprint
        return rig
    rig = solid()
    assert body.on_origin(rig, np.array([1., 0, 0])).tolist() == [0, 0, 1]
    # lifted a fifth of its height off its origin (Pidgeot's file, posed in the air), it says nothing
    rig.points = rig.points + [0, 0, .5]
    assert body.on_origin(rig, np.array([1., 0, 0])) is None
    # nor centred on its middle
    rig.points = rig.points - [0, 0, 1.6]
    assert body.on_origin(rig, np.array([1., 0, 0])) is None
    # nor with its origin off to one side of it, though its bottom is level with it
    rig = solid()
    rig.points = rig.points + [0, 2, 0]
    assert body.on_origin(rig, np.array([1., 0, 0])) is None


def test_child_assignment_maximizes_total_evidence_rather_than_greedy_edges():
    from itertools import permutations
    rng = np.random.default_rng(2)
    for rows, cols in [(2, 3), (3, 2), (4, 4)]:
        cost = -rng.integers(0, 8, size=(rows, cols)).astype(float)
        pairs = body._assignment(cost)
        small = cost if rows <= cols else cost.T
        expected = min(sum(small[i, j] for i, j in enumerate(p))
                       for p in permutations(range(max(rows, cols)), min(rows, cols)))
        assert sum(cost[i, j] for i, j in pairs) == expected


def test_axis_copies_never_count_an_ancestor_and_its_descendant_twice():
    rig = _figure(_bones())
    mass = body._Mass(rig)
    roots = [b for p in body._copies(mass) for b in p]
    assert all(not body._descends(mass.parents, a, b) for a in roots for b in roots if a != b)


def test_near_mirrored_arms_placed_by_hand_are_twins_and_the_head_stays_the_skull():
    """A carnival skeleton's arm bones differ by up to 0.6% of its height from side to side, its hands carry more
    skin than its skull, and its legs pair only by one shin-to-foot edge equal to file precision. Its arms stayed in
    the trunk and a hand was read as the head, so the fighter's spine ran out along an arm."""
    bones = []
    for name, parent, head, weight in _bones():
        head = list(head)
        if name in ('arm.R', 'fore.R', 'hand.R'):
            head = [head[0] * 1.02, head[1], head[2] + .01]
        bones.append((name, parent, head, 50 if name.startswith('hand.') else weight))
    rig = _figure(bones)
    mass = body._Mass(rig)
    assert not body._shared(mass)[0](rig.names.index('arm.L'), rig.names.index('arm.R'))[0]   # no equal edge
    assert frozenset({'arm.L', 'arm.R'}) in _pair_names(rig)
    result = _read(rig)
    assert rig.names[result.skull] == 'head'
    assert _names(rig, result.arms['right']) == ['arm.R', 'fore.R', 'hand.R']


def test_attachments_near_each_others_mirror_from_unrelated_joints_are_not_twins():
    """A costume's brooch on its chest and a skirt strand on its hips can lie near each other's mirror images, nearer
    than to any other joint. Near twins grow outward from one joint or from twins, so these two stay apart."""
    rig = _figure(_bones() + [('brooch.L', 'chest', [.15, -.1, 1.3], 2), ('strand.R', 'hips', [-.16, -.1, 1.31], 2)])
    mass = body._Mass(rig)
    twins = {rig.names[a]: rig.names[b] for a, b in body._twins(mass, np.array([1., 0., 0.])).items()}
    assert 'brooch.L' not in twins and 'strand.R' not in twins
    assert twins['arm.L'] == 'arm.R'
