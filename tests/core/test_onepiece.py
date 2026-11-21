import numpy as np
import pytest

from rigswap.core import onepiece


def _box(lo, hi):
    """A closed box: its 8 corners and 12 triangles."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    points = np.array([[(lo, hi)[i >> k & 1][k] for k in range(3)] for i in range(8)])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return points, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def _figure(parts):
    points, tris = [], []
    for lo, hi in parts:
        p, t = _box(lo, hi)
        tris.append(t + sum(len(q) for q in points))
        points.append(p)
    return np.vstack(points), np.vstack(tris)


# a figure standing on z in a T, mirrored across x = 0, its boxes worn into one another as one surface would carry
# them: legs apart, a neck narrower than the head
PARTS = {
    'leg+': ([.15, -.2, 0], [.45, .2, 2.1]), 'leg-': ([-.45, -.2, 0], [-.15, .2, 2.1]),
    'trunk': ([-.6, -.3, 2], [.6, .3, 4]),
    'arm+': ([.55, -.15, 3.6], [2.4, .15, 3.9]), 'arm-': ([-2.4, -.15, 3.6], [-.55, .15, 3.9]),
    'neck': ([-.15, -.15, 3.95], [.15, .15, 4.35]),
    'head': ([-.4, -.4, 4.3], [.4, .4, 5.1]),
}
UP = np.array([0., 0., 1.])


def _fit(points, tris, cut=True):
    """As read() fits a surface: (its volume, mirror axis and centre, its slices with a node per DOTS of them, the
    Skeleton offering every joint)."""
    vol = onepiece.Volume(points, tris)
    side, centre = onepiece.mirror(vol)
    if cut:
        vol.cut(side, centre, points, tris)
    g = onepiece.Graph(vol, onepiece.top(vol, UP, side, centre), side, centre)
    g.thin()
    return vol, side, centre, g, onepiece._bones(g, g.start)


def _name(skel, place):
    """The joint offered nearest ``place``, by its name (of a Skeleton, or of the Rig a scene reads as)."""
    names = skel.names() if callable(skel.names) else skel.names
    return names[int(np.argmin(np.linalg.norm(skel.heads - np.asarray(place, float), axis=1)))]


def _record(skel, **over):
    """A record naming the T figure's joints, as a look would by their numbers."""
    rec = {'hips': (0, 0, 2.05), 'chest': (0, 0, 3.75), 'neck': (0, 0, 4.05), 'skull': (0, 0, 4.4),
           'legs': {side: {'hip': (s * .3, 0, 1.85), 'knee': (s * .3, 0, 1.0), 'ankle': (s * .3, 0, .25)}
                    for side, s in (('left', 1), ('right', -1))},
           'arms': {side: {'shoulder': (s * .75, 0, 3.75), 'elbow': (s * 1.5, 0, 3.75), 'hand': (s * 2.2, 0, 3.75)}
                    for side, s in (('left', 1), ('right', -1))}}
    rec.update(over)

    def named(x):
        return {k: named(v) for k, v in x.items()} if isinstance(x, dict) else _name(skel, x)
    return named(rec)


def _kids(skel):
    kids = [[] for _ in skel.parents]
    for i, p in enumerate(skel.parents):
        if p >= 0:
            kids[p].append(i)
    return kids


def _below(skel, b):
    kids, out = _kids(skel), [b]
    for x in out:
        out.extend(kids[x])
    return out


def test_the_mirror_plane_parts_left_from_right():
    _, side, centre, _, _ = _fit(*_figure(PARTS.values()), cut=False)
    assert abs(side[0]) == 1 and abs(centre[0]) < .05


def test_joints_are_offered_along_every_part_to_its_end_and_none_is_named():
    _, _, _, _, skel = _fit(*_figure(PARTS.values()))
    parents, heads, tails = skel.parents, skel.heads, skel.tails
    assert parents[0] == -1 and (parents[1:] >= 0).all()
    assert heads[0][2] > 4.8 and abs(heads[0][0]) < .1                # the tree hangs from the top, on the plane
    ends = np.vstack([heads, tails])
    for x, z in ((.3, 0), (-.3, 0), (2.4, 3.75), (-2.4, 3.75), (0, 5.1)):     # feet, hands, the top of the head
        assert np.linalg.norm(ends - [x, 0, z], axis=1).min() < .3, (x, z)
    midline = np.abs(heads[:, 0]) < .1
    assert (midline & (heads[:, 2] > 4.4)).sum() >= 3                 # a head has joints of its own to name,
    assert (midline & (heads[:, 2] > 3.95) & (heads[:, 2] < 4.35)).sum() >= 1     # and so has a neck
    for s in (1, -1):                                               # a joint every little way along a limb
        assert 6 <= (s * heads[:, 0] > .7).sum() <= 20
    assert not skel.spine and not skel.arms and skel.head is None and not skel.twins


def test_legs_pressed_within_a_cell_of_each_other_part_where_the_mirror_plane_is_outside_them():
    # Richard's feet: 0.8 of a cell apart (the plane is the volume's, to a quarter cell, so it lies between them)
    parts = dict(PARTS, **{'leg+': ([.016, -.2, 0], [.316, .2, 2.1]), 'leg-': ([-.316, -.2, 0], [-.016, .2, 2.1])})
    points, tris = _figure(parts.values())
    vol, _, _, _, skel = _fit(points, tris)
    assert .032 < vol.h                                 # the gap between them is under a cell
    ends = np.vstack([skel.heads, skel.tails])
    for x in (.16, -.16):
        assert np.linalg.norm(ends - [x, 0, 0], axis=1).min() < .3, x


def _parting(skel, a, b):
    """Where the ways from the joints nearest ``a`` and ``b`` up the tree first meet."""
    def way(place):
        out = [int(np.argmin(np.linalg.norm(skel.tails - np.asarray(place, float), axis=1)))]
        while skel.parents[out[-1]] >= 0:
            out.append(int(skel.parents[out[-1]]))
        return out
    theirs = set(way(b))
    return skel.heads[next(x for x in way(a) if x in theirs)]


def test_twins_touching_below_where_they_part_are_apart():
    # scarecrow's: his legs bound together by ropes at his knees, apart above them up to his hips
    ropes = _box([-.45, -.2, .8], [.45, .2, 1.1])
    points, tris = _figure(PARTS.values())
    points, tris = np.vstack([points, ropes[0]]), np.vstack([tris, ropes[1] + len(points)])
    _, _, _, _, skel = _fit(points, tris)
    meet = _parting(skel, (.3, 0, 0), (-.3, 0, 0))
    assert 1.6 < meet[2] < 2.4, meet                    # the legs part at the hips, not at the ropes


def test_named_the_tree_hangs_from_the_hips_and_the_head_is_whole():
    vol, _, _, g, offered = _fit(*_figure(PARTS.values()))
    before = (g.alive.sum(), len(offered.parents))
    skel = onepiece.finish(g, vol, _record(offered))
    assert (g.alive.sum(), len(onepiece._bones(g, g.start).parents)) == before      # the fit is left as it was
    parents, heads, tails = skel.parents, skel.heads, skel.tails
    assert parents[0] == -1 and (parents[1:] >= 0).all()
    assert abs(heads[0][0]) < .1 and 1.6 < heads[0][2] < 2.4         # between the legs, at the trunk's bottom
    assert skel.spine[0] == 0 and skel.spine[-1] == skel.head
    assert 4.2 < heads[skel.head][2] < 4.6 and tails[skel.head][2] > 4.8
    assert not (heads[:, 2] > heads[skel.head][2] + .05).any()        # nothing past the skull is a joint of its own
    assert skel.names()[skel.head] == _record(offered)['skull']       # a named joint keeps its name


def test_a_joint_named_past_the_skull_stays_with_the_way_to_it():
    vol, _, _, g, offered = _fit(*_figure(PARTS.values()))
    crown = _name(offered, (0, 0, 4.9))
    skel = onepiece.finish(g, vol, _record(offered, neck=(0, 0, 4.9), skull=(0, 0, 4.05)))
    assert crown in skel.names()
    assert (skel.heads[:, 2] > 4.5).sum() >= 2                        # and the joints on the way to it


def test_a_shoulder_joint_lies_in_from_the_arms_first_slice_and_named_joints_are_twins():
    vol, _, _, g, offered = _fit(*_figure(PARTS.values()))
    rec = _record(offered)
    skel = onepiece.finish(g, vol, rec)
    names = skel.names()
    assert len(skel.arms) == 2
    for a in skel.arms:
        assert skel.parents[a] in skel.spine
        first = offered.heads[offered.names().index(names[a])]         # where its first slice's middle lay
        assert abs(skel.heads[a][0]) < abs(first[0]) and abs(skel.heads[a][0]) > .3
    index = {n: i for i, n in enumerate(names)}
    for kind, parts in (('legs', ('hip', 'knee', 'ankle')), ('arms', ('shoulder', 'elbow', 'hand'))):
        for part in parts:
            a, b = index[rec[kind]['left'][part]], index[rec[kind]['right'][part]]
            assert skel.twins[a] == b and skel.twins[b] == a


def test_a_limbs_joint_named_on_the_trunk_is_its_first_joint_off_the_trunk():
    # Big Smoke's (dots-1): shown the tree hanging from the head, the look named the trunk joint above his right arm
    # as its clavicle; hung from the hips that joint is not above the hand, and the record was refused
    vol, _, _, g, offered = _fit(*_figure(PARTS.values()))
    rec = _record(offered)
    plain = onepiece.finish(g, vol, rec)
    first = {plain.names()[a] for a in plain.arms}
    said = {**rec, 'arms': {**rec['arms'], 'right': {**rec['arms']['right'], 'clavicle': rec['neck']}}}
    held = onepiece.on_limbs(g, said)
    arm = held['arms']['right']
    assert arm.get('clavicle', arm['shoulder']) in first                # the arm's first joint off the trunk
    assert {part: joint for part, joint in arm.items() if part != 'clavicle'} == rec['arms']['right']
    assert held['arms']['left'] == rec['arms']['left']
    assert onepiece.on_limbs(g, rec) == rec                             # a record that holds says what it said
    skel = onepiece.finish(g, vol, said)
    assert skel.names() == plain.names() and np.allclose(skel.heads, plain.heads)
    names = skel.names()
    hand = names.index(arm['hand'])
    assert names.index(arm.get('clavicle', arm['shoulder'])) in _up_from(skel, hand)    # and the hand is below it
    low = {**rec, 'arms': {**rec['arms'], 'left': {**rec['arms']['left'], 'shoulder': rec['chest']}}}
    assert onepiece.on_limbs(g, low)['arms']['left']['shoulder'] in first


def _up_from(skel, b):
    out = [b]
    while skel.parents[out[-1]] >= 0:
        out.append(int(skel.parents[out[-1]]))
    return out


def test_a_body_said_to_have_no_neck_has_its_head_between_its_shoulders():
    # Crash's (dots-1): his neck named at the top of his grin, a few joints above his shoulders, left the rest of
    # the grin on the fighter's chest; the look places a head's start within a joint or two, the fit knows the
    # shoulders. Richard's (dots-4): where the arms leave the spine is not them, his come free of his sides at the waist
    vol, _, _, g, offered = _fit(*_figure(PARTS.values()))
    rec = _record(offered)
    assert onepiece.on_shoulders(g, vol, rec) == rec                    # a record naming its neck says what it said
    bare = {k: v for k, v in rec.items() if k != 'neck'}
    assert onepiece.on_shoulders(g, vol, bare) == bare                  # as one with no such entry: the skull's
    low = _record(offered, chest=(0, 0, 3.0))                           # a chest named below the shoulders
    held = onepiece.on_shoulders(g, vol, {**low, 'neck': None})
    assert {k: v for k, v in held.items() if k != 'neck'} == {k: v for k, v in low.items() if k != 'neck'}
    skel = onepiece.finish(g, vol, {**low, 'neck': None})
    names = skel.names()
    neck = names.index(held['neck'])
    shoulders = skel.heads[skel.arms].mean(axis=0)
    assert neck in skel.spine and abs(skel.heads[neck][2] - shoulders[2]) < .2     # the spine joint between them,
    hung = {int(skel.parents[a]) for a in skel.arms}
    assert skel.spine.index(neck) >= max(skel.spine.index(b) for b in hung)       # never before where the arms leave
    assert skel.heads[neck][2] < offered.heads[offered.names().index(rec['neck'])][2]   # below the neck it had
    high = onepiece.on_shoulders(g, vol, {**rec, 'neck': None, 'chest': rec['neck']})
    assert high['chest'] == high['neck'] == held['neck']                # a chest named past it is that joint
    armless = {k: v for k, v in rec.items() if k != 'arms'}
    assert onepiece.on_shoulders(g, vol, {**armless, 'neck': None})['neck'] == rec['chest']  # no arms: the chest's


def test_a_record_naming_no_joint_of_the_fit_is_refused():
    vol, _, _, g, offered = _fit(*_figure(PARTS.values()))
    with pytest.raises(ValueError, match='no joint fitted'):
        onepiece.finish(g, vol, _record(offered) | {'hips': 'bone0'})    # as a record made before the fit offered joints


def _tube(rings, half=(.15, .15)):
    """A closed surface through rings (each a centre, its rectangle level, ``half`` its half sides), capped at both
    ends: one piece."""
    points = np.array([c + np.array([sx * half[0], sy * half[1], 0]) for c in np.asarray(rings, float)
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    n = 4 * len(rings)
    tris = [(0, 2, 1), (0, 3, 2), (n - 4, n - 3, n - 2), (n - 4, n - 2, n - 1)]
    for r in range(len(rings) - 1):
        for k in range(4):
            a, b = 4 * r + k, 4 * r + (k + 1) % 4
            tris += [(a, b, b + 4), (a, b + 4, a + 4)]
    return points, np.array(tris)


@pytest.mark.blender                    # mathutils' KD tree
def test_an_arm_hanging_by_the_trunk_joins_it_where_its_own_surface_begins():
    # Richard's: each arm a surface of its own, against the trunk from the shoulder (3.95) down to the elbow, where
    # it parts; fused with the trunk in the grid, its slices begin by the elbow, and so do the joints a look can name
    trunk = _tube([[0, 0, z] for z in (2, 2.5, 3, 3.5, 4)], half=(.6, .3))
    arms = [_tube([[s * .75, 0, 3.95], [s * .75, 0, 3], [s * 1.1, 0, 2.65], [s * 1.45, 0, 2.3]]) for s in (1, -1)]
    nose = ([-.1, -.55, 4.6], [.1, -.35, 4.8])         # a front: left and right are the only mirror images
    boxes = _figure([PARTS[k] for k in ('leg+', 'leg-', 'neck', 'head')] + [nose])
    points, tris = [], []
    for p, t in [trunk, *arms, boxes]:
        tris.append(t + sum(len(q) for q in points))
        points.append(p)
    points, tris = np.vstack(points), np.vstack(tris)
    vol, _, _, g, offered = _fit(points, tris, cut=False)
    rec = _record(offered, chest=(0, 0, 2.9), arms={
        side: {'shoulder': (s * 1.0, 0, 2.75), 'hand': (s * 1.4, 0, 2.35)} for side, s in (('left', 1), ('right', -1))})
    skel = onepiece.finish(g, vol, rec, points, tris)
    assert len(skel.arms) == 2
    for a in skel.arms:
        assert skel.heads[a][2] > 3.5                   # the shoulder, not the elbow (3)
    assert skel.heads[skel.head][2] > 3.9


@pytest.mark.blender                    # mathutils' KD tree
def test_the_skin_keeps_each_part_on_its_own_bones():
    points, tris = _figure(PARTS.values())
    vol, _, _, g, offered = _fit(points, tris, cut=False)
    skel = onepiece.finish(g, vol, _record(offered), points, tris)
    w = onepiece.slices(points, tris, vol, skel)
    assert np.allclose(w.sum(axis=1), 1)
    for a in skel.arms:                             # a hand's corners ride its arm alone
        hand = np.flatnonzero(points[:, 0] * np.sign(skel.heads[a][0]) > 2.3)
        assert (w[hand][:, _below(skel, a)].sum(axis=1) > .99).all()
    top = np.flatnonzero(points[:, 2] > 5)          # the head is whole
    assert (w[top, skel.head] > .99).all()


@pytest.mark.blender
def test_a_scene_is_fitted_then_named_and_can_go_back_to_its_offered_joints(tmp_path):
    import re
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    points, tris = _figure(PARTS.values())
    data = bpy.data.meshes.new('statue')
    data.from_pydata(points.tolist(), [], tris.tolist())
    bpy.context.scene.collection.objects.link(bpy.data.objects.new('statue', data))
    blend = tmp_path / 'statue.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    try:
        offered = onepiece.read(blend, tmp_path, None, up=UP)           # a record's up: nothing is asked
        assert all(re.fullmatch(r'slice\d+', n) for n in offered.names) and offered.twins is None
        assert np.allclose(offered.weights.sum(axis=1), 1) and abs(offered.mirror[0]) == 1
        assert onepiece.FITTED.up[1] == 'record'
        rec = _record(offered)
        rig, held = onepiece.named(rec)
        assert held == rec                                              # its limbs' joints lay on the limbs
        assert rig.names[int(np.flatnonzero(rig.parents < 0)[0])] == rec['hips']    # the tree hangs from the hips
        assert rig is onepiece.named(rec)[0] and rig.twins
        assert len(rig.names) < len(offered.names)
        top = rig.points[:, 2] > 5
        assert (rig.weights[top, rig.names.index(rec['skull'])] > .99).all()        # the head is whole
        assert len([ob for ob in bpy.data.objects if ob.type == 'ARMATURE']) == 1   # in place of the one before
        back = onepiece.unnamed()
        assert back.names == offered.names and back.twins is None
        with pytest.raises(ValueError, match='no joint fitted'):
            onepiece.named(rec | {'hips': 'bone0'})
    finally:
        onepiece.FITTED = None
        bpy.ops.wm.read_factory_settings(use_empty=True)


def test_a_surface_takes_the_room_it_walls_in_even_open_on_one_side():
    points, tris = _box([0, 0, 0], [1, 2, 3])
    closed = onepiece.room(points, tris, cells=64)
    assert 5.4 < closed < 6.6
    hem = tris[2:]                                              # a dress: open at its bottom
    assert abs(onepiece.room(points, hem, cells=64) - closed) < .05 * closed
    apart = _figure([([0, 0, 0], [1, 1, 1]), ([3, 0, 0], [4, 1, 1])])   # nothing between two boxes
    assert onepiece.room(*apart, cells=64) < 2.5


def test_an_outline_is_what_a_surface_covers_along_an_axis_with_its_holes():
    def plate(lo, hi, y=0.):
        return np.array([(lo, y, lo), (hi, y, lo), (hi, y, hi), (lo, y, hi)], float)
    frame = [plate(0, 1), plate(4, 5), np.array([(0, 0, 4), (5, 0, 4), (5, 0, 5), (0, 0, 5.)]),
             np.array([(0, 0, 0), (5, 0, 0), (5, 0, 1), (0, 0, 1.)]), np.array([(0, 0, 0), (1, 0, 0), (1, 0, 5), (0, 0, 5.)]),
             np.array([(4, 0, 0), (5, 0, 0), (5, 0, 5), (4, 0, 5.)])]
    points = np.vstack(frame)
    tris = np.vstack([[(k, k + 1, k + 2), (k, k + 2, k + 3)] for k in range(0, len(points), 4)])
    assert abs(onepiece.outline(points, tris, 1) - 16) < 1.5        # 25 less the hole's 9
    assert onepiece.outline(points, tris, 0) < 1


def test_a_line_is_looked_at_square_along_the_way_its_steps_reach_least():
    square, along = onepiece.across([[1, 0, 5], [1, 0, 3], [1, -2, 3], [1, -4, 3]])     # down, then forward
    assert np.allclose(np.abs(square), [1, 0, 0])
    assert onepiece.across([[0, 0, 0], [0, 0, 1], [0, 0, 3]])[0] is None                # straight: no way of its own
    assert onepiece.across([[0, 0, 0]])[0] is None
    # a leg that only slants, and its foot: from the side, where the places alone would read it as flat
    square, _ = onepiece.across([[1, 0, 9], [1.3, 0, 6], [1.6, 0, 3], [1.9, 0, 0.3], [1.9, -1.5, 0]])
    assert abs(square[0]) > 0.95


def test_a_shoulder_a_look_saw_at_the_spine_joint_goes_into_the_arms_own_plane():
    """Goku's right arm (arm-1, 2026-11-10): held against the body, its slices begin by the elbow; the joint lay
    under the sleeve's hem and the upper arm's bone was a third as long as the left one's. The look, shown the arm
    square on, named the shoulder at the chest's joint: the joint goes where that joint is seen, at the arm's depth."""
    from types import SimpleNamespace
    # node 0 the chest's middle; 1, 2, 3 the arm: it leaves well in front of the chest, runs down, then bends out
    middle = np.array([[0., 0., 10.], [3., 6., 9.], [3., 6., 6.], [6., 6., 4.]])
    g = SimpleNamespace(middle=middle, thick=np.array([3., 1., 1., 1.]))
    kids = {0: [1], 1: [2], 2: [3], 3: []}
    plain = onepiece.shoulder_joints(g, middle[0], kids, [[1, 2, 3]], [], None)
    reach = middle[1] - middle[0]
    assert np.allclose(plain[0], middle[1] - reach / np.linalg.norm(reach))             # its thickness in, as before
    seen = onepiece.shoulder_joints(g, middle[0], kids, [[1, 2, 3]], [], None, seen=[(0, 3)])
    assert np.allclose(seen[0], [0., 6., 10.])
    assert np.allclose(onepiece.shoulder_joints(g, middle[0], kids, [[1, 2, 3]], [], None, seen=[None])[0], plain[0])
    g.middle = np.array([[0., 0., 10.], [3., 6., 9.], [3., 6., 6.], [3., 6., 3.]])      # a straight arm has no plane
    plain = onepiece.shoulder_joints(g, g.middle[0], kids, [[1, 2, 3]], [], None)
    assert np.allclose(onepiece.shoulder_joints(g, g.middle[0], kids, [[1, 2, 3]], [], None, seen=[(0, 3)])[0],
                       plain[0])
