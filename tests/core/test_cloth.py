import numpy as np

from rigswap.core import cloth, correspond, repose
from rigswap.core.correspond import Chain, Match
from rigswap.core.rig import Rig

HIPS, L_THIGH, L_SHIN, R_THIGH, R_SHIN = 0, 4, 5, 7, 8


def _skirtless():
    """A fighter whose game swings no skirt (its bones: hips 0, spine 1, chest 2, head 3, left leg 4-6, right leg 7-9,
    arms 10-13; hip joints 10 high, knees 5, ankles 1), wearing a model whose skirt hangs from its pelvis: a strand on
    the left (vertices 1-3, its top above the hip joint) and one down the front (4-5), both cloth, loose; and a helper
    bone at the right hip whose patch of the hips' skin (6) runs on down into the thigh's (7), no hair or cloth, riding
    the hips. The pelvis (0) is the model's own and rides the hips; the thigh skin rides the right thigh."""
    def chain(bones, joints, end, anchors, kind=None):
        return Chain(bones, np.array(joints, float), np.array(end, float), anchors, kind=kind)
    fch = {'spine': chain([0, 1, 2, 3], [[0, 0, 10], [0, 0, 11], [0, 0, 14], [0, 0, 16]], [0, 0, 18], [0, 2, 3]),
           'left arm': chain([10, 11], [[1, 0, 14], [3, 0, 14]], [5, 0, 14], [0, 0, 1]),
           'right arm': chain([12, 13], [[-1, 0, 14], [-3, 0, 14]], [-5, 0, 14], [0, 0, 1]),
           'left leg': chain([4, 5, 6], [[1, 0, 10], [1, 0, 5], [1, 0, 1]], [1, -1, 0], [0, 2, None]),
           'right leg': chain([7, 8, 9], [[-1, 0, 10], [-1, 0, 5], [-1, 0, 1]], [-1, -1, 0], [0, 2, None])}
    names = ['pelvis', 'left strand', 'front strand', 'helper', 'thigh skin']
    points = np.array([[0, 0, 10.5], [2, 0, 10.5], [3, 0, 7], [3, 0, 2], [0, -3, 9], [0, -3, 2], [-1.5, 0, 9],
                       [-1.5, 0, 8]], float)
    weights = np.zeros((len(points), len(names)))
    weights[[0], 0] = weights[[1, 2, 3], 1] = weights[[4, 5], 2] = weights[[6], 3] = weights[[7], 4] = 1.0
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [0, 6], [6, 7]])
    rig = Rig('test', names, np.array([-1, 0, 0, 0, -1]), np.zeros((5, 3)), np.zeros((5, 3)), ['skin'], points,
              weights, edges)
    embedded = {HIPS: ((0, 0, 10), (0, 0, 11))}
    for c in (fch['left leg'], fch['right leg']):
        embedded.update({b: (c.joints[i], c.joints[i + 1]) for i, b in enumerate(c.bones[:2])})
    rest = {b: tuple(np.asarray(x) * 0.5 for x in e) for b, e in embedded.items()}    # the fighter half as tall
    loose = [(0, chain([1], [[2, 0, 10.5]], [3, 0, 2], [], 'cloth'), [1]),
             (0, chain([2], [[0, -3, 9]], [0, -3, 2], [], 'cloth'), [2])]
    m = Match({0: HIPS, 1: HIPS, 2: HIPS, 3: HIPS, 4: R_THIGH}, embedded, rest, loose=loose)
    return rig, m, fch


def test_a_skirt_is_the_cloth_off_the_lower_body():
    # a lock hung from the hips is hair: it hangs on its own, and the skirt is the strand of cloth alone
    rig, m, fch = _skirtless()
    m.loose[1][1].kind = 'hair'
    g = cloth.legs(rig, m, fch)
    assert g.bones == [1] and [x[1].bones for x in m.loose] == [[2]]


def test_cloth_lying_wholly_above_the_hip_joint_is_no_skirt():
    # a ball's bandana (Waddle Dee's: his whole body is the fighter's hips) is cloth off the lower body, but none of it
    # lies below the hip joint, where a skirt is laid on the legs: it stays loose
    rig, m, fch = _skirtless()
    rig.points[[4, 5], 2] = 12, 13
    g = cloth.legs(rig, m, fch)
    assert g.bones == [1] and [x[1].bones for x in m.loose] == [[2]]


def _held(rig, m, bones, g):
    """The shares that hold the model at rest, on ``bones`` (each fighter bone's own and the gown's, summed)."""
    by, w, _ = cloth.rest(rig, m.owner, bones, g, repose.transforms(m, bones))
    out = np.zeros((len(w), len(bones)))
    for i, b in enumerate(by):
        out[:, bones.index(b[1] if isinstance(b, tuple) else b)] += w[:, i]
    return out


def test_a_skirt_on_a_fighter_with_no_skirt_is_held_by_its_legs_by_height_and_swung_from_its_knees():
    rig, m, fch = _skirtless()
    g = cloth.legs(rig, m, fch)
    assert g.columns == [L_THIGH, L_SHIN, R_THIGH, R_SHIN, HIPS]
    assert m.loose == []                                    # both strands on the legs
    bones = [HIPS, L_THIGH, L_SHIN, R_THIGH, R_SHIN]
    held, w = _held(rig, m, bones, g), cloth.weights(rig, m.owner, bones, g)
    assert np.allclose(held[1], [1, 0, 0, 0, 0]) and np.allclose(w[1], held[1])    # above the hip joint: the hips
    assert np.allclose(held[2], [0, 0.4, 0.6, 0, 0])        # the left side: by height, its thigh then shin
    assert np.allclose(held[3], [0, 0, 1, 0, 0])            # below the knee: the shin
    assert np.allclose(held[5], [0, 0, 0.5, 0, 0.5])        # the front: both legs
    assert np.allclose(w[2], [0, 0.7, 0.3, 0, 0])           # swung: what the shin holds, thigh and shin alike
    assert np.allclose(w[3], [0, 0.5, 0.5, 0, 0]) and np.allclose(w[5], [1, 0, 0, 0, 0])
    assert np.allclose(held[[6, 7]], [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])          # the helper's skin as bound
    assert np.allclose(w[[6, 7]], held[[6, 7]])


def test_a_gown_is_held_squashed_with_the_legs_and_never_stretched():
    rig, m, fch = _skirtless()                              # the fighter's legs half as long: squashed
    g = cloth.legs(rig, m, fch)
    # 3 below the knee and 2 out from the shin: 1.5 below the fighter's, as far out
    assert np.allclose(g.moves[L_SHIN](np.array([[3, 0, 2.]])), [[2.5, 0, 1]])
    rig, m, fch = _skirtless()
    m.rest = {b: tuple(np.asarray(x) * 2.0 for x in e) for b, e in m.embedded.items()}   # twice as long: kept
    g = cloth.legs(rig, m, fch)
    hem = g.moves[L_SHIN](np.array([[1, 0, 2.]]))
    assert np.allclose(hem, [[2, 0, 20 - 8]])               # 8 below the hip joint, as in the model


def _round_gown():
    """A synthetic skirt ring below both knees, authored wholly on a loose skirt strand."""
    rig, m, fch = _skirtless()
    phi = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    rig.points = np.column_stack([3 * np.sin(phi), -3 * np.cos(phi), np.full(len(phi), 2.)])
    rig.weights = np.zeros((len(phi), len(rig.names)))
    rig.weights[:, 1] = 1
    rig.edges = np.column_stack([np.arange(len(phi)), np.roll(np.arange(len(phi)), -1)])
    m.loose = m.loose[:1]
    return rig, m, fch, phi


def test_a_round_gown_hangs_between_the_legs_and_follows_only_its_own_side():
    rig, m, fch, phi = _round_gown()
    g = cloth.legs(rig, m, fch)
    bones = [HIPS, L_THIGH, L_SHIN, R_THIGH, R_SHIN]
    w = cloth.weights(rig, m.owner, bones, g)
    left, right = w[:, 1:3].sum(axis=1), w[:, 3:5].sum(axis=1)
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1)
    assert np.all((left == 0) | (right == 0))              # never average opposite rotations
    assert np.allclose(w[np.isclose(np.sin(phi), 0), 0], 1)  # front and back on hips
    assert np.isclose(left[np.isclose(phi, np.pi / 2)], 1)
    assert np.isclose(right[np.isclose(phi, -np.pi / 2)], 1)
    toward_left = (phi >= 0) & (phi <= np.pi / 2)
    assert np.all(np.diff(left[toward_left]) >= 0)
    assert np.allclose(left, right[np.mod(-np.arange(len(phi)), len(phi))])
    # Fitting keeps the old angular blend on the shins, including front/back, and the hip column holds nothing.
    held = _held(rig, m, bones, g)
    assert np.all(held[:, [0, 1, 3]] == 0)
    distance = np.abs((phi - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)
    assert np.allclose(held[:, 2], 1 - distance / np.pi)
    assert np.allclose(held[np.isclose(np.sin(phi), 0)], [0, 0, .5, 0, .5])


def test_a_gowns_angular_bend_uses_the_shared_deformation_band(monkeypatch):
    rig, m, fch, phi = _round_gown()
    g = cloth.legs(rig, m, fch)
    at = np.flatnonzero(np.isclose(phi, np.pi / 8))[0]      # halfway through the quarter-interval band
    assert np.allclose(g.ring[at], [.25, .25, 0, 0, .5])
    monkeypatch.setattr(repose, 'BLEND', repose.BLEND / 2)
    rig, m, fch, _ = _round_gown()
    narrow = cloth.legs(rig, m, fch)
    assert np.allclose(narrow.ring[at], [.5, .5, 0, 0, 0])
    assert np.array_equal(g.held, narrow.held)


def test_opposite_leg_rotations_do_not_collapse_the_front_and_back_of_a_gown():
    rig, m, fch, phi = _round_gown()
    g = cloth.legs(rig, m, fch)
    bones = [HIPS, L_THIGH, L_SHIN, R_THIGH, R_SHIN]
    moves = {HIPS: repose.Move(np.eye(4)[:3])}
    pivot = np.array([0., 0., 10.])
    for angle, side in [(np.pi / 3, (L_THIGH, L_SHIN)), (-np.pi / 3, (R_THIGH, R_SHIN))]:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        for bone in side:
            moves[bone] = repose.Move(np.column_stack([rotation, pivot - rotation @ pivot]))
    w = cloth.weights(rig, m.owner, bones, g)
    swung = repose.deform(rig.points, w, bones, moves)
    middle = np.isclose(np.sin(phi), 0)
    assert np.allclose(swung[middle], rig.points[middle])
    old = repose.deform(rig.points, _held(rig, m, bones, g), bones, moves)
    assert np.isclose(np.ptp(old[middle, 1]), np.ptp(rig.points[middle, 1]) / 2)
    for angle, bone in [(np.pi / 2, L_THIGH), (-np.pi / 2, R_THIGH)]:
        side = np.isclose(phi, angle)
        assert np.allclose(swung[side], moves[bone](rig.points[side]))


def test_the_hips_share_goes_with_the_skirt_only_on_its_own_surface_below_the_hip_joint():
    rig, m, fch = _skirtless()
    rig.weights[4] = [0.6, 0, 0.4, 0, 0]                    # the top of the front hung partly on the pelvis
    rig.weights[7] = [1.0, 0, 0, 0, 0]                      # the crotch's skin on the pelvis, below the hip joint
    bones = [HIPS, L_THIGH, L_SHIN, R_THIGH, R_SHIN]
    held = _held(rig, m, bones, cloth.legs(rig, m, fch))
    assert np.allclose(held[4], [0, 0.4, 0.1, 0.4, 0.1]) and np.allclose(held[7], [1, 0, 0, 0, 0])


def test_vertices_joined_by_edges_are_one_piece():
    assert cloth._pieces(5, np.array([[0, 1], [2, 1], [3, 4]])).tolist() == [0, 0, 0, 1, 1]
    assert cloth._pieces(3, None).tolist() == [0, 1, 2]


def test_a_model_with_no_skirt_keeps_its_own_skin():
    rig, m, fch = _skirtless()
    m.loose[:] = m.loose[2:]
    assert cloth.legs(rig, m, fch) is None


def test_a_vertex_between_two_strands_goes_to_both_by_how_near_it_is():
    theta = np.sort(np.radians([0, 90, 180, -90]))
    (a, u), (b, v) = cloth._between(np.radians([45.0, 170.0]), theta)
    assert np.degrees(theta[a]).round().tolist() == [0, 90] and np.degrees(theta[b]).round().tolist() == [90, 180]
    assert np.allclose(u, [0.5, 1 - 80 / 90]) and np.allclose(u + v, 1)


def test_a_vertex_is_shared_along_a_strand_by_how_far_it_lies_from_its_root():
    along = cloth._along(np.array([0, 0.5, 1.75, 2.5]), [0, 1, 2, 3.])
    assert np.allclose(along, [[1, 0, 0], [0.5, 0.5, 0], [0, 0.25, 0.75], [0, 0, 1]])


def test_a_strand_follows_the_gown_where_it_bends_out_along_the_floor():
    down = [[0, 0, -z] for z in np.linspace(0.5, 4, 8)]
    out = [[0, y, -4] for y in np.linspace(0.5, 4, 8)]
    pts = np.array(down + out, float)
    curve = cloth._curve(pts, np.ones(len(pts)), np.ones(len(pts)), np.zeros(3), np.array([0, 0, -1.]))
    assert np.all(np.diff(np.linalg.norm(curve, axis=1)) > 0)
    assert curve[1][1] < 0.5 < curve[-1][1] and curve[-1][2] < -3


def test_a_chain_is_cut_where_it_first_comes_down_to_a_height():
    mantle = Chain([0, 1, 2], np.array([[0, 1, 10], [0, 2, 7], [0, 3, 4.]]), np.array([0, 4, 1.]))
    at, cut = correspond._passing(mantle, 8.0)
    assert np.isclose(at, np.linalg.norm([1, 3]) * 2 / 3)
    assert cut.bones == [0, 1, 2] and np.allclose(cut.joints[0], [0, 1 + 2 / 3, 8]) and np.allclose(cut.joints[1:],
                                                                                                    mantle.joints[1:])
    assert correspond._passing(mantle, 0.5) is None
