import numpy as np
import pytest

from rigswap.core import repose
from rigswap.core.correspond import Match


def test_a_segment_stretches_between_its_joints_and_what_lies_past_them_keeps_its_distance():
    # a shin 1 long in the model, 4 long at the fighter's rest, both hanging straight down
    move = repose._bone(np.array([0, 0, 2.]), np.array([0, 0, 1.]), np.array([0, 0, 8.]), np.array([0, 0, 4.]))
    on, below, above = move(np.array([[0, 0, 1.5], [0, 0, 0.5], [0, 0, 2.25]]))
    assert np.isclose(on[2], 6) and np.isclose(below[2], 3.5) and np.isclose(above[2], 8.25)


def test_the_trunk_thickens_by_its_whole_stretch_and_a_limb_by_its_own():
    # a trunk 4 long in the model and the fighter, its joint set high in the model: hips 3 squashed to 2, neck 1
    # stretched to 2. A shin beside it stretched 1 to 4.
    embedded = {0: ([0, 0, 0], [0, 0, 3]), 1: ([0, 0, 3], [0, 0, 4]), 2: ([5, 0, 2], [5, 0, 1])}
    rest = {0: ([0, 0, 0], [0, 0, 2]), 1: ([0, 0, 2], [0, 0, 4]), 2: ([5, 0, 8], [5, 0, 4])}
    moves = repose.transforms(Match({}, embedded, rest, trunk=[0, 1]), [0, 1, 2])
    mane = moves[1](np.array([[0, -3, 3.5]]))[0]          # 3 in front of the neck: stays 3 in front
    assert np.allclose(mane, [0, -3, 3])
    assert np.isclose(moves[2](np.array([[5, -1, 1.5]]))[0][1], -2)     # the shin thickens by its own 4: 2 across
    # a trunk stretched as a whole thickens as a whole, its squashed segment too
    rest[1] = ([0, 0, 2], [0, 0, 16])          # 16 in all: 4 times
    moves = repose.transforms(Match({}, embedded, rest, trunk=[0, 1]), [0, 1])
    assert np.isclose(moves[0](np.array([[0, -1, 1.5]]))[0][1], -2)


@pytest.mark.parametrize('front', [0.89, 0.91])
def test_a_segment_turns_the_least_way_and_keeps_its_back_behind(front):
    # Bowser Odyssey's back bone: forward-up in the model, up and a little back at the fighter's rest, its shell
    # behind it. Either side of where frames from a fixed front switched to up.
    te = np.array([0, -front, np.sqrt(1 - front ** 2)])
    move = repose._bone(np.zeros(3), te, np.zeros(3), np.array([0, 0.291, 0.957]))
    shell, side = move(np.array([te / 2 + [0, np.sqrt(1 - front ** 2), front], te / 2 + [1, 0, 0]]))
    assert shell[1] > 0.5 and np.isclose(side[0], 1, atol=1e-3)


def test_an_end_keeps_its_size_everywhere():
    move = repose._bone(np.array([0, 0, 1.]), np.array([0, -1, 1.]), np.array([0, 0, 3.]), np.array([0, -2, 3.]),
                        stretch=False)
    assert np.allclose(move(np.array([[0, -3, 1.]])), [[0, -3, 3]])


@pytest.mark.parametrize('length', [0.5, 3.0])
@pytest.mark.parametrize('end', ['foot', 'hand', 'head'])
def test_parts_of_long_and_short_ends_share_the_whole_move_despite_different_rest_directions(length, end):
    import copy
    # Parent end along x in the model, y in the fighter. The child turns from x to z.
    # Deliberately request the grandchild first: numeric IDs need not follow the hierarchy.
    child, tip = end + ' part', end + ' tip'
    embedded = {end: ([0, 0, 0], [length, 0, 0]),
                child: ([length, 0, 0], [length, 0, 0.4]),
                tip: ([length, 0, 0.4], [length, 0.2, 0.4])}
    rest = {end: ([1, 2, 3], [1, 3, 3]), child: ([1, 3, 3], [1, 3, 5]),
            tip: ([1, 3, 5], [2, 3, 5])}
    match = Match({}, embedded, rest, ends=set(embedded), end_parents={child: end, tip: child})
    original_rest = copy.deepcopy(rest)
    moves = repose.transforms(match, [tip, child, end])
    for part, parent in match.end_parents.items():
        joint = np.asarray([embedded[part][0]], float)
        assert np.allclose(moves[part](joint), moves[parent](joint))
        # Off-axis points must coincide too: sharing only the joint would pass the old broken placement.
        points = np.array([[0.2, 0.3, -0.1], [length, 0.4, 0.5], [length + 0.5, -0.1, 0.2]])
        assert np.allclose(moves[part](points), moves[parent](points))
        assert np.allclose(moves[part].m[:, :3].T @ moves[part].m[:, :3], np.eye(3))
    assert np.allclose(moves[child](np.asarray([embedded[child][0]], float)), [[1, 2 + length, 3]])
    assert match.rest == original_rest                # the game's pivots remain exactly where they were
    # Blending any parts of this end also preserves its shape exactly.
    weights = np.array([[0.2, 0.3, 0.5], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(repose.deform(points, weights, [end, child, tip], moves), moves[end](points))


def test_an_explicit_prop_placement_is_preserved_inside_an_end():
    match = Match({}, {0: ([0, 0, 0], [1, 0, 0]), 1: ([1, 0, 0], [2, 0, 0])},
                  {0: ([0, 0, 0], [3, 0, 0]), 1: ([3, 0, 0], [4, 0, 0])}, ends={0, 1},
                  end_parents={1: 0}, placed={1: np.column_stack([np.eye(3), [9, 8, 7]])})
    assert np.allclose(repose.transforms(match, [1, 0])[1](np.zeros((1, 3))), [[9, 8, 7]])


def test_a_prop_a_slot_took_rides_it_alone():
    w = np.array([[0.7, 0.3, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 0.8]])
    out = repose.alone(w, [10, 11, 12], {11: np.array([True, True, False]), 13: np.array([True, False, True])})
    assert np.array_equal(out, [[0, 1, 0], [0, 1, 0], [0.2, 0, 0.8]]) and w[0, 0] == 0.7


@pytest.mark.parametrize('heading', [[0, -3, -2], [-3, 0, 2], [0, 0, -2]])
def test_a_flat_foot_yaws_toward_a_tilted_fighter_foot_and_its_parts_stay_on_the_floor(heading):
    # The source sole is at z=2, its chosen ankle one unit above it. The fighter ankle is one above its floor.
    ankle = np.array([4, 5, 1.])
    match = Match({}, {0: ([0, 0, 3], [2, 0, 2.5]), 1: ([2, 0, 2.5], [3, 0, 2])},
                  {0: (ankle, ankle + heading), 1: (ankle + heading, ankle + [0, 0, 4])},
                  ends={0, 1}, feet={0}, end_parents={1: 0})
    moves = repose.transforms(match, [1, 0])
    sole = np.array([[-1, -1, 2], [2, -1, 2], [2, 1, 2], [-1, 1, 2.]])
    placed = moves[0](sole)
    assert np.allclose(placed[:, 2], 0, atol=1e-14)
    assert np.allclose(moves[0](np.array([[0, 0, 3.]])), [ankle])
    assert np.allclose(moves[0].m[:, :3].T @ moves[0].m[:, :3], np.eye(3))
    assert moves[1] is moves[0]
    if np.any(heading[:2]):
        facing = moves[0].m[:2, 0]
        assert np.allclose(facing, np.array(heading[:2]) / np.linalg.norm(heading[:2]))
    else:
        assert np.allclose(moves[0].m[:, :3], np.eye(3))


@pytest.mark.parametrize('sole', [-1.0, 0.0, 0.5])
def test_the_floor_conflict_reports_the_placed_skin_without_moving_the_ankle(sole):
    from types import SimpleNamespace
    embedded = {0: ([0, 0, 1], [0, -1, 1])}
    match = Match({}, embedded, embedded, ends={0}, feet={0})
    moves = repose.transforms(match, [0])
    points = np.array([[0, 0, sole], [0, -1, sole + 1]])
    before = points.copy()
    foot = repose.end_report(match, SimpleNamespace(names=['foot'], parents=[-1]), [0], np.ones((2, 1)),
                             points, moves)['feet'][0]
    assert foot['sole_z'] == sole and foot['grounding_translation_z'] == -sole
    assert foot['ankle_error'] == 0 and foot['conflict'] == (sole != 0)
    assert np.array_equal(points, before)


def test_a_bone_a_fighter_joint_falls_inside_is_cut_there():
    from types import SimpleNamespace
    from rigswap.core.correspond import Chain
    # one model bone from the shoulder (x 0) to the end (x 4); the fighter's elbow at x 2 and wrist at x 3 inside it
    pts = np.array([[0.1, 0, 0], [1.0, 0, 0], [2.5, 0, 0], [3.5, 0, 0]])
    rig = SimpleNamespace(points=pts, weights=np.full((4, 2), 0.5))     # skin shared with bone 1, also on 10
    arm = Chain([0], np.array([[0, 0, 0.]]), np.array([4, 0, 0.]))
    w = repose.split(rig, np.array([[1, 0, 0]] * 4, float), [10, 11, 12], [(0, arm, [(0.5, 11), (0.75, 12)])],
                     {0: 10, 1: 10})
    assert np.allclose(w, [[1, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]])


def test_a_head_placed_by_its_neck_takes_the_records_head_size_about_its_own_joint():
    """A model with no neck: the fighter's neck has no length inside it and the head, an end, is placed by the neck's
    move. The record's head size still applies there, and the joint stays where the neck's move puts it."""
    neck, head = 0, 1
    joint = [0., 0, 2]
    embedded = {neck: (joint, joint), head: (joint, [0, 0, 3])}
    rest = {neck: ([0, 0, 5], [0, 0, 5.5]), head: ([0, 0, 5.5], [0, 0, 7])}
    match = Match({}, embedded, rest, ends={head}, end_parents={head: neck}, head=head)
    crown = np.array([[0., 0, 3], [1, 0, 2]])
    try:
        plain = repose.transforms(match, [neck, head])[head](crown)
        repose.DIALS = {'head': 0.8}
        sized = repose.transforms(match, [neck, head])
    finally:
        repose.DIALS = {}
    at = plain[0] - [0, 0, 1]                       # where the neck's move puts the head's joint
    assert np.allclose(sized[head](np.array([joint])), [at])
    assert np.allclose(sized[head](crown), at + 0.8 * (plain - at))
    assert np.allclose(sized[neck](crown), plain), 'the neck itself is not sized'
