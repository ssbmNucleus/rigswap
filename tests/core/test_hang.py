import numpy as np

from rigswap.core import hang


def _nothing(a, b):
    return np.zeros(len(a), bool)


def _below(height):
    """A body filling everything under ``height``: a move passes into it when it goes down past that height."""
    def passes(a, b):
        return (a[:, 2] >= height) & (b[:, 2] < height)
    return passes


def test_a_feeler_bound_straight_out_hangs_straight_down_from_where_it_leaves_the_head():
    """Mega Lucario's feelers are bound sticking straight out behind his head: hanging loose, each joint turns down
    from its cut and carries its surface along; a corner the head shares stays put."""
    chain = np.array([[0., 0, 10], [0, 1, 10], [0, 2, 10], [0, 3, 10]])
    points = np.array([[0., .05, 10.1], [0, 1.5, 10.1], [0, 2.9, 9.9]])
    skins = [[(2, 1.)]] + hang.skin(points[1:], [chain])
    (hung,), (propped,), moved, turned = hang.drape([chain], points, skins, _nothing)
    assert np.allclose(hung, [[0, 0, 10], [0, 0, 9], [0, 0, 8], [0, 0, 7]], atol=1e-6)
    assert propped == [False, False, False]
    assert np.allclose(moved[0], points[0])                                    # held: on the static root
    assert np.allclose(moved[2], [0, -.1, 7.1], atol=1e-6)                     # 0.1 above it, now 0.1 to its front
    assert np.allclose(turned[2] @ [0, 1, 0], [0, 0, -1], atol=1e-6)


def test_a_drape_stops_where_the_body_would_take_the_next_step():
    """Each joint turns toward straight down only as far as the surface it carries stays out of the rest of the
    costume: here a back that a chain bound out from a height of 10 meets at 9.5, so its far end rests there, and
    the body props up both its segments."""
    chain = np.array([[0., 0, 10], [0, 1, 10], [0, 2, 10]])
    points = np.array([[0., 2, 10]])
    (hung,), (propped,), moved, _ = hang.drape([chain], points, hang.skin(points, [chain]), _below(9.5))
    assert 9.5 <= moved[0][2] < 9.5 + 2 * np.sin(hang.STEP)
    assert hung[-1][2] >= 9.5 and hang.off_down(hung) > 45
    assert propped == [True, True]


def test_a_part_drawn_as_it_rests_is_only_tested_for_what_props_it():
    """A part the pose does not hold out (a lock swept across the face, a skirt flared) keeps its shape: nothing
    turns, and a segment the body would stop a step toward down is propped."""
    chain = np.array([[0., 0, 10], [0, 1, 10], [0, 2, 10]])
    points = np.array([[0., 2, 10]])
    (hung,), (propped,), moved, _ = hang.drape([chain], points, hang.skin(points, [chain]), _below(9.99),
                                               turn=False)
    assert np.allclose(hung, chain) and np.allclose(moved, points) and propped == [True, True]
    (_, (free,), _, _) = hang.drape([chain], points, hang.skin(points, [chain]), _nothing, turn=False)
    assert free == [False, False]


def test_a_chain_already_hanging_is_left_as_it_is():
    chain = np.array([[0., 0, 10], [0, 0, 9], [0, 0, 8]])
    points = np.array([[.1, 0, 9.5]])
    (hung,), (propped,), moved, _ = hang.drape([chain], points, hang.skin(points, [chain]), _nothing)
    assert np.allclose(hung, chain) and np.allclose(moved, points) and propped == [False, False]


def test_surfaces_sharing_a_corner_are_one_and_one_sewn_on_swings_from_where_it_comes_free():
    flap = [np.array([[0., 0, 3], [1, 0, 3], [0, 0, 1]])]
    beside = [np.array([[1., 0, 3], [2, 0, 3], [1, 0, 1]])]
    chains = [[[0., 0, 4], [0, 0, 0]], [[1.5, 0, 4], [1.5, 0, 0]]]
    (one,) = hang.surfaces(chains, ['JOBJ_4'] * 2, [flap, beside], np.array([[0., 0, 3], [1, 0, 3], [2, 0, 3]]))
    assert one['ids'] == [0, 1] and one['why'] is None and np.allclose(one['cut'], [1, 1])
    # sewn on down to z 1, three quarters of it: cloth on the body is no part of it, and swings from there
    (sewn,) = hang.surfaces(chains[:1], ['JOBJ_4'], [flap], np.array([[0., 0, 3], [0, 0, 1]]))
    assert sewn['why'] is None and np.allclose(sewn['cut'], [3])
    # sewn on to its far end: nothing of it hangs free
    hem = [np.array([[0., 0, 3], [1, 0, 3], [0, 0, 0]])]
    (joined,) = hang.surfaces(chains[:1], ['JOBJ_4'], [hem], np.array([[0., 0, 3], [0, 0, 0]]))
    assert joined['why'] == hang.JOINED
    (spans,) = hang.surfaces(chains, ['JOBJ_4', 'JOBJ_5'], [flap, beside], np.zeros((0, 3)))
    assert spans['why'] == hang.SPANS
    (loop,) = hang.surfaces([[[0., 0, 4], [0, -1, 3.5], [0, 0, 4]]], ['JOBJ_4'], [flap], np.zeros((0, 3)))
    assert loop['why'] == hang.LOOP                                              # out and back: no way to hang


def test_a_cut_chain_starts_where_it_is_cut():
    chain = np.array([[0., 0, 4], [0, 0, 2], [0, 0, 0]])
    assert np.allclose(hang.cut_from(chain, 1), [[0, 0, 3], [0, 0, 2], [0, 0, 0]])
    assert np.allclose(hang.cut_from(chain, 2), [[0, 0, 2], [0, 0, 0]])
    assert np.allclose(hang.cut_from(chain, 0), chain)
