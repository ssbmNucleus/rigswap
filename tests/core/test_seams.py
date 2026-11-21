from types import SimpleNamespace

import numpy as np

from rigswap.core import cloth, repose


def surface(points, edges, weights):
    return SimpleNamespace(points=np.asarray(points, float), edges=np.asarray(edges, int).reshape(-1, 2),
                           weights=np.asarray(weights, float))


def blend(rig, lengths=None):
    bones = list(range(rig.weights.shape[1]))
    lengths = lengths or [4] * len(bones)
    embedded = {b: (np.zeros(3), np.array([0, length, 0])) for b, length in zip(bones, lengths)}
    return repose.Surface(rig).blend(rig.weights, bones, embedded)


def test_a_hard_seam_blends_by_surface_distance_on_each_side_and_keeps_its_shape():
    x = [-2, -.5, -.1, .1, .5, 2]
    rig = surface([[v, 0, 0] for v in x], [[i, i + 1] for i in range(5)], [[1, 0]] * 3 + [[0, 1]] * 3)
    w, report = blend(rig)
    assert np.allclose(w[:, 1], [0, .25, .45, .55, .75, 1])
    assert report['vertices'] == 4
    assert np.allclose(w.sum(axis=1), 1) and w.shape == rig.weights.shape
    moves = {0: repose.Move(np.column_stack([np.eye(3), [-2, 0, 0]])),
             1: repose.Move(np.column_stack([np.eye(3), [2, 0, 0]]))}
    hard = repose.deform(rig.points, rig.weights, [0, 1], moves)
    bent = repose.deform(rig.points, w, [0, 1], moves)
    assert np.linalg.norm(bent[3] - bent[2]) < np.linalg.norm(hard[3] - hard[2])


def test_a_band_uses_each_bones_length_and_scales_with_the_model():
    rig = surface([[-.25, 0, 0], [.25, 0, 0]], [[0, 1]], [[1, 0], [0, 1]])
    w, _ = blend(rig, [2, 4])
    assert np.allclose(w, [[.75, .25], [.375, .625]])
    rig.points *= 13
    assert np.allclose(blend(rig, [26, 52])[0], w)


def test_a_uv_or_normal_split_has_one_skin_and_stays_closed_at_rest():
    rig = surface([[-.5, 0, 0], [0, 0, 0], [0, 0, 0], [.5, 0, 0]], [[0, 1], [2, 3]],
                  [[1, 0], [1, 0], [0, 1], [0, 1]])
    w, report = blend(rig)
    assert np.allclose(w[:, 1], [.25, .5, .5, .75]) and report['vertices'] == 4
    moves = {0: repose.Move(np.column_stack([np.eye(3), [0, 1, 0]])),
             1: repose.Move(np.column_stack([np.eye(3) * 2, [0, -1, 0]]))}
    placed = repose.deform(rig.points, w, [0, 1], moves)
    assert np.array_equal(placed[1], placed[2])


def test_disconnected_overlapping_pieces_never_share_weights():
    rig = surface([[-.2, 0, 0], [.2, 0, 0], [-.1, 0, 0], [.1, 0, 0]], [[0, 1], [2, 3]],
                  [[1, 0], [1, 0], [0, 1], [0, 1]])
    w, report = blend(rig)
    assert np.array_equal(w, rig.weights) and report['vertices'] == 0


def test_authored_blends_including_a_pure_endpoint_are_left_alone():
    rig = surface([[-.3, 0, 0], [-.1, 0, 0], [.1, 0, 0], [.3, 0, 0]], [[0, 1], [1, 2], [2, 3]],
                  [[1, 0], [.7, .3], [.3, .7], [0, 1]])
    assert np.array_equal(blend(rig)[0], rig.weights)


def test_distance_follows_a_folded_surface_and_does_not_cross_an_authored_blend():
    rig = surface([[-.1, 0, 0], [.1, 0, 0], [-.1, 2, 0], [-.15, 0, 0], [-.2, 0, 0], [-.3, 0, 0]],
                  [[0, 1], [0, 2], [2, 3], [0, 4], [4, 5]],
                  [[1, 0], [0, 1], [1, 0], [1, 0], [.9, .1], [1, 0]])
    w, _ = blend(rig)
    assert w[0, 1] > 0 and np.array_equal(w[2:], rig.weights[2:])


def test_a_multi_bone_junction_is_nonnegative_normalized_capped_and_welded():
    count = repose.MAX_INFLUENCES + 2
    rig = surface([[0, 0, 0]] * count + [[.1, i, 0] for i in range(count)],
                  [[i, i + count] for i in range(count)], np.vstack([np.eye(count), np.eye(count)]))
    w, _ = blend(rig)
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1)
    assert (np.count_nonzero(w, axis=1) <= repose.MAX_INFLUENCES).all()
    assert np.array_equal(w[:count], np.repeat(w[:1], count, axis=0))


def test_zero_length_bones_and_unweighted_vertices_are_finite():
    rig = surface([[-.1, 0, 0], [.1, 0, 0], [.2, 0, 0]], [[0, 1], [1, 2]], [[1, 0], [0, 1], [0, 0]])
    w, _ = blend(rig, [0, 0])
    assert np.array_equal(w, rig.weights) and np.isfinite(w).all()


def test_overlapping_bands_share_weight_at_a_junction_without_exceeding_the_cap():
    count = repose.MAX_INFLUENCES + 2
    rig = surface([[0, 0, 0]] + [[.01, .01 * i, 0] for i in range(1, count)],
                  [[0, i] for i in range(1, count)], np.eye(count))
    w, _ = blend(rig)
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1)
    assert (np.count_nonzero(w, axis=1) <= repose.MAX_INFLUENCES).all()
    assert np.count_nonzero(w[0]) > 1


def test_gown_game_and_held_columns_obey_the_same_surface_rule():
    rig = surface([[-.1, 0, 0], [.1, 0, 0]], [[0, 1]], [[1, 0], [0, 1]])
    g = cloth.Gown([0, 1], [], np.zeros(2), np.eye(2), [10, 11], held=np.eye(2), moves={10: None, 11: None})
    bones, owner = [10, 11], {0: 10, 1: 11}
    game = cloth.weights(rig, owner, bones, g)
    by, held, _ = cloth.rest(rig, owner, bones, g, {})
    embedded = {b: (np.zeros(3), np.array([0, 4, 0])) for b in by}
    graph = repose.Surface(rig)
    game, _ = graph.blend(game, bones, embedded)
    held, _ = graph.blend(held, by, embedded)
    assert game.shape == (2, 2) and held.shape == (2, 4)
    assert np.allclose(game, held[:, 2:]) and np.allclose(held.sum(axis=1), 1)
    assert (np.count_nonzero(held, axis=1) <= repose.MAX_INFLUENCES).all()


def test_split_and_surface_use_the_same_bone_length_band():
    from rigswap.core.correspond import Chain
    rig = surface([[x, 0, 0] for x in [1.5, 1.75, 2, 2.25, 2.5]], [], np.full((5, 2), .5))
    chain = Chain([0], np.array([[0, 0, 0.]]), np.array([4, 0, 0.]))
    w = repose.split(rig, np.tile([1., 0], (5, 1)), [10, 11], [(0, chain, [(.5, 11)])], {0: 10, 1: 10})
    assert np.allclose(w[:, 1], [0, .125, .25, .375, .5])
