import numpy as np

from rigswap.core import onepiece
from rigswap.plugins.melee.prep.flat import pushed, stood


def _box(x, y, z):
    points = np.array([(a, b, c) for a in (-x, x) for b in (-y, y) for c in (0, z)], float)
    tris = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                     (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return points, tris


def test_a_sheet_facing_front_comes_round_to_the_side_with_as_much_depth_as_width():
    points, tris = _box(3, .1, 10)
    out, side, front, depth = stood(points, tris)
    assert front > side
    assert np.allclose(np.ptp(out, axis=0), (6, 6, 10))
    assert depth == 30
    # what the game shows of it, seen along X, is what the front of it was
    assert np.isclose(onepiece.outline(out, tris, 0), front, rtol=.01)


def test_what_faced_front_faces_the_way_he_is_seen_from_facing_right():
    points, tris = _box(3, .1, 10)
    points = np.vstack([points, (0, -.1, 5)])
    assert stood(points, tris)[0][-1, 0] < 0


def test_a_body_as_broad_from_the_side_keeps_its_place_and_its_depth():
    points, tris = _box(1, 2, 10)
    out, side, front, depth = stood(points, tris)
    assert side > front
    assert depth == 2
    assert np.allclose(out[:, 1:], points[:, 1:])


def test_a_hull_stands_off_by_the_line_and_stays_shut_at_a_seam():
    points, tris = _box(1, 1, 2)
    doubled = np.vstack([points, points[:1]])       # corner 0 again, for the last face that uses it
    faces = tris.copy()
    faces[1, 0] = 8
    out = pushed(doubled, faces, .3)
    assert np.allclose(np.linalg.norm(out - doubled, axis=1), .3)
    assert np.allclose(out[0], out[8])
    assert np.all(out[0] < doubled[0])
