from types import SimpleNamespace as NS

import numpy as np
import pytest

from rigswap.plugins.melee.prep.weights import PIECE
from rigswap.plugins.melee.prep.sidedness import apply, backed, boundary_counts, covered


POINTS = np.array([(0., 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
CLOSED = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
# an open sheet beside the closed volume (one lying on its face would have its back inside it: covered)
APART = np.vstack([POINTS, POINTS + (5, 0, 0)])
SHEET = CLOSED[0] + 4


def test_open_sheet_and_closed_volume_with_split_seam_vertices():
    assert boundary_counts(POINTS, CLOSED, [1] * 4) == {1: 0}
    assert boundary_counts(POINTS, CLOSED[:1], [1]) == {1: 3}
    # UV/normal seams may give every triangle its own three vertices.
    corners = POINTS[CLOSED].reshape(-1, 3)
    assert boundary_counts(corners, np.arange(12).reshape(-1, 3), [1] * 4) == {1: 0}
    # Coincident pieces must not close each other's boundary.
    assert boundary_counts(POINTS, [CLOSED[0], CLOSED[0][::-1]], [1, 2]) == {1: 3, 2: 3}
    assert boundary_counts(POINTS, [[0, 0, 1]], [1]) == {1: 0}


class Material:
    def __init__(self, name, cull):
        self.name, self.use_backface_culling = name, cull

    def copy(self):
        return Material(self.name + '.copy', self.use_backface_culling)


def mesh(name, points, faces, pieces, material):
    me = NS(users=1, vertices=[NS(co=p) for p in points],
            polygons=[NS(index=i, vertices=f, material_index=0) for i, f in enumerate(faces)],
            materials=[material], attributes={PIECE: NS(data=[NS(value=p) for p in pieces])})
    return NS(name=name, data=me, matrix_world=np.eye(3), material_slots=[NS(material=material)])


@pytest.mark.parametrize('source_culls', [False, True])
def test_shared_material_splits_modes_without_changing_geometry(source_culls):
    material = Material('shared', source_culls)
    ob = mesh('surface', APART, list(CLOSED) + [SHEET], [1] * 4 + [2], material)
    record = apply([ob])
    assert record['pieces'] == [dict(kind='piece', id=1, boundary_edges=0, two_sided=False),
                                dict(kind='piece', id=2, boundary_edges=3, two_sided=True)]
    assert len(record['material_splits']) == 1
    assert len(ob.data.polygons) == 5 and len(ob.data.vertices) == 8
    assert all(ob.data.materials[p.material_index].use_backface_culling for p in ob.data.polygons[:4])
    assert not ob.data.materials[ob.data.polygons[4].material_index].use_backface_culling


def test_one_piece_across_objects_and_materials_is_still_closed():
    a = mesh('a', POINTS, CLOSED[:2], [1, 1], Material('a', False))
    b = mesh('b', POINTS, CLOSED[2:], [1, 1], Material('b', False))
    assert apply([a, b])['pieces'] == [dict(kind='piece', id=1, boundary_edges=0, two_sided=False)]
    assert a.data.materials[0].use_backface_culling and b.data.materials[0].use_backface_culling


@pytest.mark.parametrize('opened', [False, True])
def test_an_object_drawn_whole_takes_one_mode(opened):
    """A detail's object (the eyes) is one DObj: one open piece makes all of it two-sided, with no variant."""
    material = Material('eye', True)
    faces, pieces = (list(CLOSED) + [SHEET], [1] * 4 + [2]) if opened else (list(CLOSED), [1] * 4)
    ob = mesh('eye', APART, faces, pieces, material)
    other = mesh('body', APART + (0, 9, 0), list(CLOSED) + [SHEET], [3] * 4 + [4], Material('body', True))
    record = apply([ob, other], whole=['eye'])
    assert record['whole_objects'] == {'eye': opened}
    assert ob.data.materials == [material] and material.use_backface_culling is not opened
    assert {p.material_index for p in ob.data.polygons} == {0} and len(ob.data.polygons) == len(faces)
    assert [s['source'] for s in record['material_splits']] == ['body']


def test_an_open_sheet_with_its_own_reversed_lining_culls():
    """A sheet alone draws both sides; one with a reversed copy just behind it (a lining) culls, so the two never
    fight at one depth."""
    sheet = np.array([(0., 0, 0), (1, 0, 0), (0, 1, 0)])
    lining = sheet + (0, 0, -0.001)
    points = np.vstack([sheet, lining, [(5., 5, 5)]])
    faces = [np.array([0, 1, 2]), np.array([5, 4, 3])]
    assert backed(points, faces).tolist() == [True, True]
    assert backed(points, faces[:1]).tolist() == [False]
    assert backed(points, [np.array([0, 1, 2]), np.array([3, 4, 5])]).tolist() == [False, False]   # same way: a layer
    material = Material('gown', False)
    ob = mesh('gown', points, faces, [1, 2], material)
    apply([ob])
    assert all(ob.data.materials[p.material_index].use_backface_culling for p in ob.data.polygons)
    alone = mesh('sheet', sheet, faces[:1], [1], Material('sheet', True))
    apply([alone])
    assert not alone.data.materials[0].use_backface_culling


def test_a_face_whose_back_another_piece_hides_culls():
    """tlink's shield: a dome facing out and a flat plate facing the other way, two open meshes apart. Each one's
    back is the other's front: both cull. A piece's own far side (a sleeve) or a surface facing the same way (the
    body under a shirt) hides nothing."""
    square = np.array([(-1., -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    points = np.vstack([square, square * (0.8, 0.8, 1) + (0, 0, 0.3)])
    plate, dome = np.array([3, 2, 1, 0]), np.array([4, 5, 6, 7])            # facing down, facing up
    assert covered(points, [plate, dome], [1, 2]).tolist() == [True, True]
    assert covered(points, [plate, dome], [1, 1]).tolist() == [False, False]   # one piece: its own far side
    assert covered(points, [plate[::-1], dome], [1, 2]).tolist() == [False, False]  # both facing up: layers
    assert covered(points, [dome], [2]).tolist() == [False]
    material = Material('shield', False)
    ob = mesh('shield', points, [plate, dome], [1, 2], material)
    record = apply([ob])
    assert record['faces_hidden_behind'] == 2
    assert all(ob.data.materials[p.material_index].use_backface_culling for p in ob.data.polygons)
