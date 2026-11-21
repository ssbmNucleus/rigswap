"""Capacity decisions with small mesh/material doubles; no Blender process or game assets."""
from types import SimpleNamespace as NS

import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee.export.dobjs import FighterTables, Slot
from rigswap.plugins.melee.prep import atlas, join as prep_join


class Material:
    def __init__(self, name, culled, roughness=0.5):
        self.name, self.use_backface_culling, self.roughness = name, culled, roughness
        self.node_tree = NS(nodes=[], links=[], animation_data=None)

    def get(self, key):
        return None


class Mesh:
    def __init__(self, name, materials, joint=0):
        self.name = name
        self.material_slots = [NS(material=m) for m in materials]
        self.vertex_groups = [NS(name=f'JOBJ_{joint}')]
        self.data = NS(polygons=[NS(material_index=i) for i in range(len(materials))],
                       vertices=[NS(groups=[NS(group=0, weight=1)])])


@pytest.fixture
def scene(monkeypatch):
    meshes, atlases = [], []

    def merge_images(*args):
        atlases.append(True)
        return None

    monkeypatch.setattr(atlas, 'merge_images', merge_images)
    return meshes, atlases


@pytest.mark.parametrize('textured', [False, True])
def test_a_culled_and_a_two_sided_material_are_one_look_and_each_keeps_its_culling(scene, textured):
    """u-roy (ns-8): merging his cape's culled lining into the two-sided cape drew the lining over the cape up close.
    A PObj holds its own culling, so the two take one place and neither changes."""
    meshes, atlases = scene
    closed, opened = Material('closed', True), Material('open', False)
    if textured:
        image = type('Image', (), {'source': 'MOVIE'})()
        for m in (closed, opened):
            m.node_tree.nodes.append(NS(type='TEX_IMAGE', inputs=[], image=image))
    ob = Mesh('surface', [closed, opened])
    meshes.append(ob)
    record = prep_join.join_for_places(meshes, [Slot(0, 0)], FighterTables(body=(0,)))
    assert not atlases and [s.material for s in ob.material_slots] == [closed, opened]
    assert closed.use_backface_culling and not opened.use_backface_culling
    assert record['dobjs'] == 2 and record['looks'] == {'before': 1, 'after': 1}
    assert record['one_place'] == {'open': 'closed'} and not record['material_aliases']


def test_split_body_and_two_kept_eyes_fit_one_ordinary_place(scene):
    meshes, atlases = scene
    closed, opened, eye = Material('closed', True), Material('open', False), Material('eye', False, 0.2)
    meshes.extend([Mesh('body', [opened, closed]), Mesh('face', [opened]), Mesh('cloth', [opened]),
                   Mesh('eye_left', [eye]), Mesh('eye_right', [eye])])
    eyes = ['eye_left', 'eye_right']
    record = prep_join.join_for_places(meshes, [Slot(0, 0)], FighterTables(), keep=eyes, deferred=eyes)
    assert record['looks'] == {'before': 1, 'after': 1} and record['dobjs'] == 6 and not atlases
    assert record['kept_apart'] == eyes


@pytest.mark.parametrize('difference', ['roughness', 'format', 'graph', 'animation', 'image', 'kind'])
def test_only_culling_may_differ(scene, difference):
    meshes, _ = scene
    closed, opened = Material('closed', True), Material('open', False)
    formats = {}
    if difference == 'roughness':
        opened.roughness = 0.9
    elif difference == 'format':
        formats = {'closed': 'CMPR', 'open': 'RGB5A3'}
    elif difference == 'graph':
        for m in (closed, opened):
            m.node_tree.nodes.append(NS(type='GROUP'))
    elif difference == 'animation':
        closed.node_tree.animation_data = object()
    elif difference == 'image':
        image = type('Image', (), {'source': 'MOVIE'})
        for m in (closed, opened):
            m.node_tree.nodes.append(NS(type='TEX_IMAGE', inputs=[], image=image()))
    else:
        opened.get = lambda key: 'other'
    ob = Mesh('surface', [closed, opened])
    meshes.append(ob)
    with pytest.raises(Failed, match='2 DObjs.*1 places'):
        prep_join.join_for_places(meshes, [Slot(0, 0)], FighterTables(), formats=formats)
    assert [s.material for s in ob.material_slots] == [closed, opened]


def test_each_joint_counts_its_own_looks_and_deferred_objects_none(scene):
    meshes, _ = scene
    closed, opened, other = Material('closed', True), Material('open', False), Material('other', True, 0.1)
    crowded = Mesh('crowded', [closed, opened])
    roomy = Mesh('roomy_SINGLE', [closed, other], joint=1)
    deferred = Mesh('deferred', [other])
    meshes.extend([crowded, roomy, deferred])
    record = prep_join.join_for_places(meshes, [Slot(0, 0), Slot(1, 0), Slot(1, 1)], FighterTables(),
                                       deferred=['deferred'], formats={'closed': 'CMPR', 'open': 'CMPR'})
    assert record['dobjs'] == 5 and record['looks']['after'] == 3
    assert [s.material for s in crowded.material_slots] == [closed, opened]
    assert not record['material_aliases']  # The original format name still has drawn users.


def test_looks_on_different_joints_are_not_one(scene):
    meshes, _ = scene
    meshes.extend([Mesh('closed_SINGLE', [Material('closed', True)], joint=1),
                   Mesh('open', [Material('open', False)])])
    with pytest.raises(Failed, match='joint 1'):
        prep_join.join_for_places(meshes, [Slot(0, 0), Slot(0, 1)], FighterTables())
