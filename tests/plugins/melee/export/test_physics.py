"""The kind table, independent accessory skeletons, native definitions and DAT graft."""
import json
from math import radians
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from rigswap.core import hang
from rigswap.plugins.melee.dat import Dat, dat_joints
from rigswap.plugins.melee.export import physics
from rigswap.plugins.melee.export.write import HSDRAW_DEFAULT, export_dae
from rigswap.plugins.melee.export.checks import VERTEX_TOLERANCE
from rigswap.plugins.melee.dae import read_dae
from rigswap.plugins.melee.library import PHYSICS, dynamics
from tests.plugins.melee.test_dat import _DatBuilder


def test_kind_table_has_provenance_and_engine_units():
    assert set(PHYSICS) == {'hair', 'cape', 'tail', 'other'}
    hair = dynamics('hair', 4)
    assert hair['bone'] == 3
    assert [hair[k] for k in ('PARAM1', 'PARAM2', 'PARAM3')] == [1., 1., .07]
    assert hair['joints'][0] == [1., 0., 0., 0., 0., 0., radians(45), *([radians(360)] * 3),
                                 *([-radians(360)] * 3), radians(1), radians(6)]
    assert dynamics('tail', 1)['joints'][0][7] == radians(180)
    assert dynamics('ear', 23) == dynamics('other', 23)
    assert len(dynamics('cape', 23)['joints']) == 24
    assert dynamics('cape', 23)['joints'][-2] == dynamics('cape', 4)['joints'][-2]
    assert all('Pl' in p['source'] for p in PHYSICS.values())
    with pytest.raises(ValueError):
        dynamics('hair', 0)


def test_game_limit_counts_descriptors_not_joints(tmp_path):
    from rigswap.plugins.melee.targets.physics import record
    from rigswap.plugins.melee.library import MAX_DYNAMICS
    from types import SimpleNamespace
    b = _DatBuilder()
    root = b.add(bytes(0x30))
    group = b.add(struct.pack('>II', 2, 0))
    entries = b.add(struct.pack('>6I', 5, 0, 22, 0, 0, 0) + struct.pack('>6I', 40, 0, 23, 0, 0, 0))
    b.point(root + 0x2C, group)
    b.point(group + 4, entries)
    b.save(tmp_path / 'PlXX.dat', [('ftDataTest', root)])
    _donor(tmp_path / 'PlXXNr.dat')
    got = record(SimpleNamespace(code='XX'), tmp_path)
    assert MAX_DYNAMICS == 9
    assert got['existing_chains'] == 3 and got['moving_joints'] == 45
    assert got['fighter_chains'] == 2 and got['costume_chains'] == 1


def _donor(path, kind='hair', anchor=5):
    b = _DatBuilder()
    root = b.add(bytes(16))
    array = b.add(bytes(4))
    acc = b.add(bytes(44))
    joint = b.add(bytes(64))
    desc_array = b.add(bytes(4))
    desc = b.add(bytes(24))
    definition = dynamics(kind, 3)
    params = b.add(struct.pack('>60f', *(v for r in definition['joints'] for v in r)))
    struct.pack_into('>I', b.data, root + 8, 1)
    b.point(root + 12, array)
    b.point(array, acc)
    b.point(acc, joint)
    struct.pack_into('>II', b.data, acc + 4, anchor, 1)
    b.point(acc + 12, desc_array)
    struct.pack_into('>I', b.data, acc + 16, 1)
    b.point(desc_array, desc)
    struct.pack_into('>I', b.data, desc, 3)
    b.point(desc + 4, params)
    struct.pack_into('>I3f', b.data, desc + 8, 4, 1., 1., definition['PARAM3'])
    return b.save(path, [('mexCostume', root)])


def test_graft_preserves_original_bytes_roots_and_all_chains(tmp_path):
    b = _DatBuilder()
    b.add(bytes(range(128)))
    body = b.save(tmp_path / 'body.dat', [('body', 0)])
    donor = _donor(tmp_path / 'donor.dat')
    out = tmp_path / 'out.dat'
    assert physics.merge(body, [donor] * 17, out) == 0
    d = Dat.read(out)
    assert d.data[:128] == bytes(range(128))
    assert d.root('body') == 0
    assert len(physics.definitions(out)) == 17  # no allocation to larger chains or demotion
    for r in physics.definitions(out):
        assert r['attach_bone'] == 5
        assert r['dynamics'][0]['bone'] == 3
        assert r['dynamics'][0]['PARAM3'] == pytest.approx(.07)
        assert r['dynamics'][0]['joints'] == pytest.approx(np.array(dynamics('hair', 3)['joints']))
    again = tmp_path / 'again.dat'
    assert physics.merge(out, [donor], again) == 17
    assert len(physics.definitions(again)) == 18


def _row():
    return dict(kind='hair', anchor='JOBJ_5', source_bones=['strand_a', 'strand_b'],
                points=[[0., 0, 2], [0, 0, 1], [0, 0, 0]])


def _triangles():
    return [('cloth', np.array([[0., 0, 1.5], [.3, 0, .5], [-.3, 0, .5]]),
             np.array([[0., -1, 0]] * 3), [(0, 1), (1, 0), (0, 0)])]


def test_authoring_files_identify_local_bones_and_attachment_frame(tmp_path):
    row = _row()
    world = np.eye(4)
    world[:3, 3] = [1, 2, 3]
    smd, dyn = physics.write_accessory(row, _triangles(), world, tmp_path / 'chain')
    assert row['local_bones'] == [3, 4]
    assert '3 "JOBJ_3" 2' in smd.read_text()
    assert '5 "JOBJ_5" 4' in smd.read_text()  # terminal tip, not another moving joint
    assert json.loads(dyn.read_text()) == [dynamics('hair', 2)]
    assert '2 -1 0 -3 0 0 0' in smd.read_text()  # z-up -> y-up, then inverse anchor
    assert row['source_bones'] == ['strand_a', 'strand_b']


@pytest.mark.parametrize('start,far', [
    ([0., 0, 1.6], [.3, 0, .5]),                             # a tail on its head: out to the surface's far point
    ([0., 0, .2], [0., 0, 1.5]),                             # ... from below the surface: its far point is above
])
def test_a_chain_ending_in_a_tail_on_its_head_reaches_its_surfaces_farthest_point(start, far):
    """The chain write_accessory skins an accessory on (chain_of): an importer's terminal tail put on its last head
    has no length, so the chain reaches out to the farthest point of its surface from its start."""
    chain = physics.chain_of(dict(_row(), points=[start, start]), _triangles())
    assert np.allclose(chain[-1], np.asarray(far) @ physics.TO_GAME.T)


@pytest.mark.parametrize('kind', ['hair', 'cape', 'tail', 'other'])
@pytest.mark.parametrize('count', [1, 2, 5])
def test_declared_chain_gives_every_moving_joint_a_successor(tmp_path, kind, count):
    row = dict(_row(), kind=kind, points=[[0., 0, 2. - i] for i in range(count + 1)])
    smd, dyn = physics.write_accessory(row, _triangles(), np.eye(4), tmp_path / 'chain')
    lines = smd.read_text().splitlines()
    nodes = lines[lines.index('nodes') + 1:lines.index('end')]
    children = {int(line.split()[-1]): int(line.split()[0]) for line in nodes}
    definition, = json.loads(dyn.read_text())
    # lb_8000FD48 follows children for exactly the serialized parameter count.
    initialized, joint = [], definition['bone']
    for _ in definition['joints']:
        initialized.append(joint)
        joint = children.get(joint)
    # lb_8001044C updates only entries with a next initialized entry.
    assert initialized[:-1] == row['local_bones']
    assert initialized[-1] == count + 3 and joint is None
    assert len(definition['joints'][-1]) == 15


def _tri(*corners):
    return ('cloth', np.array(corners, float), np.array([[0., -1, 0]] * 3), [(0, 0)] * 3, [{}] * 3)


def _loose(anchor='JOBJ_4', top=4., bottom=0.):
    return dict(kind='other', source_bones=['flap'], anchor=anchor, points=[[0., 0, top], [0, 0, bottom]], pieces=[])


def test_loose_chains_sharing_a_surface_swing_together_from_where_it_leaves_the_body():
    """Two flaps whose surfaces share a corner are one accessory of two chains. Each swings from where its surface
    last meets the rest of the costume: its chain is cut there and the shared corners stay on the anchor."""
    left = [_tri([0, 0, 3], [1, 0, 3], [0, 0, 1]), _tri([1, 0, 3], [1, 0, 1], [0, 0, 1])]
    right = [_tri([1, 0, 3], [2, 0, 3], [1, 0, 1])]
    rows = [_loose(), dict(_loose(), points=[[1.5, 0, 4], [1.5, 0, 0]])]
    (row, tris, why, ids), = physics.loose_groups(rows, [left, right], np.array([[0., 0, 3], [1, 0, 3], [2, 0, 3]]))
    assert why is None and ids == [0, 1]
    assert [c[0] for c in row['chains']] == [[0, 0, 3], [1.5, 0, 3]]          # cut where the belt holds it
    assert [t[5] for t in tris] == [[True, True, False], [True, False, False], [True, True, False]]


@pytest.mark.parametrize('rest,why,cut', [
    ([[0., 0, 3], [0, 0, 1]], None, [0, 0, 1]),             # sewn on down to z 1: it swings from there on
    ([[0., 0, 3], [0, 0, 0]], hang.JOINED, None),            # sewn on to its far end: nothing of it hangs free
    ([], None, [0, 0, 4]),                                   # an earring: no surface shared, it swings whole
])
def test_a_loose_chain_sewn_on_along_its_length_swings_from_where_it_comes_free(rest, why, cut):
    """Cloth resting on the body or sewn to it is no part of it: it swings from where its surface last meets the rest
    of the costume, however far along that is, and only one joined to the end has nothing to swing."""
    tris = [_tri([0, 0, 3], [1, 0, 3], [0, 0, 1]), _tri([0, 0, 1], [1, 0, 3], [0, 0, 0])]
    (row, _, said, _), = physics.loose_groups([_loose()], [tris], np.array(rest).reshape(-1, 3))
    assert said == why
    if cut is not None:
        assert np.allclose(row['chains'][0][0], cut)


def test_two_bones_cannot_share_an_accessory():
    tris = [_tri([0, 0, 3], [1, 0, 3], [0, 0, 1])]
    (_, _, why, _), = physics.loose_groups([_loose(), dict(_loose('JOBJ_5'), points=[[1., 0, 4], [1, 0, 0]])],
                                           [tris, [_tri([1, 0, 3], [2, 0, 3], [1, 0, 1])]], np.zeros((0, 3)))
    assert why == 'its surface spans chains hanging from different bones'


def test_an_accessory_of_two_chains_hangs_each_off_the_static_root(tmp_path):
    row = dict(_row(), chains=[[[0., 0, 2], [0, 0, 1], [0, 0, 0]], [[1., 0, 2], [1, 0, 0]]])
    held = [('cloth', *_triangles()[0][1:], [{}] * 3, [True, False, False])]
    smd, dyn = physics.write_accessory(row, held, np.eye(4), tmp_path / 'chain')
    lines = smd.read_text().splitlines()
    nodes = lines[lines.index('nodes') + 1:lines.index('end')]
    assert [int(n.split()[-1]) for n in nodes[:8]] == [-1, 0, 1, 2, 3, 4, 2, 6]
    assert [d['bone'] for d in json.loads(dyn.read_text())] == [3, 6]
    assert [len(d['joints']) for d in json.loads(dyn.read_text())] == [3, 2]
    assert row['local_bones'] == [3, 4, 6]
    # a held corner stays on the root (the triangle, reaching a joint and a half along its chain, is cut: jointed)
    corners = [ln.split() for ln in lines[lines.index('triangles') + 1:] if len(ln.split()) > 9]
    held_at = [c for c in corners if np.allclose([float(v) for v in c[1:4]], [0, 1.5, 0])]
    assert held_at and all(c[-3:] == ['1', '2', '1'] for c in held_at)


def test_hair_or_cloth_swings_wherever_it_rests():
    """A loose chain is hair or cloth (labels.chains), and hangs because of what it is: resting off the vertical
    against the body, it swings, a spring holding it there. The build's own reading of its surface (loose.json's
    swings) is no gate: prep's final surface decides."""
    tris = [_tri([0, 0, 3], [1, 0, 3], [0, 0, 1])]
    rows = [dict(_loose(), kind='cloth', points=[[0., 0, 4], [3, 0, 2]], swings=False, why='it carries no face')]
    (row, _, why, _), = physics.loose_groups(rows, [tris], np.zeros((0, 3)))
    assert why is None and row['chains'] == [[[0, 0, 4], [3, 0, 2]]]


def test_a_spring_holds_each_joint_where_it_rests_against_gravity(tmp_path):
    """Gravity turns a link toward straight down by PARAM3 over its length in the game's units, times the sine of
    its angle off down, each frame (lb_80011710, lb_8001044C); the spring turns it back as far."""
    from rigswap.plugins.melee.library import PHYSICS
    chain = np.array([[0., 4, 0], [0, 2, 0], [2, 2, 0], [2 + 3 ** .5, 1, 0]])     # down, level, 60 degrees off down
    got = physics.springs('hair', chain, scale=2.)
    assert np.allclose(got, [0, PHYSICS['hair']['gravity'] / 4, PHYSICS['hair']['gravity'] / 4 * 3 ** .5 / 2])
    # propped up by the body: held whichever way the body turns, so the whole of gravity's turn
    propped = physics.springs('hair', chain, scale=2., propped=[False, True, True])
    assert np.allclose(propped, [0, PHYSICS['hair']['gravity'] / 4, PHYSICS['hair']['gravity'] / 4])
    assert physics.springs('hair', chain, 2., propped=[True]) == got        # a record for another cut: unused
    # a link resting above the horizontal (150 degrees off down) balances only as an inverted pendulum: held whole
    assert np.allclose(physics.springs('hair', np.array([[0., 0, 0], [0, 3 ** .5, 1]]), 2.),
                       [PHYSICS['hair']['gravity'] / 4])
    row = dict(_row(), kind='hair', attachment_from=physics.LOOSE, points=[[0., 0, 2], [0, 1, 2], [0, 1, 0]])
    _, dyn = physics.write_accessory(row, _triangles(), np.eye(4), tmp_path / 'loose', scale=2.)
    written = [j[1] for j in json.loads(dyn.read_text())[0]['joints'][:-1]]
    assert np.allclose(written, physics.springs('hair', physics.chains_of(row, _triangles())[0], 2.))
    assert written[0] > 0 and written[1] == 0
    old = physics.write_accessory(_row(), _triangles(), np.eye(4), tmp_path / 'old')[1]
    assert all(j[1] == 0 for j in json.loads(old.read_text())[0]['joints'])     # the old middle's: no spring


def test_mirror_twins_are_found_and_kept_together_biggest_first():
    from rigswap.plugins.melee.export.physics_choice import biggest
    rows = [dict(points=[[1., 1, 10], [1, 1, 6]], area=3.),            # a left feeler
            dict(points=[[-1., 1, 10], [-1, 1, 6]], area=3.1),         # and the right one
            dict(points=[[0., 2, 9], [0, 2, 5]], area=5., chains=[[[0., 2, 9], [0, 2, 5]], [[0., 3, 9], [0, 3, 5]]]),
            dict(points=[[2., 0, 4], [2, 0, 1]], area=1.)]             # a strap on one side
    physics.twins(rows)
    assert [r.get('twin') for r in rows] == [1, 0, None, None]
    assert biggest(rows, 3) == [0, 1, 3]       # the pair outweighs the cape; the cape's two chains don't fit after it
    assert biggest(rows, 1) == [3]             # no room for a twin without its twin, nor for the cape


def test_a_triangle_reaching_past_a_joint_of_its_chain_is_cut_until_none_does():
    """Diddy's tail on Fox (tail-1, tail-2): 92 of its 1057 triangles reached past a joint (35 past two) along the
    chain it swings on, the longest from its root to its middle; flat, they stood across the tail's curl as a second
    strand beside it. Such a triangle is cut along the chain until every piece lies within a joint of itself; the
    pieces keep its winding and its area, and one within a joint is left as it is."""
    import numpy as np
    from rigswap.plugins.melee.export import physics
    chain = np.array([[0., 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]])                # the game's frame, along x
    build = lambda pts: np.asarray(pts, float) @ physics.TO_GAME                   # the same points in the build's
    def place(tri):
        return [sum(b * w for b, w in s) for s in physics._weights(np.asarray(tri[1]) @ physics.TO_GAME.T, [chain])]
    def area(tri):
        p = np.asarray(tri[1])
        return np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 2
    normals, uv = np.tile([0., 0, 1], (3, 1)), np.array([[0., 0], [1, 0], [0, 1]])
    weights = [{'a': 1.}, {'b': 1.}, {'a': 1.}]
    long = ('fur', build([[.5, .1, 0], [5.5, .1, 0], [.5, -.1, 0]]), normals, uv, weights, [False, False, False])
    short = ('fur', build([[2.2, .1, 0], [2.9, .1, 0], [2.2, -.1, 0]]), normals, uv, weights, [False, False, False])
    out = physics.jointed([long, short], [chain])
    assert out[-1] is short and len(out) > 3
    pieces = out[:-1]
    assert all(max(place(t)) - min(place(t)) <= 1 + 1e-6 for t in pieces)
    assert np.isclose(sum(area(t) for t in pieces), area(long))
    turn = lambda t: np.cross(t[1][1] - t[1][0], t[1][2] - t[1][0])
    assert all(turn(t) @ turn(long) > 0 for t in pieces)                           # the winding kept
    assert all(len(t) == 6 and len(t[4]) == 3 and len(t[5]) == 3 for t in pieces)
    assert any(t[4][i] == {'a': .5, 'b': .5} for t in pieces for i in range(3))    # a new corner, between its ends
    # two triangles sharing the long edge are cut at the same points along it: no gap opens between them
    twin = ('fur', build([[5.5, .1, 0], [.5, .1, 0], [3, .4, 0]]), normals, uv, weights, [False, False, False])

    def on_edge(tris):          # the corners lying on the edge the two share, from (.5, .1, 0) to (5.5, .1, 0)
        game = [np.asarray(p) @ physics.TO_GAME.T for t in tris for p in t[1]]
        return {round(float(p[0]), 6) for p in game if abs(p[1] - .1) < 1e-9 and abs(p[2]) < 1e-9}
    mine, theirs = on_edge(physics.jointed([long], [chain])), on_edge(physics.jointed([twin], [chain]))
    assert mine == theirs and len(mine) > 2
    # a corner the costume holds counts where it lies, and a new corner is held only between two held ones
    held = ('fur', build([[.5, .1, 0], [5.5, .1, 0], [.5, -.1, 0]]), normals, uv, weights, [True, False, True])
    cut = physics.jointed([held], [chain])
    assert len(cut) > 1 and any(t[5] == [True, True, True] or t[5].count(True) == 2 for t in cut)
    assert not any(t[5][i] and abs((np.asarray(t[1][i]) @ physics.TO_GAME.T)[0] - .5) > 1e-9 for t in cut for i in range(3))


def test_a_corner_the_costume_holds_is_counted_where_it_lies_along_the_chain():
    """Goku's sash on Mario (2026-11-10): 112 of its 156 corners are held still by the costume, some far down its
    short chain. Counted at the chain's start, each kept every piece between it and its swinging neighbours reaching
    as far as before however often it was halved: 52 triangles became 845. A held corner counts where it lies, so a
    triangle that lies within a joint is left as it is whichever of its corners are held."""
    import numpy as np
    from rigswap.plugins.melee.export import physics
    chain = np.array([[0., 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]])                # the game's frame, along x
    build = lambda pts: np.asarray(pts, float) @ physics.TO_GAME                   # the same points in the build's
    normals, uv = np.tile([0., 0, 1], (3, 1)), np.array([[0., 0], [1, 0], [0, 1]])
    weights = [{'a': 1.}, {'b': 1.}, {'a': 1.}]
    far = ('sash', build([[4.5, .1, 0], [5.2, .1, 0], [4.5, -.1, 0]]), normals, uv, weights, [True, False, False])
    assert physics.jointed([far], [chain]) == [far]


def test_an_edge_shorter_than_the_chains_shortest_joint_is_never_halved():
    """Where a chain doubles back, a point between its two stretches is as near the one as the other, and a small
    triangle there has corners joints apart along the chain though it is shorter than any joint: the skin jumps there
    and no cut mends a jump. It is left whole; halved, it would only make more of itself."""
    import numpy as np
    from rigswap.plugins.melee.export import physics
    chain = np.array([[0., 0, 0], [4, 0, 0], [4, 1, 0], [0, 1, 0]])                # out, across and back, a unit apart
    build = lambda pts: np.asarray(pts, float) @ physics.TO_GAME
    normals, uv = np.tile([0., 0, 1], (3, 1)), np.array([[0., 0], [1, 0], [0, 1]])
    weights = [{'a': 1.}, {'b': 1.}, {'a': 1.}]
    between = ('fur', build([[1, .4, 0], [1, .6, 0], [1.2, .4, 0]]), normals, uv, weights, [False, False, False])
    places = [i + t for _, i, t, _ in physics.hang.nearest(np.asarray(between[1]) @ physics.TO_GAME.T, [chain])]
    assert max(places) - min(places) > 2                                           # its corners lie joints apart
    assert physics.jointed([between], [chain]) == [between]
