from types import SimpleNamespace

import numpy as np

from rigswap.core import assemble


def _turn(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


BODY = {'names': ['root', 'hip', 'leg_L', 'leg_R', 'chest', 'head', 'arm_L', 'arm_R'],
        'heads': np.array([[0, 0, 0], [0, 0, 1], [0.2, 0, 0.9], [-0.2, 0, 0.9], [0, 0, 1.4], [0, 0, 1.8],
                           [0.3, 0, 1.5], [-0.3, 0, 1.5]], float)}


def _part(names, scale=40.0, turn=30.0, shift=(5, -2, 1), extra=()):
    """A part's skeleton as its file writes it: the body's joints of ``names`` in other units, axes and origin, and
    joints of its own (``extra``: (name, head in the body's frame))."""
    index = {n: i for i, n in enumerate(BODY['names'])}
    heads = np.array([BODY['heads'][index[n]] for n in names] + [h for _, h in extra], float)
    return {'names': list(names) + [n for n, _ in extra], 'heads': scale * heads @ _turn(turn).T + np.array(shift)}


def test_a_similarity_is_read_back_from_points_it_moved():
    p = np.random.default_rng(0).normal(size=(6, 3))
    q = 2.5 * p @ _turn(70).T + np.array([1, 2, 3])
    m, miss = assemble.similarity(p, q)
    assert miss < 1e-9 and np.allclose(m[:3, :3], 2.5 * _turn(70))


def test_a_part_written_in_its_own_units_axes_and_origin_lands_on_the_body():
    sks = [BODY, _part(['hip', 'leg_L', 'leg_R'])]
    placed, landed = assemble.place(sks, 0, 2.0)
    assert set(placed) == {0, 1} and landed[1] == 1.0
    assert np.allclose(assemble._apply(placed[1], sks[1]['heads']), BODY['heads'][[1, 2, 3]], atol=1e-9)


def test_one_root_or_joints_in_a_line_place_nothing():
    root_only = _part(['root'], extra=[('hand_pose', [0, 0, 0])])
    in_line = _part(['root', 'hip', 'chest', 'head'])
    placed, _ = assemble.place([BODY, root_only, in_line], 0, 2.0)
    assert set(placed) == {0}


def test_joints_of_the_same_names_kept_elsewhere_place_nothing():
    other = _part(['hip', 'leg_L', 'leg_R', 'chest'])
    other['heads'][1:3] += np.array([30.0, 0, 0])          # another character's legs, far apart
    placed, landed = assemble.place([BODY, other], 0, 2.0)
    assert set(placed) == {0} and landed[1] == 0.0


def test_a_placed_file_places_another_through_the_joints_it_brings():
    strands = [('strand1', [0.1, 0.1, 1.9]), ('strand2', [0.1, 0.2, 1.7]), ('strand3', [-0.1, 0.2, 1.6])]
    bridge = _part(['root', 'hip', 'chest', 'head', 'arm_L', 'arm_R'], scale=1.0, turn=0, shift=(0, 0, 0),
                   extra=strands)
    hair = {'names': ['hair_root', 'strand1', 'strand2', 'strand3'],
            'heads': 3.0 * (np.array([[0, 0, 1.8]] + [h for _, h in strands]) - [0, 0, 1.8])}
    placed, _ = assemble.place([BODY, hair, bridge], 0, 2.0)
    assert set(placed) == {0, 1, 2}
    assert np.allclose(assemble._apply(placed[1], hair['heads'][:1]), [[0, 0, 1.8]], atol=1e-9)


def test_a_part_spanning_the_whole_body_is_a_copy_of_it():
    body = np.array([[-1, -0.3, 0], [1, 0.3, 2]], float)
    assert assemble.covers(np.array([[-0.98, -0.29, 0.02], [0.97, 0.3, 1.99]]), body)
    assert not assemble.covers(np.array([[-0.3, -0.3, 0], [0.3, 0.3, 1.0]]), body)


def test_a_part_the_body_already_has_adds_nothing():
    body = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [2.5, 2.2, 2.4]], float)
    pieces = [(100, 0, 2), (500, 2, 4)]
    assert assemble.carried(body[2:] + 0.001, [(505, 0, 2)], body, pieces, 0.01)
    assert not assemble.carried(body[2:] + 0.001, [(300, 0, 2)], body, pieces, 0.01)
    assert not assemble.carried(body[2:] + 0.5, [(500, 0, 2)], body, pieces, 0.01)


def test_parts_riding_the_same_joints_are_worn_on_one_place():
    names = ['hip', 'leg', 'shin', 'chest', 'head', 'strand']
    rig = SimpleNamespace(names=names, parents=np.array([-1, 0, 1, 0, 3, 4]))
    rows = {'shorts': [0, 1], 'trousers': [2, 3, 4], 'tank': [5, 6], 'hair': [7, 8]}
    w = np.zeros((9, 6))
    w[0, 0] = w[1, 1] = 1                                  # shorts: hip and leg
    w[2, 0] = w[3, 1] = w[4, 2] = 1                        # trousers: hip, leg, shin
    w[5, 3] = w[6, 3] = 1                                  # tank: chest
    w[7, 4] = w[8, 5] = 1                                  # hair: the head and its own strand
    rig.weights = w
    units = [SimpleNamespace(key=k, rows=np.array(r)) for k, r in rows.items()]
    groups = assemble.places(rig, units, {'strand'})
    assert [[u.key for u in g] for g in groups] == [['shorts', 'trousers'], ['tank'], ['hair']]


def test_joints_in_a_line_place_a_file_only_where_they_coincide_as_they_are():
    spine = np.array([[0, 0, 0], [0, -0.03, 0.9], [0, -0.03, 0.91], [0, -0.04, 1.05], [0, -0.03, 1.17]])
    found = assemble.land(spine, spine, 0.02 * 1.7)
    assert found is not None and np.allclose(found[0], np.eye(4)) and found[1].all()
    assert assemble.land(spine * 1.3, spine, 0.02 * 1.7) is None              # in other units: a turn about it is open
    assert assemble.land(spine[[0, 0, 0]], spine[[0, 0, 0]], 0.02 * 1.7) is None    # one place says nothing


def test_a_head_file_anchoring_its_root_elsewhere_lands_by_the_joints_it_rides():
    head = {'names': ['chest', 'head', 'hair'], 'heads': np.array([[0, 0, 1.8], [0, 0, 1.8], [0, 0, 1.4]]),
            'rides': ['head', 'hair']}                    # chest (its root) kept on the neck; head, hair as the body has them
    body = {'names': BODY['names'] + ['hair'], 'heads': np.vstack([BODY['heads'], [[0, 0, 1.4]]])}
    placed, landed = assemble.place([body, head], 0, 2.0)
    assert set(placed) == {0, 1} and np.allclose(placed[1], np.eye(4)) and landed[1] == 1.0
    head.pop('rides')                                     # all its joints speaking, the root's place outvotes it
    assert set(assemble.place([body, head], 0, 2.0)[0]) == {0}


def test_on_is_the_share_of_points_lying_on_others():
    b = np.array([[0, 0, 0], [1, 0, 0]], float)
    a = np.array([[0.005, 0, 0], [1.0, 0.004, 0], [5, 5, 5], [0.5, 0.5, 0.5]], float)
    assert assemble.on(a, b, 0.01) == 0.5 and assemble.on(a[:0], b, 0.01) == 0.0


def test_a_part_reaching_as_far_as_the_body_s_head_file_is_no_copy_when_its_joints_reach_further():
    head = np.array([[-0.1, -0.1, 1.4], [0.1, 0.1, 1.7]])
    joints = np.array([[0, 0, 0], [0, 0, 0.9], [0, 0, 1.5], [0.3, 0, 1.3], [-0.3, 0, 1.3]])
    torso = np.array([[-0.5, -0.15, 0.8], [0.5, 0.15, 1.45]])
    assert assemble.covers(torso, head)
    assert not assemble.covers(torso, np.vstack([head, joints]))


def _found(tmp_path, body_points, files):
    """A survey's inputs: a body file and files placed as they are, each its mesh's points and its bones."""
    import json
    entries = [{'n': 1, 'file': 'body.dae', 'opened': True, 'armatures': 1, 'bones': 60, 'triangles': 100,
                'blend': '1-body.blend'}]
    sks = [{'names': ['root', 'hip', 'chest'], 'parents': [-1, 0, 1],
            'heads': np.array([[0, 0, 0], [0.1, 0, 1], [0, 0.1, 1.4]]), 'tails': np.zeros((3, 3)),
            'points': np.asarray(body_points, float), 'pieces': [(100, 0, len(body_points))],
            'rides': ['hip', 'chest']}]
    for k, (name, bones, points) in enumerate(files, 2):
        entries.append({'n': k, 'file': f'{name}.dae', 'opened': True, 'armatures': 1, 'bones': bones,
                        'triangles': 50 + k, 'blend': f'{k}-{name}.blend'})
        sks.append(dict(sks[0], points=np.asarray(points, float), pieces=[(50 + k, 0, len(points))]))
    (tmp_path / 'models.json').write_text(json.dumps(entries), encoding='utf-8')
    return entries, sks


def test_a_file_lying_where_the_body_has_none_is_worn_and_rivals_or_a_body_on_it_are_not(tmp_path, monkeypatch):
    grid = np.array([[x, y, z] for x in np.linspace(-0.2, 0.2, 5) for y in (-0.1, 0.1) for z in np.linspace(0, 1, 11)])
    torso = grid + [0, 0, 1.0]
    legs, hair_a, hair_b = grid, grid * [1, 1, 0.3] + [0, 0, 1.9], grid * [1, 1, 0.3] + [0, 0, 1.92]
    costume = np.vstack([torso, legs])[::3] + 0.002
    entries, sks = _found(tmp_path, torso, [('legs', 60, legs), ('hairA', 10, hair_a), ('hairB', 10, hair_b),
                                            ('costume', 60, costume[:40])])
    monkeypatch.setattr(assemble, 'skeletons', lambda load_dir, e: sks)
    found = assemble.survey(tmp_path, entries[0])
    name = {i: entries[i]['file'] for i in range(len(entries))}
    assert [name[i] for i in found.pieces] == ['legs.dae']
    assert [name[i] for i in found.parts] == ['hairA.dae', 'hairB.dae']


def test_a_place_most_answers_dress_is_dressed_in_the_version_named_most():
    sizes = [3, 2]                                 # A: three bottoms (1-3); B: two tanks (4, 5)
    assert assemble.tally([{'A': [2], 'B': [5]}, {'A': [3], 'B': [5]}, {'A': [], 'B': []}], sizes) == [2, 5]
    assert assemble.tally([{'A': [1], 'B': [4, 5]}, {'A': [3], 'B': [4, 5]}, {'A': [3]}], sizes) == [3, 4, 5]
    assert assemble.tally([{'A': [2]}, {'A': []}, {'cant_tell': True}], sizes) == []


def test_a_file_nothing_places_whose_mesh_rides_one_joint_of_its_own_is_an_object_once_per_model(tmp_path,
                                                                                             monkeypatch):
    grid = np.array([[x, 0, z] for x in (-0.2, 0.2) for z in np.linspace(0, 1, 5)])
    entries, sks = _found(tmp_path, grid, [('legs', 60, grid)])
    alone = {'parents': [-1, 0], 'heads': np.array([[5.0, 5, 5], [5, 5, 6]]), 'tails': np.zeros((2, 3)),
             'points': grid[:3], 'pieces': [(4, 0, 3)]}
    files = [('gauge', 'fbx', ['gauge', 'top'], ['gauge']), ('gauge', 'dae', ['gauge', 'top'], ['gauge']),
             ('wig', 'dae', ['head_root', 'strand'], ['head_root', 'strand']),     # rides two
             ('crop', 'dae', ['head_root', 'strand'], ['head_root'])]             # rides one every hair has
    for k, (name, ext, names, rides) in enumerate(files, len(entries) + 1):
        entries.append({'n': k, 'file': f'{name}.{ext}', 'opened': True, 'armatures': 1, 'bones': 2,
                        'triangles': 4 + k * (name != 'gauge'), 'blend': f'{k}-{name}.blend'})
        sks.append(dict(alone, names=names, rides=rides))
    (tmp_path / 'models.json').write_text(__import__('json').dumps(entries), encoding='utf-8')
    monkeypatch.setattr(assemble, 'skeletons', lambda load_dir, e: sks)
    found = assemble.survey(tmp_path, entries[0])
    assert [entries[i]['file'] for i in found.objects] == ['gauge.fbx']


def test_an_object_hangs_on_the_tile_every_answer_names_in_the_colour_nearest_theirs():
    rows = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    answers = [{'rows': [{'row': 'A', 'tile': 2, 'colour': '#ff8000'}, {'row': 'B', 'tile': 5, 'colour': 'none'}]},
               {'rows': [{'row': 'A', 'tile': 2, 'colour': '#f07010'}, {'row': 'B', 'tile': 5, 'colour': 'none'}]},
               {'rows': [{'row': 'A', 'tile': 2, 'colour': '#ff8000'}, {'row': 'B', 'tile': 0, 'colour': 'none'}]}]
    assert assemble.hung(answers, rows) == {'A': (2, (255, 128, 0))}       # B: two of three, a coin
    # a tile of another object's names nothing
    assert assemble.hung([{'rows': [{'row': 'A', 'tile': 5, 'colour': 'none'}]}] * 3, rows) == {}


def test_joints_at_one_place_and_turn_are_one_frame_whatever_their_scale():
    a, b = np.eye(4) * 0.025, np.eye(4) * 0.05
    a[3, 3] = b[3, 3] = 1
    assert assemble._same(a, b, 0.01)
    b[:3, 3] = [0, 0, 0.1]
    assert not assemble._same(a, b, 0.01)


def test_a_joint_a_part_brought_is_framed_as_that_part_is_placed():
    def frame(at):
        f = np.eye(4)
        f[:3, 3] = at
        return f
    body = {'names': ['root', 'chest'], 'frames': np.array([frame([0, 0, 0]), frame([0, 0, 1])])}
    tank = {'names': ['chest', 'Tank_Ink'], 'frames': np.array([frame([0, 0, 0]), frame([0, 0.5, 0])])}
    placed = np.eye(4) * 2
    placed[3, 3], placed[:3, 3] = 1, [0, 0, 1]
    found = SimpleNamespace(skeletons=[body, tank], body=0, placed={0: np.eye(4), 1: placed},
                            joined=[{'index': 1, 'names': {}}])
    frames = assemble.hosts(found, ['root', 'chest', 'Tank_Ink'])
    assert np.allclose(frames['chest'], body['frames'][1])           # the body's own, not the part's
    assert np.allclose(frames['Tank_Ink'][:3, 3], [0, 1, 1]) and np.allclose(frames['Tank_Ink'][:3, :3], 2 * np.eye(3))
