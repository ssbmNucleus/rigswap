from rigswap.library import body_types
from rigswap.library.body_types import FIT
from rigswap.vocabulary import BODY_TYPES


def test_every_part_of_every_body_type_runs_between_its_joints():
    assert list(body_types.TYPES) == list(BODY_TYPES)
    for body_type in BODY_TYPES:
        joints = body_types.joints_of(body_type)
        names = [n for n, _, _ in joints]
        assert len(names) == len(set(names))
        assert joints[0][1] is None and all(p in names for _, p, _ in joints[1:])     # one root, the rest hang
        ends = body_types.part_ends(body_type)
        assert list(ends) == BODY_TYPES[body_type]
        assert {j for pair in ends.values() for j in pair} == set(names)
    assert len(body_types.joints_of('biped')) == 20


def test_each_entry_names_only_its_own_parts():
    for name, t in body_types.TYPES.items():
        assert t.name == name and list(t.parts) == BODY_TYPES[name]
        assert sorted(p for _, chain in t.chains for p in chain) == sorted(t.parts)     # each part in one chain once
        assert t.chains[0][0] is None and all(parent in t.parts for parent, _ in t.chains[1:])
        assert set(t.fit) <= set(t.parts) and set(t.fit.values()) <= set(FIT)
        assert set(t.required) <= set(t.parts) and t.head in t.required
        assert set(t.pose_moves) | set(t.pose_moves.values()) <= set(t.parts)
        joints = [n for n, _, _ in body_types.joints_of(name)]
        assert set(t.height.ankles) <= set(joints)
        assert not t.height.ankles or t.height.top in joints


def test_the_rules_name_the_head_the_hands_and_the_feet():
    biped = body_types.of('biped')
    assert biped.head == 'head' and biped.hands == ['left_hand', 'right_hand']
    assert biped.feet == ['left_foot', 'right_foot'] and biped.rule('left_thigh') is FIT['span']
    assert biped.tipped == ['left_hand', 'right_hand', 'left_foot', 'right_foot']
    assert FIT['hand'].to_tip == 'whole' and FIT['foot'].to_tip == 'ground' and not FIT['span'].to_tip
    assert body_types.of('quadruped').feet == ['left_forefoot', 'right_forefoot', 'left_hindfoot', 'right_hindfoot']
    assert body_types.of('ball').hands == [] and body_types.of('fish').feet == []
    # the part of a last segment that keeps its orientation in the world: the head anywhere in it, a foot that starts it
    assert body_types.world_part('biped', ['neck', 'head']) == 'head'
    assert body_types.world_part('biped', ['left_foot']) == 'left_foot'
    assert body_types.world_part('biped', ['left_shin', 'left_foot']) is None
    assert body_types.world_part('biped', ['left_hand']) is None


def test_a_bipeds_shoulders_and_hips_are_links_and_its_neck_a_neck():
    biped = body_types.of('biped')
    assert all(biped.rule(p) is FIT['link'] for p in ('left_shoulder', 'right_shoulder', 'hips'))
    # the neck stretches as a span, the head after it moving whole with its stretch (fit.head_seam)
    assert biped.rule('neck') is FIT['neck'] and body_types.of('quadruped').rule('neck') is FIT['neck']
    assert not FIT['neck'].link and not FIT['neck'].world and not FIT['neck'].to_tip
    assert FIT['link'].link and not FIT['link'].world and not FIT['link'].to_tip
    assert not any(r.link for name, r in FIT.items() if name != 'link')
    assert not biped.rule('spine').link and not biped.rule('left_upper_arm').link
    assert 'left_shoulder' not in biped.tipped and body_types.world_part('biped', ['left_shoulder']) is None
    assert biped.head == 'head' and 'neck' not in biped.hands + biped.feet
