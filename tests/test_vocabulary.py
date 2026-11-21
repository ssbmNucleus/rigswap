from rigswap.vocabulary import Description, Joint, Piece, BODY_TYPES


def test_round_trip(tmp_path):
    d = Description(body_type='biped', up='+y', front='-z', has=['tail'],
                    alt={'up': '+x'}, from_default=['holds'],
                    pieces=[Piece(id=1, moves_as='body', anchored_to='chest'),
                            Piece(id=2, moves_as='layer', anchored_to='left_thigh', belongs='layer_of', layer_of=1, sure=False)],
                    joints={'left_elbow': Joint(pos=[1, 2, 3], sure=False, alt=[1, 2.5, 3])})
    p = tmp_path / 'd.json'
    d.save(p)
    back = Description.load(p)
    assert back == d
    assert back.parts() == BODY_TYPES['biped']
    assert back.unsure() == [('field', 'up', '+x'), ('joint', 'left_elbow', [1, 2.5, 3]), ('piece', 2, None)]


def test_mirrored_joints_remain_one_fact_in_every_alternative_build(tmp_path):
    from rigswap.vocabulary import Joint, Description
    desc=Description('biped','+z','-y',pose='neutral',joints={
        'left_knee':Joint([3,4,5],False,[3,6,7]),
        'right_knee':Joint([-3,4,5],False,[-3,6,7])},
        joint_mirrors={'right_knee':'left_knee'})
    desc.save(tmp_path/'description.json')
    loaded=Description.load(tmp_path/'description.json')
    assert loaded.joint_mirrors==desc.joint_mirrors
    facts=loaded.unsure()
    assert facts==[('joint','left_knee',[3,6,7])]
    taken=loaded.taking(facts[0])
    assert taken.joints['right_knee'].pos==[-3,6,7]
    assert taken.joints['right_knee'].alt==[-3,4,5]
    assert loaded.joints['right_knee'].pos==[-3,4,5]


def test_a_build_takes_an_unsure_joint_at_its_alternative_and_nothing_else():
    """The copy has the joint at its other position, still unsure, its first position now the alternative; the
    description itself is unchanged. An alternative up, front or body type, or an unsure piece, can't be taken alone."""
    import pytest
    d = Description(body_type='biped', up='+y', front='-z', alt={'up': '+x', 'body_type': 'quadruped'},
                    pieces=[Piece(id=2, moves_as='body', anchored_to='chest', sure=False)],
                    joints={'left_elbow': Joint(pos=[1, 2, 3], sure=False, alt=[1, 2.5, 3]),
                            'neck': Joint(pos=[0, 0, 5])})
    facts = d.unsure()
    taken = d.taking(facts[2])
    assert taken.joints['left_elbow'] == Joint(pos=[1, 2.5, 3], sure=False, alt=[1, 2, 3])
    assert taken.joints['neck'] == d.joints['neck'] and taken.pieces == d.pieces and taken.alt == d.alt
    assert d.joints['left_elbow'].pos == [1, 2, 3]
    for fact, said in zip((facts[0], facts[1], facts[3]), ('describe again from orient on',
                                                           'describe again from round 2 on', 'no alternative')):
        with pytest.raises(ValueError, match=said):
            d.taking(fact)


def test_a_role_reads_as_the_list_of_what_it_names():
    """A role that is a list (one bone across several parts) is read as given, any other as a list of one: a chain's
    root too, which every reader reads this way, never as a key."""
    from rigswap.vocabulary import CANT_TELL, role_parts
    assert role_parts(['spine', 'chest']) == ['spine', 'chest'] and role_parts(('spine', 'chest')) == ['spine', 'chest']
    assert role_parts('chest') == ['chest'] and role_parts(CANT_TELL) == [CANT_TELL] and role_parts('none') == ['none']
