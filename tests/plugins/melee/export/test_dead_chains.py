"""Target dynamics liveness uses surface weights, with no epsilon."""
from types import SimpleNamespace

from rigswap.plugins.melee.export.physics import skin_weights
from rigswap.plugins.melee.targets.physics import classify


def test_exact_zero_is_dead_and_any_positive_weight_is_alive():
    target = dict(existing_chains=3, chains=[[19, 20], [25, 26], [90, 91]])
    result = classify(target, [{'JOBJ_19': 0, 'JOBJ_25': 1e-30}, {'JOBJ_91': 1}])
    assert [r['index'] for r in result['alive_chains']] == [1, 2]
    assert result['alive_chains'][0]['weighted_vertices'] == {'JOBJ_25': 1}
    assert result['dead_chains'] == [dict(index=0, bones=['JOBJ_19', 'JOBJ_20'],
                                        weighted_vertices={}, why='zero skin weight on every chain bone')]
    assert result['reclaimable_chains'] == 1
    # Evidence alone cannot alter the engine's table in a different DAT.
    assert result['existing_chains'] == 3 and result['reclaimed_chains'] == 0
    assert 'alive_chains' not in target


def test_loose_vertices_and_empty_vertex_groups_do_not_keep_chains_alive():
    group = lambda index, weight: SimpleNamespace(group=index, weight=weight)
    vertex = lambda *groups: SimpleNamespace(groups=groups)
    mesh = SimpleNamespace(type='MESH', vertex_groups=[SimpleNamespace(index=0, name='JOBJ_19'),
                                                      SimpleNamespace(index=1, name='JOBJ_25')],
                           data=SimpleNamespace(polygons=[SimpleNamespace(vertices=[0, 1, 2])],
                                                vertices=[vertex(group(0, 0)), vertex(), vertex(),
                                                          vertex(group(1, 1))]))
    result = classify(dict(existing_chains=2, chains=[[19], [25]]),
                      skin_weights([SimpleNamespace(type='ARMATURE'), mesh]))
    assert result['alive_chains'] == []
    assert len(result['dead_chains']) == 2
