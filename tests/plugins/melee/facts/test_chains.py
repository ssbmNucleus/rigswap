"""facts/chains.py: each fighter's chains and own roles, against the game's joint names and physics strands."""
import pytest

from rigswap.plugins.melee.targets import roles as describe_bones
from rigswap.plugins import melee
from rigswap.plugins.melee.facts import chains as facts_chains, fighters
from rigswap.plugins.melee.targets import bones as melee_bones
from rigswap.vocabulary import CHAIN_KINDS
from tests.plugins.melee.targets.test_bones import FILES


def test_chains_and_own_roles_agree_with_the_names_they_cover():
    for name, chains in facts_chains.CHAINS.items():
        names, table = melee_bones.JOINT_NAMES[name], melee_bones.ROLES[melee_bones.body_type(name)]
        joints = [j for _, js in chains for j in js]
        assert len(joints) == len(set(joints)), name
        for kind, js in chains:
            assert kind in CHAIN_KINDS
            for j in js:
                assert j not in names or table[names[j]] == kind, (name, j)
    for name, own in facts_chains.OWN_ROLES.items():
        allowed = describe_bones.roles(melee_bones.body_type(name))
        assert set(own.values()) <= set(allowed), name


@pytest.mark.plugin
@pytest.mark.skipif(not (FILES / 'PlCo.dat').is_file(), reason=f'needs the vanilla game files in {melee.FILES_VAR}')
def test_every_chain_the_games_physics_moves_is_listed():
    tables = melee_bones.bone_tables(FILES / 'PlCo.dat')
    for f in fighters.in_scope():
        listed = [set(js) for _, js in facts_chains.CHAINS.get(f.name, [])]
        for moved in melee_bones.dynamics(FILES / f'Pl{f.code}.dat', tables[f.name]['skipped']):
            assert any(set(moved) <= c for c in listed), (f.name, moved)
