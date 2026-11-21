"""facts/fighters.py: the fighter table."""
import os
from pathlib import Path

import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee import dat as melee_dat
from rigswap.plugins.melee.facts import fighters


def test_fighter_table_has_no_duplicates_and_every_target_has_a_code_and_a_root():
    for field in ('name', 'code', 'symbol'):
        values = [getattr(f, field) for f in fighters.FIGHTERS]
        assert len(values) == len(set(values)), field
    for f in fighters.in_scope():
        assert f.code and f.symbol.endswith('_Share_joint')
    for f in fighters.FIGHTERS:
        assert f.in_scope or f.why_out
        assert f.preloaded and all(size > 0 for size in f.preloaded)
    for f in fighters.in_scope():
        assert f.triangles > 0 and f.kept > 0


def test_zelda_and_sheik_are_two_targets_and_so_are_popo_and_nana():
    names = {f.name for f in fighters.in_scope()}
    assert {'Zelda', 'Sheik', 'Popo', 'Nana'} <= names and len(names) == 27
    assert fighters.fighter('zelda').code != fighters.fighter('Sheik').code
    assert fighters.fighter('Popo').code != fighters.fighter('Nana').code
    assert fighters.fighter('Nana').motions == 'Pp' and not fighters.fighter('Popo').motions
    with pytest.raises(Failed, match='a boss'):
        fighters.fighter('Master Hand')
    with pytest.raises(Failed, match='no Melee fighter'):
        fighters.fighter('Waluigi')
    assert fighters.fighter('Captain Falcon') is fighters.fighter('Ca')


def test_every_fighter_shows_model_0_at_spawn():
    for f in fighters.FIGHTERS:
        assert f.shown[0] == 0 and list(f.shown) == sorted(set(f.shown))


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlMr.dat').is_file(),
                    reason=f'needs vanilla files in {melee.FILES_VAR}')
def test_fighter_caps_and_scales_are_measured_from_vanilla():
    files = Path(os.environ[melee.FILES_VAR])
    most = {f.name: melee_dat.vanilla_max_influences(files / f'Pl{f.code}Nr.dat', f.symbol) for f in fighters.in_scope()}
    for f in fighters.in_scope():
        assert f.scale == pytest.approx(melee_dat.model_scale(f, files), abs=1e-6)
    # the cap is the most joints any vanilla costume weights a vertex to, the same for every fighter
    assert max(most.values()) == fighters.MAX_INFLUENCES == 6 and most['Kirby'] == 6 and most['Ness'] == 2
    assert {f.max_influences for f in fighters.in_scope()} == {fighters.MAX_INFLUENCES}
