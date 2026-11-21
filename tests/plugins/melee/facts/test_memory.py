import os
import json
from pathlib import Path
import struct

import pytest

from rigswap.plugins.melee.dat import Dat
from rigswap.plugins.melee.facts.fighters import fighter
from rigswap.plugins import Failed
from rigswap.plugins.melee.facts.memory import (_sha256, _triangles, importer_build, measure, read_record,
                                                write_record)


@pytest.mark.plugin
@pytest.mark.parametrize('name', ['Mario', 'Zelda', 'Link', 'Bowser'])
def test_vanilla_memory_measurement_accounts_for_the_whole_file(name):
    f = fighter(name)
    if not (Path(os.environ.get('RIGSWAP_MELEE_FILES', '.')) / f'Pl{f.code}Nr.dat').is_file():
        pytest.skip('needs vanilla costume files')
    row = measure(f)
    assert row['file_bytes'] == row['kept'] + sum(row['bytes'][k] for k in ('geometry', 'textures', 'retained'))
    assert row['triangles'] >= f.triangles and row['pobjs'] > 0
    assert row['kept'] >= 0
    assert all(row['per_triangle'][k] > 0 for k in ('geometry', 'retained', 'placement'))
    for k, rate in row['per_triangle'].items():
        assert rate * row['triangles'] == pytest.approx(row['bytes'][k])
    assert measure(f) is row  # once per file version in this process


@pytest.mark.parametrize('change', [{'triangles': 0}, {'triangles': 1.5}, {'kept': -1}, {'fighter': 'Link'},
                                  {'schema': 99}, {'bytes': {}}, {'importer': {}}, {'method': ''}])
def test_invalid_record_fails_with_regeneration_instruction(tmp_path, change):
    from tests.plugins.melee.test_melee import _memory
    (tmp_path / 'Mario.memory.json').write_text(json.dumps(_memory() | change), encoding='utf-8')
    with pytest.raises(Failed, match=r'Mario.memory.json.*melee.prepare'):
        read_record(fighter('Mario'), tmp_path)


@pytest.mark.parametrize('exe,library', [('HSDRawViewer.exe', 'HSDRawViewer'), ('hsdrawviewer', 'hsdrawviewer')])
def test_importer_build_names_the_windows_and_the_headless_viewer(tmp_path, exe, library):
    """Windows ships HSDRawViewer.exe; macOS and Linux ship the headless build, whose assembly is hsdrawviewer."""
    stem = Path(exe).stem
    for name in (exe, f'{stem}.dll', 'HSDRaw.dll'):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / f'{stem}.deps.json').write_text(json.dumps(
        {'libraries': {'HSDRaw/1.0.0': {}, f'{library}/2.3.4': {}}}), encoding='utf-8')
    build = importer_build(tmp_path / exe)
    assert build['version'] == '2.3.4' and build['version_source'] == f'{stem}.deps.json'
    assert set(build['sha256']) == {exe, f'{stem}.dll', 'HSDRaw.dll'}


def test_importer_build_fails_on_a_viewer_it_cannot_name(tmp_path):
    (tmp_path / 'viewer').write_bytes(b'')
    (tmp_path / 'viewer.deps.json').write_text(json.dumps({'libraries': {'Other/1.0.0': {}}}), encoding='utf-8')
    with pytest.raises(Failed, match='cannot identify the HSDRawViewer build'):
        importer_build(tmp_path / 'viewer')


@pytest.mark.plugin
def test_real_importer_record_written_and_read_without_changing_vanilla(tmp_path):
    from rigswap.plugins.melee.export.write import HSDRAW_DEFAULT, HSDRAW_VAR
    files = Path(os.environ.get('RIGSWAP_MELEE_FILES', '.'))
    dat = files / 'PlMrNr.dat'
    if not dat.is_file() or not Path(os.environ.get(HSDRAW_VAR) or HSDRAW_DEFAULT).is_file():
        pytest.skip('needs vanilla files and HSDRawViewer')
    before = _sha256(dat)
    record = write_record(fighter('Mario'), files, tmp_path)
    assert read_record(fighter('Mario'), tmp_path) == record
    assert record['vanilla_sha256'] == before == _sha256(dat)
    exe = Path(os.environ.get(HSDRAW_VAR) or HSDRAW_DEFAULT)
    assert record['importer']['version'] and len(record['importer']['sha256'][exe.stem + '.dll']) == 64
    assert record['options'] == {'names': 'place', 'dobjs': 'high', 'show': ['0:0'], 'alpha_mode': 'Test'}
    assert record['triangles'] == fighter('Mario').triangles
    assert record['file_bytes'] == record['kept'] + sum(record['bytes'][k]
                                                      for k in ('geometry', 'textures', 'retained'))
    assert 'round trip' in record['method']
    assert list(tmp_path.iterdir()) == [tmp_path / 'Mario.memory.json']


@pytest.mark.parametrize('opcode,indices,count', [
    (0x90, [0, 1, 2, 2, 2, 3], 1), (0x98, [0, 1, 2, 2, 3, 4], 2),
    (0xA0, [0, 1, 2, 3], 2), (0x80, [0, 1, 2, 3], 2), (0xA8, [0, 1], 0)])
def test_triangle_count_excludes_strip_stitching(opcode, indices, count):
    d = Dat(struct.pack('>5I', 32, 0, 0, 0, 0) + bytes(12))
    p = d.add(bytes(0x18))
    attributes = d.add(struct.pack('>4I', 9, 2, 1, 4) + bytes(8) + struct.pack('>I', 0xFF) + bytes(20))
    command = bytes([opcode]) + struct.pack('>H', len(indices)) + bytes(indices)
    display = d.add(command + bytes(32 - len(command)))
    d.point(p + 8, attributes)
    d.point(p + 0x10, display)
    struct.pack_into('>H', d.data, p + 0x0E, 1)
    assert _triangles(d, p) == count
