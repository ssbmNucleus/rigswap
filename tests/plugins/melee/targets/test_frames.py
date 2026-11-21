"""targets/frames.py: the fighter's game frames, dumped once as target data and read by the export."""
import json

from rigswap.plugins.melee.facts.fighters import fighter
from rigswap.plugins.melee.targets import frames

LISTING = '''Main called with 3 arguments
PlyNess5K_Share_ACTION_Wait_figatree\t4239
PlyNess5K_Share_ACTION_RunBrake_figatree\t100
PlyNess5K_Share_ACTION_Run_figatree\t200
PlyNess5K_Share_ACTION_SquatWait1_figatree\t300
PlyNess5K_Share_ACTION_AppealL_figatree\t400
PlyNess5K_Share_ACTION_AppealR_figatree\t500
PlyNess5K_Share_ACTION_AttackS4_figatree\t600
'''

ORIGINAL = {
    ('Wait1', 1), ('WalkMiddle', 10), ('Dash', 4), ('Run', 6), ('JumpF', 8), ('JumpAerialF', 5), ('Landing', 2),
    ('Squat', 4), ('SquatWait', 1), ('Guard', 5), ('Attack11', 3), ('AttackDash', 6), ('AttackS3S', 5),
    ('AttackHi3', 6), ('AttackLw3', 8), ('AttackS4S', 12), ('AttackHi4', 10), ('AttackLw4', 8),
    ('AttackAirN', 4), ('AttackAirF', 15), ('AttackAirB', 8), ('AttackAirHi', 6), ('AttackAirLw', 10),
    ('Catch', 8), ('ThrowF', 8), ('DamageHi1', 4), ('DamageFlyHi', 10), ('CliffWait', 1), ('Appeal', 30),
}
ADDED = {
    ('Run', 1), ('AttackAirF', 8), ('Appeal', 20), ('Landing', 4), ('LandingAirN', 2), ('LandingAirN', 6),
    ('DamageHi1', 2), ('DamageFlyHi', 4), ('CliffWait', 10), ('CliffWait', 40),
}


def test_gate_keeps_original_poses_and_covers_every_harness_shot_without_duplicates():
    assert len(frames.FRAMES) == len(set(frames.FRAMES)) == 39
    assert set(frames.FRAMES) == ORIGINAL | ADDED
    assert ';'.join(frames.INGAME_SHOTS) == (
        'idle:1;run:1,6;crouch:4;jab:3;utilt:6;fsmash:12;dsmash:8;nair:4;fair:8;dair:10;taunt:20;'
        'landing:2,4;landingair:2,6;damagehi:2,4;damagefly:4,10;cliffwait:10,40')
    assert {shot: (action, at) for action, at, shot in frames.POSES if shot} == {
        'idle': ('Wait1', (1,)), 'run': ('Run', (1, 6)), 'crouch': ('Squat', (4,)),
        'jab': ('Attack11', (3,)), 'utilt': ('AttackHi3', (6,)), 'fsmash': ('AttackS4S', (12,)),
        'dsmash': ('AttackLw4', (8,)), 'nair': ('AttackAirN', (4,)), 'fair': ('AttackAirF', (8,)),
        'dair': ('AttackAirLw', (10,)), 'taunt': ('Appeal', (20,)), 'landing': ('Landing', (2, 4)),
        'landingair': ('LandingAirN', (2, 6)), 'damagehi': ('DamageHi1', (2, 4)),
        'damagefly': ('DamageFlyHi', (4, 10)), 'cliffwait': ('CliffWait', (10, 40)),
    }


def test_existing_target_dumps_only_added_frames_and_reader_uses_the_same_list(tmp_path, monkeypatch):
    from rigswap.plugins.melee.export import write as export_write
    out = tmp_path / 'frames' / 'Mario'
    out.mkdir(parents=True)
    for action, frame in ORIGINAL:
        (out / f'{action}-{frame}.json').write_text(json.dumps({'frame': frame}), encoding='utf-8')
    # An unrelated dump must not silently extend the gate's documented coverage.
    (out / 'Run-999.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(export_write, 'vanilla_copies', lambda *args: args[1:])
    dumped = []

    def hsdraw(*args):
        if args[0] == '--anim':
            return '\n'.join(f'Ply_ACTION_{action}_figatree 100' for action, _ in frames.FRAMES)
        _, _, _, symbol, path, frame = args
        action = symbol.split('_')[-2]
        dumped.append((action, frame))
        path.write_text(json.dumps({'frame': frame}), encoding='utf-8')

    monkeypatch.setattr(export_write, 'hsdraw', hsdraw)
    made = frames.write(fighter('Mario'), tmp_path, tmp_path)
    assert made['there'] == 29 and made['none'] == []
    assert len(made['made']) == len(dumped) == 10 and set(dumped) == ADDED
    got = frames.read('Mario', tmp_path)
    assert set(got) == {frames.name(a, n) for a, n in ORIGINAL | ADDED}
    assert got['CliffWait-40']['frame'] == 40 and got['LandingAirN-6']['frame'] == 6
    assert frames.write(fighter('Mario'), tmp_path, tmp_path) == {'made': [], 'there': 39, 'none': []}


def test_an_action_is_found_by_its_own_name_then_the_names_fighters_give_it():
    assert frames.symbol(LISTING, 'Run') == 'PlyNess5K_Share_ACTION_Run_figatree'          # not RunBrake
    assert frames.symbol(LISTING, 'SquatWait') == 'PlyNess5K_Share_ACTION_SquatWait1_figatree'
    assert frames.symbol(LISTING, 'Appeal') == 'PlyNess5K_Share_ACTION_AppealR_figatree'   # the taunt to the right
    assert frames.symbol(LISTING, 'Wait1') == 'PlyNess5K_Share_ACTION_Wait_figatree'       # the family's name
    assert frames.symbol(LISTING, 'AttackS4S') == 'PlyNess5K_Share_ACTION_AttackS4_figatree'
    assert frames.symbol(LISTING, 'Guard') is None
    assert frames.symbol('PlyMario5K_Share_ACTION_Appeal_figatree\t1\n', 'Appeal') == \
        'PlyMario5K_Share_ACTION_Appeal_figatree'


def test_write_dumps_each_frame_once_from_copies_and_read_gives_them_back(tmp_path, monkeypatch):
    from rigswap.plugins.melee.export import write as export_write
    files, targets = tmp_path / 'files', tmp_path / 'targets'
    files.mkdir()
    for name in ('PlNsNr.dat', 'PlNsAJ.dat'):
        (files / name).write_bytes(b'vanilla')
    dumped = []

    def hsdraw(*args):
        if args[0] == '--anim':
            assert args[2].parent.name == 'vanilla-files'          # never the game's own file
            return LISTING
        _, costume, aj, symbol, path, frame = args
        assert costume.parent.name == 'vanilla-files' and aj.name == 'PlNsAJ.dat'
        dumped.append((symbol, frame))
        path.write_text(json.dumps({'symbol': symbol, 'frame': frame, 'bones': []}), encoding='utf-8')
        return ''
    monkeypatch.setattr(export_write, 'hsdraw', hsdraw)
    ness = fighter('Ness')
    made = frames.write(ness, files, targets)
    found = {s.split('_')[-2] for s, _ in dumped}
    assert found == {'Wait', 'Run', 'SquatWait1', 'AppealR', 'AttackS4'}
    assert len(made['made']) == 7 and made['there'] == 0
    assert 'Guard' in made['none'] and 'WalkMiddle' in made['none']
    again = frames.write(ness, files, targets)                     # kept: dumped once
    assert again['made'] == [] and again['there'] == 7 and len(dumped) == 7
    got = frames.read('Ness', targets)
    assert list(got) == sorted(got) and got[frames.IDLE]['symbol'] == 'PlyNess5K_Share_ACTION_Wait_figatree'
    assert got['Appeal-30']['frame'] == 30
    assert frames.read('Mario', targets) == {}
