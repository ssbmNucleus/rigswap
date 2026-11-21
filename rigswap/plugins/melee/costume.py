"""One download made into a Melee costume, for an app to run as a subprocess: load it, swap it onto the fighter and
export the fighter's .dat. Between loading and swapping, the targets the swap reads are made where this machine has
none yet (``ready``).

    python -m rigswap.plugins.melee.costume <download> <Fighter> <out> [--cache DIR]
    python -m rigswap.plugins.melee.costume --fix <out> "<what looks wrong>" [--heard "<summary>"] [--cache DIR]
    python -m rigswap.plugins.melee.costume --set <out> head=0.8 [limbs=1.2 ...] [--cache DIR]

The first makes a costume. The other two put one right (``fix``): the record ``make`` left is changed in one entry,
by a look at what the person said or by a size named outright, and the costume is built again from it. The result
then also says ``changed`` and ``why``; ``record`` is where the costume's record is, ``tripwire`` what the look at
the finished draft said, and ``warning`` is there when that look called it a wreck. ``<out>/done.json`` is the newest
result that ended ``done``: a change that fails leaves it alone, so the next one still starts from the last costume
that was made.

The download is an archive or a model file (load.open). The fighter is its CamelCase name (facts/fighters.py), or
``auto``: the fighter the model suits best, picked inside the swap (pick.py), said in the result as ``fighter_pick``
{fighter, why, picked, ...}. Everything the pipeline logs goes to stdout as it runs. ``<out>/result.json`` says how
it ended: {status: 'done' | 'failed', fighter, fighter_pick (auto only), dat, seconds, cost_usd, asks, failed:
{stage: 'load' | 'ready' | 'pick' | 'swap', cause}}. The targets, the game's files and the tools are where the environment
says (RIGSWAP_MELEE_TARGETS, RIGSWAP_MELEE_FILES, RIGSWAP_BLENDER4, RIGSWAP_HSDRAW: plugins/melee/__init__.py); the
AI's key too (OPENROUTER_API_KEY: ai.py). Asks are cached in ``--cache`` (else RIGSWAP_CACHE, else
``<out>/cache``); asks answered through the relay (RIGSWAP_ASK_RELAY) in ``<out>/cache`` whatever is named, so a new
import is answered afresh by whoever answers it and a change to an import reads that import's own answers (ai.answers)."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .pick import AUTO

# Headless on Linux (``headless``): an app runs this with the desktop's DISPLAY, and bpy then draws its pictures
# (Workbench) through the X server's GLX. A GLX that can't make a context crashes the whole process: the tower's
# NVIDIA driver, upgraded and not yet rebooted (kernel module 580.173, libraries 580.178), segfaulted every render
# with DISPLAY=:0, 2026-11-19. Without DISPLAY bpy draws through EGL, which uses the GPU's EGL where it works and falls
# back to Mesa's software renderer where it doesn't. Nothing here is drawn on screen, so the display is never needed.
HEADLESS_DROPS = ('DISPLAY', 'WAYLAND_DISPLAY')


def headless(environ=os.environ, platform=sys.platform, modules=sys.modules):
    """Drop the display variables on Linux before bpy is loaded (once loaded it has read them). The names dropped."""
    if not platform.startswith('linux') or 'bpy' in modules:
        return []
    return [name for name in HEADLESS_DROPS if environ.pop(name, None) is not None]


headless()


def make(download, fighter, out, cache=None):
    """Load ``download`` into ``out/1-load``, swap it onto ``fighter`` (``AUTO``: the one it suits best) into
    ``out/2-swap`` and export: the result (also written to ``out/result.json``)."""
    from ... import ai
    from ...core import run
    from ...load import open as load
    from . import pick
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = ai.answers(cache or os.environ.get('RIGSWAP_CACHE') or out / 'cache', out / 'cache')
    spent = []

    def ask(q, p, s, d, route):
        answer, info = ai.ask(q, p, s, d, cache_dir=cache, route=route)
        spent.append(float(info.get('cost') or 0.))
        return answer, info

    started, stage = time.monotonic(), 'load'
    result = {'fighter': fighter}
    try:
        print(f'loading {Path(download).name}', flush=True)
        load.run(download, out / '1-load')
        stage = 'ready'
        ready(fighter)
        stage = 'swap'
        print(f'swapping onto {fighter}', flush=True)
        report = run.swap(out / '1-load', fighter, out / '2-swap', ask, name=Path(download).name)
        result['fighter'] = report['fighter']
        if 'fighter_pick' in report:
            result['fighter_pick'] = report['fighter_pick']
            print(f"picked {report['fighter']}: {report['fighter_pick']['why']}", flush=True)
        # The tripwire called the draft a wreck (not upright, or in pieces). Where a look filled the body, a second
        # model is asked for it and the costume built once more; the second stands where the tripwire passes it, else
        # the first does, and the result says plainly that it probably failed. There is no third try: no look can be
        # trusted to say a draft is good, so nothing loops on one.
        alarm = (report.get('tripwire') or {}).get('alarm')
        if alarm and (report.get('record') or {}).get('source') == 'look':
            print(f"the draft looks wrong ({report['tripwire'].get('what') or 'not upright, or in pieces'}): "
                  'trying once more', flush=True)
            before = os.environ.get('RIGSWAP_FILL')
            os.environ['RIGSWAP_FILL'] = 'again'
            try:
                second = run.swap(out / '1-load', report['fighter'], out / '2-swap-again', ask,
                                  name=Path(download).name)
            finally:
                os.environ.pop('RIGSWAP_FILL', None) if before is None else os.environ.update(RIGSWAP_FILL=before)
            result['second_try'] = {'alarm': bool((second.get('tripwire') or {}).get('alarm'))}
            if not result['second_try']['alarm']:
                report, alarm = second, False
        if alarm:
            result['warning'] = ('this probably did not work: ' + (report['tripwire'].get('what')
                                                                   or 'the result is not upright, or is in pieces'))
            print(result['warning'], flush=True)
        result['tripwire'] = report.get('tripwire')
        result['record'] = str(Path(report['export']).parent.parent / 'record.json')
        result.update(status='done', dat=report['export'])
    except Exception as e:
        traceback.print_exc()
        result.update(status='failed', failed={'stage': 'pick' if isinstance(e, pick.Failed) else stage,
                                               'cause': f'{type(e).__name__}: {e}'[:600]})
    result.update(seconds=round(time.monotonic() - started, 1), cost_usd=round(sum(spent), 4), asks=len(spent))
    _ended(out, result)
    return result


def _ended(out, result):
    """Write how a run ended: ``result.json`` always, and ``done.json`` too when it made a costume."""
    text = json.dumps(result, indent=1)
    (out / 'result.json').write_text(text, encoding='utf-8')
    if result.get('status') == 'done':
        (out / 'done.json').write_text(text, encoding='utf-8')


def _last_done(out):
    """The newest result in ``out`` that made a costume (``done.json``, else a ``result.json`` that ended done)."""
    for name in ('done.json', 'result.json'):
        if (out / name).is_file():
            found = json.loads((out / name).read_text(encoding='utf-8'))
            if found.get('status') == 'done':
                return found
    return {}


def fix(out, words=None, sizes=None, cache=None, heard=None):
    """Put right the costume ``make`` left in ``out``: change one entry of its record and build it again.

    ``words``: what the person says looks wrong, in their own words; one look names the change (core/complaint.py).
    ``heard``: what an assistant between the person and here made of them, shown to the look apart from the
    person's words and only to say which part they mean. ``sizes``: a size
    named outright, {'head': 0.8} (multiplied into the record's, no look). The changed record goes beside the load,
    the costume is built again into ``out/2-swap-fix<n>``, and ``out/result.json`` says how it ended, with
    ``changed`` ({entry: [was, is]}) and ``why`` (the look's sentence). It starts from the newest costume that was
    made (``done.json``), so a change that failed costs nothing. A complaint the record cannot hold fails at stage
    'fix', its cause in plain words (``Unfixable: ...``)."""
    from ... import ai
    from ...core import complaint, pieces, record as records, run
    out = Path(out)
    cache = ai.answers(cache or os.environ.get('RIGSWAP_CACHE') or out / 'cache', out / 'cache')
    spent = []

    def ask(q, p, s, d, route):
        answer, info = ai.ask(q, p, s, d, cache_dir=cache, route=route)
        spent.append(float(info.get('cost') or 0.))
        return answer, info

    started, stage = time.monotonic(), 'fix'
    before = _last_done(out)
    result = {'fighter': before.get('fighter')}
    try:
        if not before.get('record') or not Path(before['record']).is_file():
            raise complaint.Unfixable('there is no finished costume here to change')
        swap_dir = Path(before['record']).parent
        if sizes:
            changed_record, changed = complaint.sized(
                json.loads((swap_dir / 'record.json').read_text(encoding='utf-8')), sizes)
            why = ''
        else:
            print('looking at what was said', flush=True)
            changed_record, changed, why = complaint.fix(swap_dir, words, ask, heard=heard)
        print('changing ' + ', '.join(f'{entry}: {was} -> {now}' for entry, (was, now) in changed.items()), flush=True)
        records.write(out / '1-load' / 'record.json', changed_record)
        n = 1 + sum(1 for p in out.iterdir() if p.name.startswith('2-swap-fix'))
        stage = 'swap'
        print(f"building it again onto {before['fighter']}", flush=True)
        try:
            report = run.swap(out / '1-load', before['fighter'], out / f'2-swap-fix{n}', ask)
        except pieces.NoPiece as e:     # a piece no piece of the model answers: the complaint's failure, not the build's
            stage = 'fix'
            raise complaint.Unfixable(str(e)) from e
        if report.get('pieces'):        # which pieces the request turned out to be
            done = report['pieces']
            changed = {'left out' if done['way'] == 'out' else 'brought back': [None, done['pieces']]}
        # A change that changes nothing is no change: the same file as before is said as a failure, never handed on as
        # a second costume (2026-11-09: a head size the build did not apply came back as "rebuilt with a smaller head")
        old, new = Path(before.get('dat') or ''), Path(report['export'])
        if old.is_file() and new.is_file() and old.read_bytes() == new.read_bytes():
            stage = 'fix'
            said = ', '.join(f'{entry}: {was} to {now}' for entry, (was, now) in changed.items())
            raise complaint.Unfixable(f'the change ({said}) made no difference: the costume came out the same as '
                                      'before')
        if (report.get('tripwire') or {}).get('alarm'):
            result['warning'] = ('this probably did not work: ' + (report['tripwire'].get('what')
                                                                   or 'the result is not upright, or is in pieces'))
            print(result['warning'], flush=True)
        result.update(status='done', dat=report['export'], changed=changed, why=why, tripwire=report.get('tripwire'),
                      record=str(Path(report['export']).parent.parent / 'record.json'))
    except Exception as e:
        traceback.print_exc()
        result.update(status='failed', failed={'stage': stage, 'cause': f'{type(e).__name__}: {e}'[:600]})
    result.update(seconds=round(time.monotonic() - started, 1), cost_usd=round(sum(spent), 4), asks=len(spent))
    _ended(out, result)
    return result


TARGET_FILES = ('.glb', '.camera.json', '.memory.json', '.description.json')   # what ``target`` reads, per fighter


def offered(files):
    """The fighters an import naming none may be put on: those in scope whose game files are in ``files``, the
    costume, the data and the animations a target is made from (targets/prepare.py, targets/frames.py). Whoever
    gives the files says so which fighters it offers: an app that saves no costume for the Ice Climbers gives no
    animations of theirs, and an import naming no fighter failed there getting Popo ready, in every build since Popo
    and Nana became targets (2026-11-01 to 11-10)."""
    from .facts.fighters import in_scope
    files = Path(files)
    return [f for f in in_scope() if all((files / name).is_file() for name in (
        f'Pl{f.code}Nr.dat', f'Pl{f.code}.dat', f'Pl{f.motions or f.code}AJ.dat'))]


def ready(fighter):
    """Make the targets the swap reads that are not made yet on this machine: the fighter's, or for ``auto`` those
    of every fighter the game files are there for (offered), from the vanilla files (prepare, then the description
    from the game's tables), about 13 seconds a fighter, once. An app ships no target, since a target is the game's
    own model: each is made from the user's disc. Failed naming the fighter that could not be made."""
    from .. import Failed
    from . import FILES_VAR, TARGETS_VAR, _folder, describe_from_tables, prepare
    from .facts.fighters import fighter as named
    from .pick import AUTO
    if not os.environ.get(TARGETS_VAR):
        raise Failed(f'where are the Melee targets? set {TARGETS_VAR}')
    folder = Path(os.environ[TARGETS_VAR])
    folder.mkdir(parents=True, exist_ok=True)
    if fighter == AUTO:
        files = _folder(None, FILES_VAR, 'vanilla game files')
        wanted = offered(files)
        if not wanted:
            raise Failed(f'no fighter has its game files in {files}')
    else:
        wanted = [named(fighter)]
    missing = [f for f in wanted if not ((folder / 'frames' / f.name).is_dir()
                                         and all((folder / f'{f.name}{end}').is_file() for end in TARGET_FILES))]
    for i, f in enumerate(missing, 1):
        print(f'getting {f.name} ready from the game files ({i} of {len(missing)}, once on this computer)', flush=True)
        made = prepare(names=[f.name]).get(f.name, '')
        if made not in ('made', 'already there'):
            raise Failed(f'{f.name} could not be made ready from the game files: {made}')
        describe_from_tables(f.name, folder)


if __name__ == '__main__':
    args = sys.argv[1:]
    cache = args[args.index('--cache') + 1] if '--cache' in args else None
    heard = args[args.index('--heard', 3) + 1] if '--heard' in args[3:] else None     # after the words, never them
    rest = [a for i, a in enumerate(args) if a != '--cache' and (i == 0 or args[i - 1] != '--cache')]
    if args and args[0] == '--fix':             # --fix <out> "<what looks wrong>": the words are one argument
        done = fix(args[1], words=args[2], cache=cache, heard=heard)
    elif rest and rest[0] == '--set':           # --set <out> head=0.8 [limbs=1.2 ...]
        done = fix(rest[1], sizes={k: float(v) for k, v in (a.split('=', 1) for a in rest[2:])}, cache=cache)
    else:
        download, fighter, out = rest
        done = make(download, fighter, out, cache)
    sys.exit(0 if done['status'] == 'done' else 1)
