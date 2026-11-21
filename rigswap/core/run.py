"""One swap through the new middle: load's output in, build.blend + prep + export out.

    python -m rigswap.core.run <1-load dir> <Fighter|auto> <out dir> [--no-export]

``auto``: the fighter the model suits (plugins/melee/pick.py), picked at the seam between what every fighter reads
alike and what turns on the fighter.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import (assemble, body as body_reading, build, cloth, correspond, eyes, fill, forks as fork_reading,
               front as front_reading, hang, labels, look, onepiece, opacity, paint, refine, repose, rig as rigs,
               segments, variants)


def swap(load_dir, fighter_name, out, ask, export=True, targets=None, name=None, pick_only=False):
    """``name``: the download's, which the pick is told (else the model file's); ``pick_only``: stop once ``auto`` has
    picked the fighter (pick.py's own command)."""
    import os
    from ..plugins.melee import pick as fighter_pick
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # the prepared targets: RIGSWAP_MELEE_TARGETS, else ./melee-targets (tests/README.md says how to make them)
    targets = Path(targets or os.environ.get('RIGSWAP_MELEE_TARGETS') or 'melee-targets')
    report = {'fighter': fighter_name}
    # the record (record.py): one beside the load is taken as it is (a hand's, a look's, an earlier run's with an entry
    # changed), and what it says is not read or asked again; with none, the run reads the model and writes one.
    # Either way the build below goes on from <out>/record.json
    from . import record as records
    given = records.read(load_dir)
    told = given or {}
    has_body = 'body' in told or 'hips' in told
    body_reading.RECORD = given if has_body else None
    repose.DIALS = {key: float(told[key]) for key in records.DIALS if key in told}
    report['record'] = {'source': told.get('source', 'hand') if given else 'reading'}
    if fighter_name == fighter_pick.AUTO and told.get('fighter'):
        fighter_name = report['fighter'] = told['fighter']
    # with no fighter named, every fighter's skeleton as the match reads it, read once per machine (pick.prepare):
    # before the model opens, since a fighter is read in a scene of its own
    fighters = fighter_pick.prepare(targets) if fighter_name == fighter_pick.AUTO else None
    # a load with no rigged file is a segmented rip: its skeleton is read from its pieces (segments.py)
    entry = rigs.pick(load_dir) or segments.pick(load_dir)
    report['model'] = entry['file']
    blend = Path(load_dir) / 'models' / Path(entry['blend'].replace('\\', '/')).name
    onepiece.FITTED = None
    try:            # with the parts its other files place on it, by the joints they share with it (assemble.py)
        rig, parts = assemble.read(load_dir, entry, blend) if entry.get('armatures') else (segments.read(blend), None)
        pieces = not entry.get('armatures')
        if segments.lost(rig):      # a rig whose skeleton carries little of the mesh is read from its pieces too
            rig, parts, pieces = segments.read(blend), None, True
    except segments.OnePiece:       # one surface carrying its limbs: the joints a look can name, fitted inside it
        rig, parts, pieces = onepiece.read(blend, out, ask, told.get('up')), None, False     # (onepiece.py)
    # which way it stands: the way a file stands on its origin, when it does (body.on_origin: a model is made standing
    # on the floor at its origin). Else one AI pick of four seen in profile (along the skeleton's left to right, or a
    # one-piece model's volume's: Rig.mirror; a posed statue's skeleton pairs no limbs), which the skeleton may overrule
    # once the labels say which pairs are legs (below). Then the mirror axis from the skeleton among the axes across
    # up; then the front, one AI pick of the two it allows
    if 'up' in told:
        up, up_answer, sure, floor = np.asarray(told['up'], float), 'record', True, True
    elif onepiece.FITTED is not None:       # a fit's slices start from the top: up was settled there, the same way
        up, up_answer, sure, floor = onepiece.FITTED.up
    else:
        lateral = body_reading.symmetry_axis(rig) if rig.mirror is None else rig.mirror
        floor = body_reading.on_origin(rig, lateral)
        if floor is not None:
            up, up_answer, sure = floor, 'origin (the file stands on it)', True
        else:
            up, up_answer, sure = front_reading.choose_up(lateral, out, ask)
    # The look that fills the body of the record (fill.py): with no body given, one ask names the body's joints and
    # its front, and code reads the sides. A model of one file is asked before anything else of it; one of several
    # files or of pieces, once its parts are on (those are chosen with the front read the old way). A body the look
    # cannot name is left to the reading below. RIGSWAP_FILL: 0 leaves every body to the reading; "again" asks the
    # second model first (the costume's second try, after the tripwire called its first a wreck)
    mode = os.environ.get('RIGSWAP_FILL', '1')
    looks = mode != '0' and not has_body
    whole = not (parts and parts.joined) and not pieces
    filled = None
    settle = onepiece.named if onepiece.FITTED is not None else None    # a fit is finished from the names (fill.entries)
    if looks and whole:
        filled, report['fill'] = fill.read(rig, up, out, ask, second_first=mode == 'again', settle=settle)
    if filled:
        front, answer = np.asarray(filled['front'], float), 'the look'
    elif 'front' in told:
        front, answer = np.asarray(told['front'], float), 'record'
    else:
        side = body_reading.mirror_axis(rig, up) if rig.mirror is None else rig.mirror
        facing = np.cross(up, side)
        facing /= np.linalg.norm(facing)
        front, answer = front_reading.choose(facing, out, ask, up)
    report['up'] = {'vector': np.round(up, 3).tolist(), 'answer': up_answer}
    report['front'] = {'vector': front.round(3).tolist(), 'answer': answer}
    if parts and parts.joined:      # of the parts, the ones the everyday look wears (its pieces are worn): one ask
        rig, report['parts'] = assemble.choose(rig, parts, up, front, out, ask)
    # of a segmented rip's contacts on loops through its trunk, those where one piece only rests on the other (Sofia's
    # hand on her thigh): one ask, a picture of each; they join pieces only where nothing else does (segments.py)
    if pieces:
        rig, report['touching'] = segments.touching(rig, blend, up, front, out, ask)
    if looks and not whole:
        filled, report['fill'] = fill.read(rig, up, out, ask, second_first=mode == 'again', settle=settle)
        if filled:
            front, answer = np.asarray(filled['front'], float), 'the look'
            report['front'] = {'vector': front.round(3).tolist(), 'answer': answer}
    if filled:          # the body is the look's from here on: nothing of it is read or asked again
        told = {**told, **filled, 'source': 'look',
                'character': told.get('character') or report['fill'].get('character')}
        has_body = True
        body_reading.RECORD = told
        report['record']['source'] = 'look'
    # A skeleton fitted inside a file that had none (onepiece.py) is finished now its joints are named, or goes back
    # to the joints it offered when a try at naming them did not hold. The numbered joints a complaint is shown
    # (below) are the offered ones, every one of them, so a record's are drawn before the fit is finished
    joints_drawn = 'fill' in report
    if onepiece.FITTED is not None:
        if has_body and 'hips' in told:
            if not joints_drawn:
                joints_drawn = True
                try:
                    fill.shown(rig, up, out)
                except Exception as e:      # noqa: BLE001
                    report['record']['joints_not_drawn'] = f'{type(e).__name__}: {e}'
            rig, told = onepiece.named(told)        # the record as the fit holds it is the one the build goes on from
            body_reading.RECORD = told
        elif onepiece.FITTED.done is not None:
            rig = onepiece.unnamed()
        onepiece.FITTED = None
    # what each twin pair is, legs, arms or a four-legged body's front legs: one ask, with a picture of the pairs
    # (labels.py); body reading takes the limbs from it
    if has_body:        # the record names the body: nothing of it is asked
        roles, report['labels'] = {}, {'pairs': [], 'character': [told['character']] if told.get('character') else []}
        # the pictures are still drawn: the first, the character in its own colours, is what the attach and paint asks
        # show first, and an ask shown other pictures is a new ask (seam-2: Kamek's attach was asked again, live)
        pairs = labels.candidates(rig, up)
        if pairs:
            labels.draw(rig, pairs, up, front, Path(out) / 'labels.png')
        # and the numbered joints, where no look drew them in this run (the body came from a record): what a complaint
        # about this build is shown, so one change can follow another (complaint.py). Never fails a build
        if not joints_drawn:
            try:
                fill.shown(rig, up, out)
            except Exception as e:      # noqa: BLE001
                report['record']['joints_not_drawn'] = f'{type(e).__name__}: {e}'
    else:
        roles, report['labels'] = labels.read(rig, up, front, out, ask)
    # the skeleton's word on up, now the labels say which pairs are legs: unless the picture's bottom has none of them
    # on it where another way's has (body.standing), a body stands on its legs, and a forward lean or a long tail can
    # turn the picture's answer over (Sceptile). A pick most asks didn't give (a ball with wings reads every way) is no
    # answer either: the way its legs stand best then, when they do (Meta Knight). Then the side and the front again
    if floor is None and not has_body:
        lateral = body_reading.symmetry_axis(rig) if rig.mirror is None else rig.mirror
        stands = body_reading.standing(rig, lateral, [n for n, role in roles.items() if role in ('legs', 'front legs')])
        # A picture's up matches the nearest candidate direction despite camera float noise.
        picked = max(stands, key=lambda us: np.dot(us[0], up))[1]
        best = sorted(stands, key=lambda us: -us[1])
        if (picked == 0 or not sure) and best[0][1] > picked and best[0][1] > best[1][1]:
            up, up_answer = best[0][0], f'skeleton (the picture said {up_answer})'
            if 'front' not in told:
                side = body_reading.mirror_axis(rig, up) if rig.mirror is None else rig.mirror
                facing = np.cross(up, side)
                facing /= np.linalg.norm(facing)
                front, answer = front_reading.choose(facing, out, ask, up)
            report['up'] = {'vector': np.round(up, 3).tolist(), 'answer': up_answer}
            report['front'] = {'vector': front.round(3).tolist(), 'answer': answer}
    said = report['labels'].get('character') or []
    character = Counter(said).most_common(1)[0][0] if said else None
    # the objects in files of their own (her tank's ink, a broom), on the joints the everyday look hangs them on: one
    # ask, a picture of each on each joint carrying nothing, the character named (assemble.attach)
    if parts and parts.objects:
        rig, report['attach'] = assemble.attach(rig, parts, up, front, out, ask, character, out / 'labels-0.png')
    # which way each of those limbs goes where its bones branch, toward its hand or foot: one ask, a picture of each
    # fork (forks.py)
    if has_body:
        forks, report['forks'] = {}, {'kinds': {n: 'digits' for n in told.get('digits', [])}}
    else:
        forks, report['forks'] = fork_reading.read(rig, up, front, out, ask, roles)
    digits ={n for n, k in (report['forks'].get('kinds') or {}).items() if k == 'digits'}
    # with no fighter named, the one it suits: on each fighter the match says what the swap would do to the model,
    # and one ask picks, every fighter offered with that said (pick.py). Everything after this turns on the fighter
    if fighters is not None:
        try:
            fighter_name, report['fighter_pick'] = fighter_pick.fighter(
                rig, up, front, roles, forks, digits, fighters, targets, out, ask,
                name or Path(entry['file'].replace('\\', '/')).name, character)
        except Exception as e:
            raise fighter_pick.Failed(f'{type(e).__name__}: {e}') from e
        report['fighter'] = fighter_name
        if pick_only:
            (out / 'core.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
            return report
    # the looks the game switches between, all in the file: only the everyday one goes on (look.py)
    from ..plugins.melee.targets.weapons import read_weapons
    slots = read_weapons(targets / f'{fighter_name}.glb')
    rig, made = variants.surfaces(rig)      # a file of one mesh: the surfaces it is welded from are its pieces
    decided = 'drop' in told        # the record says what the everyday look leaves out: nothing of it is asked
    # armed: the fighter holds its weapon through play, the part the game names 'sword' (Link, Young Link, Marth,
    # Roy); one taken out for a move is a 'weapon' (Peach's club, Ganondorf's sword, Game & Watch's)
    armed = any(s.get('part') == 'sword' for s in slots)
    # a file's own idle has settled its look (load.stance): there is no set to ask about, and nothing it shows goes
    settled = bool(entry.get('stance'))
    report['stance'] = entry.get('stance')
    if 'pieces' in told and not decided:    # a request with no list to change (a hand record): there is none to keep
        del told['pieces']
    if decided:
        import bpy
        if 'pieces' in told:        # a complaint's piece to leave out or bring back: settled now, every mesh still
            from . import pieces    # in the scene, and spent (pieces.py); the record written below holds the result
            told['drop'], report['pieces'] = pieces.settle(rig, told, up, front, out, ask)
            del told['pieces']
        gone = [n for n in told['drop'] if n in bpy.data.objects]
        layers = look.layered(rig, variants.units(rig)[0], gone)
        rig, report['variants'] = variants.drop(rig, gone), {'dropped': gone, 'record': True, 'layers': layers}
    else:       # one look at every piece says what the everyday look leaves out, and nothing overrides it (look.py)
        rig, report['variants'] = look.keep(rig, up, front, out, ask, fighter_name, character=character,
                                            weapons=[s['kind'] for s in correspond.armed(slots)] if armed else [],
                                            settled=settled)
    if made is not None:    # the pieces kept are the file's one mesh again; layers are between meshes, and it is one
        rig, report['variants']['layers'] = variants.whole(rig, made), []
    # what every chain off the body is, hair, cloth, a tail, armour, a prop...: one ask, with a picture of the chains
    # (labels.chains). Hair and cloth hang, from the bone they hang from; the rest rides it (correspond, hang)
    try:
        seen = body_reading.read(rig, up, front, roles, forks)
    except (StopIteration, ValueError):     # no body read: no chains off it to name
        seen = None
    if 'parts' in told:     # the record names the chains off the body
        named, report['labels']['chains'] = told['parts'], {'parts': [], 'record': True}
    else:
        named, report['labels']['chains'] = ({}, {'parts': []}) if seen is None else \
            labels.chains(rig, seen, up, front, out, ask, report['forks'].get('kinds'))
    # what each prop is, where the fighter's hands hold things of more than one kind (Link's sword and shield): one ask,
    # with a picture of the props (labels.equipment). A weapon slot takes only a prop of its kind (correspond)
    kinds = sorted({s['kind'] for s in correspond.armed(slots)})
    if 'equipment' in told:
        held_kinds, report['labels']['equipment'] = told['equipment'], {'objects': [], 'record': True}
    else:
        held_kinds, report['labels']['equipment'] = ({}, {'objects': []}) if seen is None or len(kinds) < 2 else \
            labels.equipment(rig, [p[:3] for p in correspond.props(rig, seen)], up, front, out, ask, kinds)
    # a hand that holds its weapon holds it alone, and a prop no slot takes on bones left at the file's origin is laid
    # nowhere: they go, and the rig is read again (correspond.unheld)
    if held_kinds and not decided:
        report['labels']['equipment']['gone'] = correspond.unheld(rig, seen, held_kinds, slots)
        rig = variants.drop(rig, report['labels']['equipment']['gone'])
    (out / 'core.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    import bpy
    source_arm = max((ob for ob in bpy.data.objects if ob.type == 'ARMATURE'), key=lambda ob: len(ob.data.bones))
    frig, farm, fmeshes = rigs.import_fighter(targets / f'{fighter_name}.glb')
    report['resolution'] = refine.resolution(rigs.objects(rig), fmeshes)
    if report['resolution']['cuts']:
        rig = rigs.extract(rig.blend, source_arm, rigs.objects(rig))
    description = correspond.read_description(targets / f'{fighter_name}.description.json')
    repose.normalize(rig, up, front, 1.0)
    canonical = (np.array([0., 0., 1.]), np.array([0., -1., 0.]))
    from ..plugins.melee.targets.bones import item_joints
    from ..plugins import melee
    drawn = melee.drawn(fighter_name, fmeshes)
    k = scale(room(rigs.objects(rig)), room(fmeshes)) if drawn is None else outlined(rigs.objects(rig), drawn)
    said = float(told.get('size', 1.0))     # the record's size, a multiple of the reading's
    if drawn is not None:                   # a flat fighter's size is what the game draws: read, and done
        k *= said
    report['scale'] = round(float(k), 4)
    repose.normalize(rig, *canonical, k)
    # The seam. What the run read is written down, and the build goes on from the file: the body, its marks, the
    # chains' names, the digits and the props' kinds all come back from record.json, none from the reading's memory.
    if not has_body:
        shape = body_reading.read(rig, *canonical, roles, forks)
        chains, shared = correspond.model_chains(rig, shape)
        dropped = (list(report['variants'].get('dropped') or [])
                   + list(report['labels']['equipment'].get('gone') or []))
        read = records.of_reading(rig, shape, chains, shared, up, front, named, digits, held_kinds, dropped,
                                  fighter_name, character)
        told = records.write(out / 'record.json', {**read, **{key: told[key] for key in told if key not in read}})
    else:       # the body was given (a hand's, a look's, an earlier run's): with it, what this run decided beside it
        dropped = (list(report['variants'].get('dropped') or [])
                   + list(report['labels']['equipment'].get('gone') or []))
        beside = {'version': records.VERSION, 'source': 'hand', 'up': [float(x) for x in up],
                  'front': [float(x) for x in front], 'character': character, 'fighter': fighter_name,
                  'drop': dropped, 'parts': named, 'digits': sorted(digits), 'equipment': held_kinds}
        told = records.write(out / 'record.json', {**beside, **told})
    body_reading.RECORD = told
    repose.DIALS = {key: float(told[key]) for key in records.DIALS if key in told}
    shape = body_reading.read(rig, *canonical)
    named, digits, held_kinds = told.get('parts', named), set(told.get('digits', digits)), told.get('equipment',
                                                                                                    held_kinds)
    if drawn is None:
        # The model stands at the fighter's room. Where it stands as the fighter does, its own height to the head
        # joint says what size keeps its build (grown()), and it is stood at that size before it is matched
        grow, read = grown(rig, shape, frig, description, said)
        report['size'] = {'room': round(float(k), 4), 'height': None if read is None else round(float(k * read), 4),
                          'record': said, 'alike': alike(shape, description)}
        if abs(grow - 1.0) > 1e-3:
            k *= grow
            repose.normalize(rig, *canonical, grow)
            shape = body_reading.read(rig, *canonical)
        report['scale'] = round(float(k), 4)
    m = correspond.match(rig, shape, frig, description, items=item_joints(fighter_name),
                         slots=slots, named=named, kinds=held_kinds, digits=digits)
    # a skirt with fewer strands than the fighter's ring of them is laid on the whole ring by angle; on a fighter whose
    # game swings no skirt, on its two legs (cloth.py)
    gown = cloth.gown(rig, m, frig, description)
    report['chains'] = m.notes
    # a model bone and the one it hangs from on fighter bones not next to each other: they part (correspond)
    report['neighbours'] = [list(x) for x in m.apart]
    fighter_bones = sorted(m.embedded)
    w = cloth.weights(rig, m.owner, fighter_bones, gown) if gown else repose.fighter_weights(rig, m.owner,
                                                                                             fighter_bones)
    w = repose.alone(w, fighter_bones, m.whole)    # a prop a weapon slot took rides it alone (correspond._match_props)
    # a fighter joint inside a model bone (a one-bone arm's elbow) cuts its mesh
    before_split = w
    w = repose.split(rig, w, fighter_bones, m.splits, m.owner)
    split_vertices = int(np.any(w != before_split, axis=1).sum())
    surface = repose.Surface(rig, refine.seam(rigs.objects(rig)))
    game_before = w.copy()
    w, game_seams = surface.blend(w, fighter_bones, m.embedded)
    # a gown on the legs is held at rest by moves and shares of its own (cloth.legs): the points come from those (cut
    # and blended on the same surface as w), the game moves them by w
    if gown and gown.held is not None:
        held_by, held, moves = cloth.rest(rig, m.owner, fighter_bones, gown, {})
        held = repose.split(rig, repose.alone(held, held_by, m.whole), held_by, m.splits, m.owner)
        rest_before = held.copy()
        embedded = {**m.embedded, **{('held', fb): m.embedded[fb] for fb in gown.columns}}
        held, rest_seams = surface.blend(held, held_by, embedded)
    else:
        held_by, held, moves = fighter_bones, w, {}
        rest_before, rest_seams = game_before, game_seams
    T = repose.transforms(m, fighter_bones)
    moves = {**T, **moves}
    points = repose.deform(rig.points, held, held_by, moves)
    report['ends'] = repose.end_report(m, frig, fighter_bones, w, points, T)
    np.savez_compressed(out / 'ends.npz', source=rig.points, placed=points, weights=w,
                        bones=np.asarray([frig.names[b] for b in fighter_bones]))
    influences = np.count_nonzero(rig.weights, axis=1)
    report['seams'] = {'source_rigid_vertices': int((influences == 1).sum()),
                       'source_smooth_vertices': int((influences > 1).sum()), 'split_vertices': split_vertices,
                       'game': game_seams, 'rest': rest_seams}
    # Keep the numerical evidence beside the build so a tear can be traced to its surface and weights without a rerun.
    if game_seams['vertices'] or rest_seams['vertices']:
        np.savez_compressed(out / 'seams.npz', points=rig.points, edges=surface.edges, placed=points,
                            game_before=game_before, game=w, rest_before=rest_before, rest=held,
                            source_influences=influences)
    report['owners'] = {rig.names[mb]: frig.names[fb] for mb, fb in m.owner.items()}
    # per fighter bone carrying mesh: how far its stretch of the model is stretched (below 1: squashed), the vertices
    # it carries, and whether it keeps the model's own size (head, hands, feet, swung chains)
    report['stretch'] = {frig.names[fb]: [round(float(np.linalg.norm(np.subtract(*m.rest[fb][::-1])) /
                                                max(np.linalg.norm(np.subtract(*m.embedded[fb][::-1])), 1e-6)), 2),
                                          int((w[:, i] > 0.5).sum()), fb in m.ends]
                         for i, fb in enumerate(fighter_bones) if fb not in m.placed and (w[:, i] > 0.5).any()}
    chains, carried = build.loose(rig, m, T, frig, w, fighter_bones)
    blend = build.write(out / '3-build', rig, points, w, fighter_bones, frig, farm, fmeshes, fighter_name, chains,
                        carried)
    report['build'] = str(blend)
    # whether an image's alpha cuts it, where a material wiring no alpha draws it cut: one ask, the parts cut and
    # whole; an image drawn whole goes opaque (opacity.py). Before the paint, whose images it may be
    shown = opacity.ready(out / '3-build', out, ask, character)
    report['opacity'] = {k: shown[k] for k in ('skipped', 'asked', 'opaque') if shown.get(k)}
    # the colours a game paints on in play (a team's ink), where a mask beside a colour image says where: one ask
    # (paint.py). Before the eyes, so an iris the game paints is painted in the pictures they read and the image they
    # take their own from
    painted = paint.ready(out / '3-build', load_dir, out, ask, character, out / 'labels-0.png',
                          report['variants'].get('layers'))
    report['paint'] = {k: painted[k] for k in ('skipped', 'painted', 'unpainted', 'layered') if painted.get(k)}
    # the eyes: their faces as objects of their own, with the pictures of their blink (eyes.py); a model whose eyes
    # can't be found or made ready exports without a blink
    try:
        made = eyes.ready(out / '3-build', _on_head(rig, shape), load_dir, out, ask)
    except Exception as e:      # the build on disk is the one before the eyes: prep and export go on without them
        import traceback
        made = eyes._write(out / '3-build', {'skipped': f'failed: {type(e).__name__}: {e}',
                                             'trace': traceback.format_exc()})
    report['eyes'] = {k: made[k] for k in ('skipped', 'named', 'answer', 'objects', 'made', 'lid', 'notes')
                      if k in made}
    # which of the hair and cloth can swing (where their surfaces are joined, and to what), and those the pose holds
    # out draped as they hang at the fighter's rest, clear of the body (hang.py). After the eyes: their pictures show
    # the head as bound, so a strand draped over the face never changes what they read
    report['hang'] = hang.settle(out / '3-build')
    # the weapons a prop took: prep and export leave out the vanilla weapon and its stowed forms (facts/parts.py)
    taken = out / '3-build' / 'weapons_replaced.json'
    taken.unlink(missing_ok=True)
    if m.replaced:
        taken.write_text(json.dumps(m.replaced, indent=1), encoding='utf-8')
        report['weapons_replaced'] = [r['part'] for r in m.replaced]
    (out / 'core.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    if export:
        melee.prep(out / '3-build')
        dat = melee.export(out / '3-build', out / '4-export' / f'{fighter_name}.dat')
        report['export'] = str(dat)
        (out / 'core.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
        # the tripwire (plugins/melee/tripwire.py): the finished draft in two of the fighter's frames beside the
        # source; is it upright, is every part attached. Last, since it opens the build: nothing after needs the scene.
        # An alarm fails nothing here; the costume's caller decides what a second try is. RIGSWAP_TRIPWIRE=0: not asked
        if os.environ.get('RIGSWAP_TRIPWIRE', '1') != '0':
            from ..plugins.melee import tripwire
            source = next((p for p in (out / 'fill-colours.png', out / 'labels-0.png') if p.is_file()), None)
            report['tripwire'] = tripwire.check(out / '3-build', fighter_name, targets, out, ask, source)
            (out / 'core.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def room(objects):
    """The room the surface of mesh ``objects`` walls in (onepiece.room)."""
    return onepiece.room(*segments._surface(segments._meshes(objects))[:2])


def outlined(objects, theirs):
    """The size of a model on a flat fighter, whose size is what the game draws (facts/flat.py): the scale at which the
    model's broad side, the one the game will show (prep/flat.py), covers what the outline of mesh objects ``theirs``
    covers. One reading of two areas."""
    points, tris = segments._surface(segments._meshes(objects))[:2]
    his, faces = segments._surface(segments._meshes(theirs))[:2]
    broad = max(onepiece.outline(points, tris, along) for along in (0, 1))
    return (onepiece.outline(his, faces, 0) / broad) ** (1 / 2)


def scale(ours, theirs):
    """The scale at which a model, in its own proportions, takes the fighter's room. One reading of two rooms
    (``room``): where a model first stands, and its size where its bones say nothing (bones())."""
    return (theirs / ours) ** (1 / 3)


ROOM = 1.5      # a model sized by its own height takes at most this many times the fighter's room


def stands(rig, shape, frig, description):
    """How many times its present size a model must be to stand as tall to its head joint as the fighter does, or
    None where it has no such height. The re-pose puts every bone at the fighter's length whatever the model's size,
    so the size only says how thick the model comes out and how big its head, hands and feet are beside it. At the
    size where its legs and trunk together are already the fighter's length, it keeps the build it was made with.
    Sized by room alone a skeleton, thin bones of almost no room, stood twice Ganondorf's height, was squeezed to
    0.45 along every bone and kept a skull twice too big (the app's import of 2026-11-10); Mecha Elmo on Ness stood
    at 56.1, bloated, and at 35.0 by this height. People porting by hand keep a character's build: of 169 costumes on
    the website a slim one takes a quarter of its fighter's room and a stout one two and a half times it
    (measured 2026-11-10).
    The height is the trunk's head-joint level (correspond.levels): one length of the whole standing body. Before
    2026-11-01 this was the size; the room then replaced it everywhere. (Between the two, the size was read bone by
    bone, the stretch of the bone carrying the middle point of the stretched surface: bones of one body disagree, a
    leg squeezed to 0.6 beside a trunk stretched to 1.5, and which of them carries the middle point is how the points
    happen to be shared out. It said 0.65 of the room for Crash, 0.58 for a bean, 0.49 for a kangaroo, where this
    height says 0.87, 0.86 and 0.68.)"""
    ours = correspond.levels(correspond.model_chains(rig, shape)[0])[2]
    theirs = correspond.levels(correspond.fighter_chains(frig, description), fighter=True)[2]
    return float(theirs / ours) if ours > 1e-6 and theirs > 1e-6 else None


def alike(shape, description):
    """Whether a model stands as the fighter does, so that its height to the head joint says its size (stands()): a
    two-legged body on a two-legged fighter. A four-legged body's trunk lies along the ground and its front legs
    stand in for arms, and a ball has no trunk or limbs to speak of: their bones are laid on the fighter's but are not
    the same bones, and the room stands, as the hand ports of such bodies have it (hand-diff, 2026-11-01). Vulpix on
    Pikachu, sized by its bones in size-2: its long trunk on Pikachu's short one carried 61 % of its body and it stood
    at 0.42 of its room, tiny and thin."""
    return shape.plan == 'biped' and correspond.fighter_plan(description) == 'biped'


def grown(rig, shape, frig, description, said=1.0):
    """(how many times the size it stands at, the fighter's room, a model is to be; what its height said, or None):
    for a body that stands as the fighter does (alike()), the size at which it stands as tall to its head joint
    (stands()), no more than ROOM times the fighter's room; for any other, the room. Times the record's size
    (``said``). One reading for the swap and for the pick of a fighter (plugins/melee/pick.fit)."""
    read = stands(rig, shape, frig, description) if alike(shape, description) else None
    return (1.0 if read is None else min(read, ROOM ** (1 / 3))) * said, read


def _on_head(rig, shape):
    """{mesh name: the share of each vertex's weight on the model's own head}: the first bone of its spine above the
    chest (the neck it turns on; the head joint itself when none is between) and every bone it carries. Not from the
    head joint alone: where a cap's bone sits on the head's, the skull mesh read is the cap's and the face hangs below
    the joint matched to the fighter's head (Ultimate Ness)."""
    spine = correspond.model_chains(rig, shape)[0]['spine']
    below, kids = [spine.bones[min(spine.anchors[1] + 1, spine.anchors[2])]], rig.children()
    for b in below:
        below += kids[b]
    share = rig.weights[:, below].sum(axis=1)
    import bpy
    counts = [len(bpy.data.objects[name].data.vertices) for name in rig.meshes]
    return {name: share[a:a + n] for name, a, n in zip(rig.meshes, np.cumsum([0] + counts), counts)}


def _ask(cache):
    from .. import ai

    def ask(q, p, s, d, route):
        return ai.ask(q, p, s, d, cache_dir=cache, route=route)
    return ask


if __name__ == '__main__':
    import os
    from .. import ai
    load_dir, fighter, out = sys.argv[1:4]
    r = swap(load_dir, fighter, out, _ask(ai.answers(os.environ.get('RIGSWAP_CACHE') or Path(out) / 'cache',
                                                     Path(out) / 'cache')),
             export='--no-export' not in sys.argv)
    print(json.dumps({k: r[k] for k in ('model', 'front', 'variants', 'chains')}, indent=1))
