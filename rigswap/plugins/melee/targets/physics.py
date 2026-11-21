"""Existing dynamics counted from the target's own files, including costume accessories."""
import hashlib
import json
from pathlib import Path

from .bones import bone_tables, dynamics


def classify(target, weights):
    """Account for exact positive skin influences on each target dynamics chain.

    This is evidence, not permission to subtract slots: fighter dynamics live in
    Pl<code>.dat and m-ex's costume descriptor cannot override that table.
    """
    carried = {}
    for vertex in weights:
        for bone, weight in vertex.items():
            if weight > 0:
                carried[bone] = carried.get(bone, 0) + 1
    alive, dead = [], []
    for index, chain in enumerate(target['chains']):
        bones = [f'JOBJ_{n}' for n in chain]
        weighted = {bone: carried[bone] for bone in bones if bone in carried}
        row = dict(index=index, bones=bones, weighted_vertices=weighted,
                   why='positive skin weight on at least one chain bone' if weighted else
                   'zero skin weight on every chain bone')
        (alive if weighted else dead).append(row)
    return dict(target, alive_chains=alive, dead_chains=dead,
                weight_rule='strictly positive weights on vertices used by surface faces; no threshold',
                reclaimable_chains=len(dead), reclaimed_chains=0,
                reclamation_blocker=('fighter dynamics are in Pl<code>.dat; the m-ex costume format has no '
                                     'override for that table') if dead else None)


def record(f, files):
    from ..export.physics import definitions
    files = Path(files)
    fighter, costume = files / f'Pl{f.code}.dat', files / f'Pl{f.code}Nr.dat'
    common = files / 'PlCo.dat'
    skipped = bone_tables(common)[f.name]['skipped'] if common.is_file() else ()
    chains = dynamics(fighter, skipped)
    added = sum(len(a['dynamics']) for a in definitions(costume))
    return dict(existing_chains=len(chains) + added, fighter_chains=len(chains), costume_chains=added,
                moving_joints=sum(map(len, chains)), chains=chains,
                skipped_parts=list(skipped),
                source=f'{fighter.name}: ftData +0x2C, descriptor count; {costume.name}: mexCostume DynamicCount',
                sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (fighter, costume, *([common] if common.is_file() else []))})


def write(f, files, targets):
    value = record(f, files)
    (Path(targets) / f'{f.name}.physics.json').write_text(json.dumps(value, indent=1), encoding='utf-8')
    return value
