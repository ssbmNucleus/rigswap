"""Choose the largest loose surfaces that fit the fighter's dynamics limit."""

from ..library import MAX_DYNAMICS, DYNAMICS_SOURCE


def select(rows, target):
    """Which loose surfaces fit in the game's room; the rest keep the build's skin."""
    room = max(0, MAX_DYNAMICS - target['existing_chains'])
    need = sum(len(r.get('chains') or [None]) for r in rows)   # an accessory of several chains uses as many
    report = dict(limit=MAX_DYNAMICS, limit_unit='dynamics chains per fighter', limit_source=DYNAMICS_SOURCE,
                  target=target, available=room, requested=need, asked=False, answer=None)
    if need <= room:
        report.update(selected=list(range(len(rows))), rigid=[])
        return report
    if room == 0:                   # a full fighter: nothing to choose, so nothing to ask
        report.update(selected=[], rigid=list(range(len(rows))), answer=None,
                      why='the fighter already uses every dynamics chain: no new chain, nothing asked')
        return report
    chosen = biggest(rows, room)
    report.update(selected=chosen, rigid=[i for i in range(len(rows)) if i not in chosen],
                  why='the biggest surfaces first, a twin with its twin, while their chains fit')
    return report


def biggest(rows, room):
    """Which of ``rows`` (accessories with their surface's ``area`` and, when one mirrors another, its ``twin``) fill
    ``room`` dynamics chains: the biggest surface shows the most swing, so the biggest first, a twin only with its twin
    (one feeler of a pair swinging and the other held reads as a fault); one that does not fit leaves room for smaller
    ones after it."""
    units, seen = [], set()
    for i in range(len(rows)):
        if i in seen:
            continue
        unit = [i] + ([rows[i]['twin']] if rows[i].get('twin') is not None and rows[rows[i]['twin']].get('twin') == i
                      else [])
        seen.update(unit)
        units.append(unit)
    chosen, used = [], 0
    for unit in sorted(units, key=lambda u: -sum(rows[i]['area'] for i in u)):
        need = sum(len(rows[i].get('chains') or [None]) for i in unit)
        if used + need <= room:
            chosen += unit
            used += need
    return sorted(chosen)
