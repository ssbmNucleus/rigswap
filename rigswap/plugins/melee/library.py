"""Accessory dynamics by chain kind, in degrees before engine serialization.

Source: untangle app/backend/randall/model_import/studio/physics/dynamics.py,
TEMPLATES and KIND_TEMPLATES (2026-10-12), measured from the vanilla files below.
No size heuristics or per-model exceptions. Other hanging kinds
use 'other'; fingers are inside the hand (library.moves_as.INSIDE), not accessories.
"""
from math import radians

# A count of dynamics descriptors (chains), not joints or bytes. m-ex adds the
# fighter's PlXX.dat chains to every costume's accessory DynamicCount, then
# cmpwi r3,10; bge DyanmicsOver. Nine is the largest accepted total.
# Source: akaneia/m-ex, asm/m-ex/mexCostume/Init.asm (Dynamics_Loop);
# untangle shared/modellab/physics_accessory.py:MEX_MAX_DYNAMICS and
# randall/model_import/studio/physics/chains.py:room. The old importer's separate
# MAX_MOVING_JOINTS=45 is a vanilla maximum, not this engine limit.
MAX_DYNAMICS = 9
DYNAMICS_SOURCE = 'https://github.com/akaneia/m-ex/blob/master/asm/m-ex/mexCostume/Init.asm'


def _row(limit, speed, damping=.5, axis=360.):
    return dict(inertia=1., spring=0., limit=limit, speed=speed, damping=damping, axis=axis)


PHYSICS = {
    'hair': dict(source='PlZd.dat, chain at bone 90 (zelda_hair)', gravity=.07,
                 rows=[_row(45, 6, 1), _row(45, 4, .8), _row(60, 3), _row(60, 3)]),
    'cape': dict(source='PlFe.dat, chain at bone 47 (roy_cape)', gravity=.1745329,
                 rows=[_row(60, 5), _row(60, 4), _row(60, 3.5), _row(60, 8)]),
    'tail': dict(source='PlFx.dat, chain at bone 17 (fox_tail)', gravity=.0436332,
                 rows=[_row(45, 3, axis=180.) for _ in range(4)]),
    'other': dict(source='PlFe.dat, chain at bone 47 (roy_cape, old importer fallback)', gravity=.1745329,
                  rows=[_row(60, 5), _row(60, 4), _row(60, 3.5), _row(60, 8)]),
}


def dynamics(kind, count, bone=3, springs=None):
    """One SBM_DynamicDesc for ``count`` moving joints and their terminal tip. ``springs``: per moving joint, how far
    it turns back toward its rest each frame (radians), in place of the template's.

    Resampling keeps both ends of the source template, as the old importer does.
    The engine initializes exactly the declared joints (lb_8000FD48) and updates
    only entries with a successor (lb_8001044C). Include the tip's parameter row
    so the last moving joint has a simulated segment, even for a one-bone chain.
    PARAM1/2 = inertia multiplier / engine scale; PARAM3 = gravity. Joint fields:
    inertia, spring, natural quaternion (zero), swing limit, positive XYZ limits,
    negative XYZ limits, damping and maximum angular change per frame. The spring is
    lb_8001044C's convergence (DynamicsData +0x50): each frame, after gravity and
    the swing's momentum, the link turns back toward its rest direction by that
    much, or onto it when nearer.
    """
    if count < 1:
        raise ValueError('an accessory chain needs a moving bone')
    template = PHYSICS.get(kind, PHYSICS['other'])
    rows = template['rows']
    result = []
    for i in range(count):
        r = rows[round(i * (len(rows) - 1) / (count - 1)) if count > 1 else 0]
        axis = radians(r['axis'])
        spring = springs[i] if springs is not None else radians(r['spring'])
        result.append([r['inertia'], spring, 0., 0., 0., 0., radians(r['limit']),
                       axis, axis, axis, -axis, -axis, -axis, radians(r['damping']), radians(r['speed'])])
    result.append(result[-1].copy())  # endpoint: initialized, but never rotated by dynamics
    return dict(bone=bone, PARAM1=1., PARAM2=1., PARAM3=template['gravity'], joints=result)
