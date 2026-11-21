"""A Melee costume's DObjs as the game counts them: the rules the export (write.py: fill_vanilla_dobjs) keeps to when
it lays a new model into the vanilla ones.

A costume's model is a joint tree. Each joint holds a chain of DObjs (drawn objects); a DObj has a material (an MObj,
with its textures, a chain of TObjs) and its geometry (PObjs). The game knows a DObj and a TObj by its place, never by
a name. What it needs, from the decomp (the doldecomp melee project, read 2026-10-10):

  The DObj list. When a fighter is made the game lists every DObj of the costume: the joints in tree order (a joint,
  then its children, then its next sibling; not into an instanced joint's children), each joint's chain in order. At
  most 124. (ft/ftparts.c ftParts_SetupParts and ftParts_80074194, the assert on line 466)

  Visibility. The fighter's own file, Pl<code>.dat, names DObjs by their place in that list. Its ftData root, +0x08,
  points to (ft/types.h struct ftData_x8): +0x00 the number of models; +0x04 for each costume, four sets, each for
  every model a list of options, each option a list of DObj places (u8). Set 0 is the high-detail model the game draws
  in play; set 1 the low-detail one, drawn for shadows (lb/lbshadow.c, ftDrawCommon_80081200), the magnifier bubble of
  a fighter off screen (if/ifmagnify.c, ftDrawCommon_80080C28) and the Fountain of Dreams reflection (gr/grizumi.c):
  each of these hides set 0 and shows set 1 while it draws. For each model the game shows one option and hides the
  others (ftParts_80074B6C), and every fighter shows option 0 of model 0 when it spawns (ftParts_80074A4C(gobj, 0, 0)
  in each ft*_Init.c). A costume with fewer DObjs than these lists name has the game hiding and showing entries past
  the end of its list.

  Objects. A model's option is one object, and every set lists its places under the same two numbers: the game shows
  each set's option by the one index it holds for that model (ftParts_80074B6C, fp->x5F4_arr[model].idx). So set 1's
  (model, option) is the low-detail form of set 0's. The options of a model are the forms of one object, swapped by
  the fighter's scripts (ftaction.c, ftParts_80074B0C): Kirby's body has its jump's form at model 0 option 1, its
  places 0-2 for his everyday 3-5 and 8-10 for 11-13, in the same order, which his jumps show; a form of the body the
  export leaves empty draws nothing while it is shown (``forms``). An option that is an object of its own stands in
  for the everyday one instead (Bowser's shell, Kirby's stones: facts/parts.py keeps them, each with its low-detail
  form).

  Outline. Game & Watch's model is black fill over an outline hull, a fifth set of the same objects
  (ftGw_Init_OnLoad, ftGw_Init.c: fp->x5AC.xC[4] = items[10]; ftdrawcommon.c shows and hides it with the others):
  HULL, read with the rest. It is one of the sets ``forms`` knows, so the hull's other options follow the first once
  the build fills it (export/flat.py, a detail of the build: export/__init__.py, DETAILS).

  Metal. A metal fighter (a Metal Box, or one always metal) has set 0 hidden and sets 2 and 3 shown
  (ftDrawCommon_800805C8, ftParts_800750C8). Set 2 names places in the fighter's own metal model: the joint tree at
  Pl<code>.dat's ftData +0x5C, whose DObjs the game hangs on the costume's joints in a list of their own (ft/ftmetal.c
  ft_800C85B8); Mario's is eight DObjs. Set 3 names places in the costume's list, drawn with it (Peach's: DObjs 52-54,
  81 and 84 and the options of models 3 and 4). The game does not fall back to set 0: a metal fighter is the vanilla
  metal model plus whatever the costume has at its set-3 places. Mario has no set 3, so metal Mario is PlMr.dat's metal
  Mario whatever the costume.

  Animated textures. +0x08 the number of textures the game animates (the eyes), +0x0C for each costume a list of that
  many u16, each a place in the list of every TObj of every DObj: the DObjs in order, each MObj's TObjs in order.
  ftAnim_80070200 (ft/ftanim.c, lines 1005-1020) looks each one up when the match loads; a costume whose TObjs run out
  first stops the game there: "can't find tobj!", the assert on ft/ftparts.c line 948 (ftParts_80075240). The TObj it
  finds must have a texture animation, or "can't find fighter texture anim!" (ftanim.c line 1236). TObjs get their
  animations from the costume file's own <root>_matanim_joint tree: matanim joints pair with the costume's joints as
  both trees are walked together, each one's list of material animations pairs with that joint's DObjs in order, and
  a texture animation goes to the TObj with its texture map id (sysdolphin jobj.c HSD_JObjAddAnimAll, dobj.c
  HSD_DObjAddAnimAll, tobj.c HSD_TObjAddAnim). A costume with no list of its own uses costume 0's; the export writes
  costume 0, Pl<code>Nr.dat.

  Mario (PlMr.dat and PlMrNr.dat, vanilla v1.02): 59 DObjs, all on the root joint, 60 TObjs. Model 0 option 0 of set 0
  is DObjs 0-36 and 53-58, set 1 is DObjs 37-52. The animated TObjs are 14 and 15: the only TObjs of DObjs 14 and 15
  (the eyes), whose material animations carry a texture animation of six images each.

  Loading. A DObj, its MObj, TObjs and PObjs are made from their file structures each time the game meets them
  (sysdolphin dobj.c HSD_DObjLoadDesc, mobj.c, tobj.c, pobj.c HSD_PObjLoadDesc): a PObj copies its fields and points
  at the file's vertex arrays and display lists, and finds its joints by the joint structures' addresses
  (HSD_IDGetData). Nothing writes back to the file's structures, so two DObjs may point at one MObj and one chain of
  PObjs: the game makes two DObjs that draw the same thing.

  The stack. The game loads a DObj's chain of PObjs by recursion, one level for each PObj (pobj.c PObjLoad calls
  HSD_PObjLoadDesc for the next before it returns), and a costume's DObjs the same way (dobj.c DObjLoad loads the
  next DObj first). The main thread's stack is 64 KB (ldscript.lcf: _stack_addr is 0x10000 above the small data, and
  the small data lies just below it), so a chain too long runs the stack into the game's globals and the match hangs
  as it loads, on whatever broke first (an assert in texp.c or in lbarchive.c). See MAX_POBJS.

How the export lays the build into them: write.py's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
import struct

from ... import Failed
from ..dat import JOINT_INSTANCE, Dat, joints, joint_dobjs


MAX_DOBJS = 124                             # ftparts.c line 466
# A conservative export policy, NOT a derived game limit. ldscript.lcf reserves 0x10000 bytes for the main stack;
# pobj.c PObjLoad -> HSD_PObjLoadDesc recurses along the chain. The previous DOL inspection reported 0x18-byte
# frames for each function (48 bytes per PObj, and 48 per DObj). Even accepting those sizes, allocating the entire
# stack to 124 DObjs and PObjs gives floor(65536 / 48) - 124 = 1241 PObjs, not 512. Caller and joint recursion,
# class callbacks and other live frames also consume stack; their worst-case depth has not been established.
# The old 512 was chosen below a successful 725-PObj load at DObj place 37, subtracting 86 later DObjs and an
# arbitrary 127-frame reserve. That experiment does not prove a limit for every fighter. Keep the existing policy
# until a worst-case stack bound is measured; do not promote it to facts/ or claim that arithmetic derives 512.
MAX_POBJS = 512
SETS = 4
HIGH_DETAIL, LOW_DETAIL, OTHER_LIST, METAL, HULL = range(SETS + 1)
HULL_ITEMS = {'PlGw': 10}                   # which of ftData +0x48's items is the outline set, of the one fighter with one
# src/sysdolphin/baselib/pobj.c:1131 loads at most ten envelope matrices; :1154 blends all joints of each envelope.
MAX_ENVELOPES = 10

# A DObj that keeps its place but draws nothing gets a blank material: as many TObjs as vanilla's, each an 8x8 I4
# image of zeros that passes the colour and alpha through untouched (the dummy texture HSDRawViewer's ModelImporter
# CreateDummyTObj writes). No PObj, so nothing is drawn; the game still sets the material up each frame
# (dobj.c HSD_DObjDisp), so it is a whole one.
RENDER_DIFFUSE = 1 << 2
RENDER_TEX0 = 1 << 4
TOBJ_PASS = (1 << 4) | (6 << 16) | (5 << 20)    # diffuse light map, colour PASS, alpha PASS
GX_TG_TEX0 = 4
GX_LINEAR = 1
GX_I4 = 0
GX_TEXTURE_MAPS = 8                        # GX_TEXMAP0 through GX_TEXMAP7
IMAGE_ALIGN = 32                                # GX reads texture data from 32-byte boundaries


@dataclass(frozen=True)
class Slot:
    """One DObj of a costume, at its place in the game's list."""
    joint: int                  # its joint's place in the joint tree
    local: int                  # its place in that joint's chain
    textures: tuple = ()        # each of its TObjs' texture map id, in order
    drawn: bool = True          # it has geometry
    animated: tuple = ()        # texture map ids the material animation gives a texture animation here
    material_animated: bool = False     # the material animation also animates its colours or render mode


@dataclass(frozen=True)
class FighterTables:
    """What a fighter's Pl<code>.dat says about a costume's DObjs and TObjs, for one costume."""
    animated: tuple = ()        # places in the TObj list the game animates
    body: tuple | None = None   # DObj places drawn at spawn: set 0, model 0, option 0 (None: no visibility tables)
    elsewhere: frozenset = frozenset()  # DObj places another option/model of set 0 or 1, or any of set 3 names (shown or hidden
                                        # apart from the body)
    low: tuple = ()             # DObj places drawn for shadows and the magnifier: set 1, model 0, option 0, less any
                                # place another option or model of set 0 or 1 names
    reserved: frozenset = frozenset()  # equipment and replacement objects kept in their vanilla places
    options: dict = field(default_factory=dict)     # {(set, model, option): DObj places}: all the tables say


def _material_animations(d, symbol, joint_list):
    """{joint offset: [material animation offset]} as the game pairs them: the matanim tree walked with the joint
    tree, child with child and sibling with sibling (HSD_JObjAddAnimAll)."""
    root = d.root(symbol[:-len('_joint')] + '_matanim_joint') if symbol.endswith('_joint') else None
    out = {}

    def walk(j, m):
        out[j] = d.chain(d.ptr(m + 0x08), 0x00) if m is not None else []
        if d.u32(j + 4) & JOINT_INSTANCE:
            return
        child, mchild = d.ptr(j + 0x08), (d.ptr(m) if m is not None else None)
        while child is not None:
            walk(child, mchild)
            child, mchild = d.ptr(child + 0x0C), (d.ptr(mchild + 0x04) if mchild is not None else None)
    if joint_list:
        walk(joint_list[0], root)
    return out


def layout(path, symbol):
    """[Slot] for every DObj of the costume in ``path`` under root ``symbol``, in the game's order."""
    d = path if isinstance(path, Dat) else Dat.read(path)
    tree = joints(d, symbol)
    animations = _material_animations(d, symbol, tree)
    slots = []
    for k, j in enumerate(tree):
        mats = animations.get(j, [])
        for n, dobj in enumerate(joint_dobjs(d, j)):
            mobj = d.ptr(dobj + 0x08)
            tobjs = d.chain(d.ptr(mobj + 0x08), 0x04) if mobj is not None else []
            mat = mats[n] if n < len(mats) else None
            texture_anims = d.chain(d.ptr(mat + 0x08), 0x00) if mat is not None else []
            slots.append(Slot(joint=k, local=n, textures=tuple(d.u32(t + 0x08) for t in tobjs),
                              drawn=d.ptr(dobj + 0x0C) is not None,
                              animated=tuple(d.u32(t + 0x04) for t in texture_anims if d.ptr(t + 0x08) is not None),
                              material_animated=mat is not None and (d.ptr(mat + 0x04) is not None
                                                                     or d.ptr(mat + 0x0C) is not None)))
    return slots


def fighter_tables(path, costume=0):
    """The animated TObj places and the visibility tables in a fighter's Pl<code>.dat, for ``costume`` (a costume
    with no table of its own uses costume 0's, as the game does)."""
    d = Dat.read(path)
    root = next((offset for name, offset in d.roots if name.startswith('ftData')), None)
    if root is None:
        raise Failed(f'{Path(path).name} has no ftData root')
    x8 = d.ptr(root + 0x08)
    if x8 is None:
        return FighterTables()
    models, sets = d.u32(x8), d.ptr(x8 + 0x04)
    count, lists = d.u32(x8 + 0x08), d.ptr(x8 + 0x0C)

    def own_or_first(table, stride, k=0):
        at = d.ptr(table + costume * stride + 4 * k)
        return at if at is not None else d.ptr(table + 4 * k)

    animated = ()
    if count:
        at = own_or_first(lists, 4) if lists is not None else None
        if at is None:
            raise Failed(f'{Path(path).name} animates {count} textures and lists none')
        animated = tuple(d.u16(at + 2 * i) for i in range(count))
    if sets is None or not models:
        return FighterTables(animated=animated)
    options = {}        # (set, model, option) -> DObj places

    def read(k, lookup):
        for m in range(models if lookup is not None else 0):
            n, opts = d.u32(lookup + 8 * m), d.ptr(lookup + 8 * m + 4)
            for o in range(n if opts is not None else 0):
                size, places = d.u32(opts + 8 * o), d.ptr(opts + 8 * o + 4)
                options[k, m, o] = tuple(d.data[places + i] for i in range(size)) if places is not None else ()
    for k in range(SETS):
        read(k, own_or_first(sets, 4 * SETS, k))
    if Path(path).stem in HULL_ITEMS:
        read(HULL, d.ptr(d.ptr(root + 0x48) + 4 * HULL_ITEMS[Path(path).stem]))

    def named_but_by(key):
        # Set 3 addresses the costume list too (src/melee/ft/ftparts.c:741); set 2 addresses the separate metal list.
        return {p for k, places in options.items() if k != key and k[0] in (HIGH_DETAIL, LOW_DETAIL, METAL)
                for p in places}
    body = options.get((HIGH_DETAIL, 0, 0), ())
    elsewhere = frozenset(named_but_by((HIGH_DETAIL, 0, 0)))
    low = tuple(p for p in options.get((LOW_DETAIL, 0, 0), ()) if p not in named_but_by((LOW_DETAIL, 0, 0)))
    return FighterTables(animated=animated, body=body, elsewhere=elsewhere, low=low, options=options)


def forms(tables):
    """{DObj place: the place it is another form of}: what the game draws when it shows an option other than 0.

    A model's options are the forms of one object (the module's docstring), so an option draws what the model's option
    0 draws, place for place in the order the tables list them, in each set of the object (high detail, low detail,
    the outline). Neither may be a kept part (``tables.reserved``): that is an object of its own, in vanilla's places,
    not a form of the build. The fighter's own tables list a form's places on the joints of the ones it stands for, but
    for the last of Game & Watch's heads, on joint 20, standing for a place no build fills (facts/flat.py)."""
    out = {}
    for (detail, model, option), places in tables.options.items():
        first = tables.options.get((detail, model, 0), ())
        if detail in (HIGH_DETAIL, LOW_DETAIL, HULL) and option and not tables.reserved.intersection(places + first):
            out.update(zip(places, first))
    return out


def texture_places(slots, animated):
    """For each place in ``animated`` (a TObj's place in the list of every TObj), (DObj place, its TObj's place in
    that DObj), or None when the TObjs run out first."""
    walk = [(i, n) for i, s in enumerate(slots) for n in range(len(s.textures))]
    return [walk[a] if a < len(walk) else None for a in animated]


def layout_differs(vanilla, written, animated):
    """Why the written costume's DObjs would not work where the vanilla's do, or None: the same number of DObjs, each
    on the same joint at the same place in its chain, and each animated TObj at the same DObj and the same place in
    it, with the same texture map id, given a texture animation by the material animation."""
    if len(written) != len(vanilla):
        return f'{len(written)} DObjs; the vanilla costume has {len(vanilla)}'
    for i, (v, w) in enumerate(zip(vanilla, written)):
        if (v.joint, v.local) != (w.joint, w.local):
            return (f'DObj {i} is number {w.local} on joint {w.joint}; vanilla\'s is number {v.local} on joint '
                    f'{v.joint}')
    total = sum(len(s.textures) for s in written)
    for a, v, w in zip(animated, texture_places(vanilla, animated), texture_places(written, animated)):
        if v is None:
            return f'the game animates TObj {a} and the vanilla costume has none there either'
        if w is None:
            return f'the game animates TObj {a} and the costume has {total}'
        if w != v:
            return (f'the game animates TObj {a}: TObj {w[1]} of DObj {w[0]}, which in vanilla is TObj {v[1]} of '
                    f'DObj {v[0]}')
        texture = written[w[0]].textures[w[1]]
        if texture != vanilla[v[0]].textures[v[1]]:
            return f'TObj {a} has texture map id {texture}; vanilla\'s has {vanilla[v[0]].textures[v[1]]}'
        if texture not in written[w[0]].animated:
            return f'the game animates TObj {a} (TObj {w[1]} of DObj {w[0]}) and it has no texture animation'
    return None


# ---- laying the build into the vanilla DObjs -----------------------------------------------------------------------

def choose_places(vanilla, tables, build):
    """The vanilla DObj place each of the build's DObjs takes; ``build`` is [(its joint, its number of TObjs)].

    A place qualifies when the game draws it at spawn (set 0, model 0, option 0; every DObj when the fighter has no
    visibility tables) and no other option or model of sets 0 and 1, or any option of set 3, names it, on the same joint as the
    build's DObj (the import puts a DObj on the root joint, or on the one joint a mesh named ..SINGLE.. is wholly bound
    to, and draws it in that joint's space), and when the material animation leaves it alone (a texture or colour
    animation there would play on the build's material). Prefer the same TObj count before animated textures; a
    smaller count also fits: blank DObjs carry the unused TObjs, else the material is padded to its place's
    (texture_counts). Places after the last animated TObj go first, then the ones before; the most constrained DObjs
    first.
    """
    body = range(len(vanilla)) if tables.body is None else tables.body
    open_, last = _open_places(vanilla, tables, [i for i in body if i not in tables.elsewhere])
    order = sorted(i for i in open_ if i > last) + sorted(i for i in open_ if i <= last)

    def fits(n, exact=True):
        joint, need = build[n]
        return [i for i in order if vanilla[i].joint == joint and
                (i > last or (len(vanilla[i].textures) == need if exact else len(vanilla[i].textures) >= need))]
    places = {}
    for n in sorted(range(len(build)), key=lambda n: (len(fits(n)), n)):
        place = next((i for i in fits(n) + fits(n, exact=False) if i not in places.values()), None)
        if place is None:
            joint, need = build[n]
            mine = sum(1 for j, _ in build if j == joint)
            free = sum(1 for i in open_ if vanilla[i].joint == joint)
            raise Failed(f'the build has {mine} DObjs (one for each mesh and material) on joint {joint} and the '
                         f'vanilla costume has {free} places for them there (drawn at spawn, not animated); DObj {n} '
                         f'({need} textures) is left without one')
        places[n] = place
    texture_counts(vanilla, tables, {p: build[n][1] for n, p in places.items()})
    return [places[n] for n in range(len(build))]


def texture_counts(vanilla, tables, drawn):
    """TObj counts by place, retaining every animated TObj's global index; ``drawn``: {place: the TObjs its DObj's
    material has}.

    An unused vanilla texture slot belongs to the enumeration, not to the incoming material. Move that slot to a
    blank DObj in the same interval between animated DObjs. Kept and animated materials are never altered. Blank
    materials use consecutive GX texture maps (eight hardware maps). When the interval has no blank DObj to carry
    them (the build takes every place before the animated DObj: Pikachu has 13 places drawn at spawn, its eyes the
    6th and its cheeks the 13th, and none after), the drawn DObjs that lack TObjs are counted up to their place's
    own: the export pads their materials with pass-through TObjs (padded_material), as HSDRawViewer's import pads a
    material to a texture count (ModelImporter.cs ForceTextureCount, CreateDummyTObj). An untextured material (the
    import's white) at a place with one TObj, or a one-texture material at a place with two, then keeps the count
    and draws as it did. A drawn count is never raised past its place's: places before an animated DObj take
    materials with no more TObjs than vanilla's there (choose_places, choose_low_places)."""
    counts = [drawn.get(i, len(s.textures)) for i, s in enumerate(vanilla)]
    animated = sorted({p[0] for p in texture_places(vanilla, tables.animated) if p is not None})
    start = 0
    for end in animated:
        missing = sum(len(vanilla[i].textures) - counts[i] for i in range(start, end))
        for i in range(start, end):
            if i in drawn or i in tables.reserved or vanilla[i].animated or vanilla[i].material_animated:
                continue
            added = min(missing, GX_TEXTURE_MAPS - counts[i])
            counts[i] += added
            missing -= added
        for i in range(start, end):         # no blank DObj left to carry them: the drawn ones are padded
            if missing > 0 and i in drawn:
                added = min(missing, len(vanilla[i].textures) - counts[i])
                counts[i] += max(added, 0)
                missing -= max(added, 0)
        if missing:
            raise Failed(f'no blank texture slots before animated DObj {end}: {missing} TObjs cannot keep their indices')
        start = end + 1
    return counts


def padded_material(d, mobj, count, image, cache):
    """A copy of material ``mobj`` with ``count`` TObjs: its own, copied, then pass-through TObjs drawing nothing (the
    blank material's: an 8x8 I4 ``image`` of zeros, colour and alpha PASS), each on the next GX texture map, and the
    material's render mode told of them. The copy, so a place that draws ``mobj`` with its own count keeps it."""
    if (mobj, count) in cache:
        return cache[mobj, count]
    own = d.chain(d.ptr(mobj + 0x08), 0x04)
    copies = [_copy(d, t, 0x5C) for t in own]
    for k in range(len(own), count):
        tobj = bytearray(0x5C)
        struct.pack_into('>2I', tobj, 0x08, k, GX_TG_TEX0)
        struct.pack_into('>3f', tobj, 0x1C, 1., 1., 1.)                  # scale
        struct.pack_into('>2B', tobj, 0x3C, 1, 1)                        # repeat
        struct.pack_into('>IfI', tobj, 0x40, TOBJ_PASS, 1., GX_LINEAR)   # flags, blending, magnification filter
        copies.append(d.add(tobj))
        d.point(copies[-1] + 0x4C, image)
    for a, b in zip(copies, copies[1:] + [None]):
        d.point(a + 0x04, b)
    at = _copy(d, mobj, 0x18)
    d.point(at + 0x08, copies[0] if copies else None)
    struct.pack_into('>I', d.data, at + 0x04, d.u32(mobj + 0x04) | sum(RENDER_TEX0 << i for i in range(count)))
    cache[mobj, count] = at
    return at


def _copy(d, at, size):
    """A copy of the ``size`` bytes at ``at``, its pointers pointing where the original's do."""
    new = d.add(bytes(d.data[at:at + size]))
    for field in range(0, size, 4):
        if at + field in d.relocations:
            d.point(new + field, d.ptr(at + field))
    return new


def _open_places(vanilla, tables, candidates):
    """(the places of ``candidates`` the material animation leaves alone, the place of the last animated TObj's DObj
    or -1)."""
    positions = {i for i, _ in (p for p in texture_places(vanilla, tables.animated) if p is not None)}
    open_ = [i for i in candidates if i not in tables.reserved and i < len(vanilla) and not vanilla[i].animated and not vanilla[i].material_animated
             and i not in positions]
    return open_, max(positions, default=-1)


def choose_low_places(vanilla, tables, build, most=None):
    """Which of the build's DObjs each low-detail place draws: ``build`` is [(its joint, its number of TObjs, its size,
    its look, its number of PObjs)], size any measure of how much it draws, look equal for DObjs that draw with the
    same material. -> ({place: [the build's DObjs it draws; the first one's material draws them all]}, [DObjs no place
    draws])

    A place qualifies the way a high-detail one does, from the low-detail list (``FighterTables.low``) instead of the
    body: the material animation leaves it alone, it is on the same joint as the DObjs it draws, and a place before the
    last animated TObj takes a material with no more TObjs than vanilla's there (texture_counts moves the unused
    slots to blank DObjs). On each joint the DObjs that look
    the same go to one place, the group that draws most first, places after the last animated TObj first; a place
    draws at most ``most`` PObjs (MAX_POBJS: the game loads them by recursion on its stack), so a group with more is
    split, in order, over as many places. A group left without a place (more materials than places, or none with its
    number of TObjs) is drawn from the first place given on its joint that has room for its PObjs, in that place's
    material: right in shadows, the wrong texture in the magnifier. A DObj with no place to go is left out, as are the
    DObjs of a joint with no low-detail place."""
    most = MAX_POBJS if most is None else most
    open_, last = _open_places(vanilla, tables, tables.low)
    order = sorted(i for i in open_ if i > last) + sorted(i for i in open_ if i <= last)
    places, left_out = {}, []

    def pobjs(ns):
        return sum(build[n][4] for n in ns)
    for joint in sorted({b[0] for b in build}):
        groups = {}
        for n, b in enumerate(build):
            if b[0] == joint:
                groups.setdefault(b[3], []).append(n)
        groups = sorted(groups.values(), key=lambda g: (-sum(build[n][2] for n in g), g[0]))
        free = [i for i in order if vanilla[i].joint == joint]
        given, unplaced = [], []
        for group in groups:
            need = build[group[0]][1]
            for run in _runs(group, [build[n][4] for n in group], most):
                compatible = [i for i in free if i > last or len(vanilla[i].textures) == need]
                compatible += [i for i in free if len(vanilla[i].textures) > need]
                place = next(iter(compatible), None)
                if place is None:
                    unplaced += run
                    continue
                free.remove(place)
                places[place] = list(run)
                given.append(place)
        for n in unplaced:
            room = next((p for p in given if pobjs(places[p]) + build[n][4] <= most), None)
            if room is None:
                left_out.append(n)
            else:
                places[room].append(n)
    return places, sorted(left_out)


def _runs(items, sizes, most):
    """``items`` in order, cut into runs whose ``sizes`` add up to at most ``most`` (an item larger than ``most`` alone
    in its run)."""
    runs, total = [], 0
    for item, size in zip(items, sizes):
        if not runs or total + size > most:
            runs.append([])
            total = 0
        runs[-1].append(item)
        total += size
    return runs


def split_long_chains(d, build, most=None):
    """The build's DObjs, [(joint, DObj offset)], with each one whose chain holds more than ``most`` PObjs (MAX_POBJS)
    cut into as few chains of near equal length as keep each within it: the first stays the DObj's, each other one
    gets a new DObj with the same MObj, so they take places of their own and draw with the one material.
    -> ([(joint, DObj offset)], {the import's DObj number: the chains it became} for those cut)"""
    most = MAX_POBJS if most is None else most
    out, cut = [], {}
    for n, (k, dobj) in enumerate(build):
        chain = d.chain(d.ptr(dobj + 0x0C), 0x04)
        if len(chain) <= most:
            out.append((k, dobj))
            continue
        count = -(-len(chain) // most)
        ends = list(accumulate(len(chain) // count + (i < len(chain) % count) for i in range(count)))
        parts = [chain[a:b] for a, b in zip([0, *ends[:-1]], ends)]
        for i, part in enumerate(parts):
            d.point(part[-1] + 0x04, None)
            if i == 0:
                out.append((k, dobj))
                continue
            unit = d.add(bytes(0x10))
            d.point(unit + 0x08, d.ptr(dobj + 0x08))
            d.point(unit + 0x0C, part[0])
            out.append((k, unit))
        cut[n] = [len(part) for part in parts]
    return out, cut


def _blank_material(d, textures, image, material, cache):
    """A material for a DObj that draws nothing: one blank TObj for each texture map id in ``textures``."""
    if textures in cache:
        return cache[textures]
    head = None
    for texture in reversed(textures):
        tobj = bytearray(0x5C)
        struct.pack_into('>2I', tobj, 0x08, texture, GX_TG_TEX0)
        struct.pack_into('>3f', tobj, 0x1C, 1., 1., 1.)                  # scale
        struct.pack_into('>2B', tobj, 0x3C, 1, 1)                        # repeat
        struct.pack_into('>IfI', tobj, 0x40, TOBJ_PASS, 1., GX_LINEAR)   # flags, blending, magnification filter
        at = d.add(tobj)
        d.point(at + 0x04, head)
        d.point(at + 0x4C, image)
        head = at
    mobj = bytearray(0x18)
    struct.pack_into('>I', mobj, 0x04, RENDER_DIFFUSE | sum(RENDER_TEX0 << i for i in range(len(textures))))
    at = d.add(mobj)
    d.point(at + 0x08, head)
    d.point(at + 0x0C, material)
    cache[textures] = at
    return at


def _joined_pobjs(d, dobjs):
    """(the first PObj of a chain drawing every PObj of ``dobjs`` in order, the bytes the copies take with their
    pointers' relocation entries): the last DObj's own chain, after copies of the others' PObjs, each copy pointing at
    the same vertices, display list and joints."""
    head, added = d.ptr(dobjs[-1] + 0x0C), 0
    for dobj in reversed(dobjs[:-1]):
        for pobj in reversed(d.chain(d.ptr(dobj + 0x0C), 0x04)):
            copy = d.add(bytes(d.data[pobj:pobj + 0x18]))
            for field in (0x00, 0x08, 0x10, 0x14):        # class name, vertices, display list, joint or envelopes
                if pobj + field in d.relocations:
                    d.relocations.add(copy + field)
            d.point(copy + 0x04, head)
            added += 0x18 + 4 * sum(copy + field in d.relocations for field in range(0, 0x18, 4))
            head = copy
    return head, added
