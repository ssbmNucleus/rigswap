"""export/dobjs.py: the costume's DObjs as the game counts them."""
import os
from pathlib import Path

import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee.export import dobjs as melee_dobjs
from rigswap.plugins.melee.export.dobjs import FighterTables, Slot
from rigswap.plugins.melee.facts import fighters, parts as facts_parts
from tests.plugins.melee.test_dat import _fighter_file


# Mario as PlMr.dat and PlMrNr.dat have him (vanilla v1.02, read 2026-10-10): 59 DObjs, all on the root joint, one TObj
# each but DObjs 32 and 53 (two) and 58 (none); the game animates TObjs 14 and 15, the eyes; set 0 model 0 option 0 is
# DObjs 0-36 and 53-58, set 1 (low detail) 37-52.
MARIO_ANIMATED = (14, 15)


MARIO_BODY = tuple(range(37)) + tuple(range(53, 59))


MARIO_LOW = tuple(range(37, 53))


def _mario_slots():
    textures = {32: (0, 1), 53: (0, 1), 58: ()}
    return [Slot(joint=0, local=i, textures=textures.get(i, (0,)), animated=(0,) if i in MARIO_ANIMATED else ())
            for i in range(59)]


def _mario_tables():
    return FighterTables(animated=MARIO_ANIMATED, body=MARIO_BODY, elsewhere=frozenset(MARIO_LOW), low=MARIO_LOW)


def test_the_game_finds_an_animated_texture_by_counting_every_tobj_before_it():
    mario = _mario_slots()
    assert melee_dobjs.texture_places(mario, MARIO_ANIMATED) == [(14, 0), (15, 0)]
    # a second TObj on DObj 0 moves every later TObj one on: TObj 14 is then DObj 13's
    shifted = [Slot(0, 0, (0, 1))] + mario[1:]
    assert melee_dobjs.texture_places(shifted, MARIO_ANIMATED) == [(13, 0), (14, 0)]
    # the export as it was: three DObjs, three TObjs, and the game runs out before TObj 14 ("can't find tobj!")
    assert melee_dobjs.texture_places(mario[:3], MARIO_ANIMATED) == [None, None]


def test_the_dobj_check_on_a_synthetic_list():
    vanilla = _mario_slots()
    same = [Slot(s.joint, s.local, s.textures, drawn=i in (16, 17, 18), animated=s.animated)
            for i, s in enumerate(vanilla)]
    assert melee_dobjs.layout_differs(vanilla, same, MARIO_ANIMATED) is None
    assert '3 DObjs' in melee_dobjs.layout_differs(vanilla, vanilla[:3], MARIO_ANIMATED)
    assert '58 DObjs' in melee_dobjs.layout_differs(vanilla, vanilla[:58], MARIO_ANIMATED)
    moved = vanilla[:20] + [Slot(5, 0, (0,))] + vanilla[21:]
    assert 'DObj 20' in melee_dobjs.layout_differs(vanilla, moved, MARIO_ANIMATED)
    extra_texture = [Slot(0, 0, (0, 1))] + vanilla[1:]
    assert 'TObj 0 of DObj 14' in melee_dobjs.layout_differs(vanilla, extra_texture, MARIO_ANIMATED)
    no_animation = vanilla[:15] + [Slot(0, 15, (0,))] + vanilla[16:]
    assert 'no texture animation' in melee_dobjs.layout_differs(vanilla, no_animation, MARIO_ANIMATED)
    other_map = vanilla[:14] + [Slot(0, 14, (1,), animated=(1,))] + vanilla[15:]
    assert 'texture map id 1' in melee_dobjs.layout_differs(vanilla, other_map, MARIO_ANIMATED)


def test_the_builds_dobjs_take_places_the_game_draws_at_spawn():
    mario, tables = _mario_slots(), _mario_tables()
    # after the eyes first, in order
    assert melee_dobjs.choose_places(mario, tables, [(0, 1)] * 3) == [16, 17, 18]
    # every place drawn at spawn but the eyes; before the eyes only with the same number of TObjs
    places = melee_dobjs.choose_places(mario, tables, [(0, 1)] * 41)
    assert sorted(places) == sorted(set(MARIO_BODY) - set(MARIO_ANIMATED))
    assert not set(places) & set(MARIO_LOW)
    # a DObj without a texture can only go after the eyes, so it chooses before the others
    assert melee_dobjs.choose_places(mario, tables, [(0, 1)] * 27 + [(0, 0)])[-1] == 16
    with pytest.raises(Failed, match='42 DObjs .* on joint 0 .* 41 places'):
        melee_dobjs.choose_places(mario, tables, [(0, 1)] * 42)
    with pytest.raises(Failed, match='on joint 7 .* 0 places'):
        melee_dobjs.choose_places(mario, tables, [(7, 1)])
    # no visibility tables: every DObj is drawn
    assert melee_dobjs.choose_places(mario[:3], FighterTables(), [(0, 1)]) == [0]


def test_the_low_detail_places_draw_the_builds_dobjs_again():
    mario, tables = _mario_slots(), _mario_tables()
    # [(joint, TObjs, size, look, PObjs)]: three materials, the one that draws most first, after the eyes, in order
    low, left_out = melee_dobjs.choose_low_places(mario, tables, [(0, 1, 100, 'a', 1), (0, 1, 900, 'b', 1),
                                                                  (0, 1, 50, 'c', 1)])
    assert low == {37: [1], 38: [0], 39: [2]} and left_out == []
    # DObjs with one material (one mesh's pieces, or meshes sharing it) draw from one place
    low, _ = melee_dobjs.choose_low_places(mario, tables, [(0, 1, 100, 'a', 1), (0, 1, 900, 'b', 1),
                                                           (0, 1, 50, 'a', 1)])
    assert low == {37: [1], 38: [0, 2]}
    # more materials than places: the rest draw from the first place, in its material
    low, left_out = melee_dobjs.choose_low_places(mario, tables, [(0, 1, 20 - n, n, 1) for n in range(20)])
    assert sorted(low) == list(MARIO_LOW) and low[37] == [0, 16, 17, 18, 19] and left_out == []
    assert sorted(n for drawn in low.values() for n in drawn) == list(range(20))
    # a joint with no low-detail place leaves its DObjs out; the high-detail places are never used
    low, left_out = melee_dobjs.choose_low_places(mario, tables, [(0, 1, 10, 'a', 1), (7, 1, 10, 'b', 1)])
    assert low == {37: [0]} and left_out == [1]
    assert not set(low) & set(MARIO_BODY)
    # before the last animated TObj a place takes only a material with vanilla's number of TObjs there
    early = FighterTables(animated=MARIO_ANIMATED, body=tuple(range(30, 59)), low=(10, 11, 12))
    assert melee_dobjs.choose_low_places(mario, early, [(0, 2, 10, 'a', 1), (0, 1, 5, 'b', 1)]) == ({10: [1, 0]}, [])
    # no visibility tables: no low-detail set to fill
    assert melee_dobjs.choose_low_places(mario, FighterTables(), [(0, 1, 10, 'a', 1)]) == ({}, [0])


def test_untextured_objects_keep_later_animation_indices_using_blank_places():
    slots = [Slot(0, n, (0,)) for n in range(9)] + [Slot(1, 0, (0,), animated=(0,))]
    tables = FighterTables(animated=(9,), body=tuple(range(8)))
    build = [(0, 1), (0, 1), (0, 0), (0, 0), (0, 1), (0, 1), (0, 0), (0, 1)]
    places = melee_dobjs.choose_places(slots, tables, build)
    assert len(set(places)) == len(build)
    counts = melee_dobjs.texture_counts(slots, tables, {p: build[n][1] for n, p in enumerate(places)})
    assert counts[8] == 4 and sum(counts[:9]) == 9
    assert [counts[p] for p in places] == [b[1] for b in build]
    # every place before the animation drawn, none blank: the untextured ones are padded to their place's one TObj
    assert melee_dobjs.texture_counts(slots, tables, dict.fromkeys(range(9), 0)) == [1] * 10


def test_texture_slots_cannot_cross_an_animation_or_change_a_kept_material():
    """With no blank DObj before the animated one to carry it, a drawn DObj short of its place's TObjs is padded to
    them (dobjs.padded_material), never past them; spare capacity after the animation pays for nothing before it, and
    a kept material is never changed."""
    slots = [Slot(0, 0, (0,)), Slot(0, 1, (0,), animated=(0,)),
             Slot(1, 0, (0,)), Slot(1, 1, (0,))]
    tables = FighterTables(animated=(1,), body=(0, 2, 3))
    assert melee_dobjs.texture_counts(slots, tables, {0: 0}) == [1, 1, 1, 1]
    slots.insert(1, Slot(0, 1, (0,)))
    tables = FighterTables(animated=(2,), body=(0,), reserved=frozenset({1}))
    assert melee_dobjs.texture_counts(slots, tables, {0: 0}) == [1, 1, 1, 1, 1]
    # Yoshi's and Bowser's places with two TObjs: a one-texture material there gets a second, passing through
    slots = [Slot(0, 0, (0, 1)), Slot(0, 1, (0,), animated=(0,))]
    assert melee_dobjs.texture_counts(slots, FighterTables(animated=(2,), body=(0,)), {0: 1}) == [2, 1]


def test_no_low_detail_place_draws_more_pobjs_than_the_game_loads_in_one():
    """The game loads a DObj's PObjs by recursion on a 64 KB stack: Goku Jump's eight DObjs of one look, 944 PObjs
    joined under one low-detail place, hung Mario's match as it loaded. A group is split over places in order, and a
    DObj with no place left goes only where there is room for it."""
    mario, tables = _mario_slots(), _mario_tables()
    most = melee_dobjs.MAX_POBJS
    goku = [(0, 1, 10, 'body', n) for n in (154, 216, 217, 219, 49, 32, 26, 31)]
    low, left_out = melee_dobjs.choose_low_places(mario, tables, goku)
    assert low == {37: [0, 1], 38: [2, 3, 4], 39: [5, 6, 7]} and left_out == []
    assert all(sum(goku[n][4] for n in drawn) <= most for drawn in low.values())
    # more groups than places: one that fits nowhere is left out rather than overfilling a place
    many = [(0, 1, 100 - n, n, most - 10) for n in range(17)] + [(0, 1, 1, 'x', 20)]
    low, left_out = melee_dobjs.choose_low_places(mario, tables, many)
    assert left_out == [16, 17] and all(sum(many[n][4] for n in drawn) <= most for drawn in low.values())
    assert melee_dobjs._runs([1, 2, 3], [5, 5, 5], 10) == [[1, 2], [3]]


def test_kept_places_are_reserved_in_both_detail_sets():
    tables = facts_parts.reserve(_mario_tables(), (16, 37))
    assert melee_dobjs.choose_places(_mario_slots(), tables, [(0, 1)]) == [17]
    assert melee_dobjs.choose_low_places(_mario_slots(), tables, [(0, 1, 1, 'skin', 1)]) == ({38: [0]}, [])
    assert {f.name for f in fighters.in_scope()} == set(facts_parts.PARTS)
    assert all(f.keep == facts_parts.KEPT[f.name] for f in fighters.in_scope())
    assert {71, 79, 89, 93} <= set(fighters.fighter('Link').keep)
    assert 19 not in fighters.fighter('Kirby').keep  # open-mouth body is not a replacement object


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlMr.dat').is_file(),
                    reason=f'needs the vanilla game files in {melee.FILES_VAR}')
def test_marios_facts_in_the_vanilla_files():
    files = Path(os.environ[melee.FILES_VAR])
    tables = melee_dobjs.fighter_tables(files / 'PlMr.dat')
    slots = melee_dobjs.layout(files / 'PlMrNr.dat', fighters.fighter('Mario').symbol)
    assert tables.animated == MARIO_ANIMATED and tables.body == MARIO_BODY
    assert tables.elsewhere == frozenset(MARIO_LOW) and tables.low == MARIO_LOW
    assert [(s.joint, s.local, s.textures) for s in slots] == [(s.joint, s.local, s.textures) for s in _mario_slots()]
    assert [i for i, s in enumerate(slots) if s.animated] == list(MARIO_ANIMATED)


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlPe.dat').is_file(),
                    reason=f'needs the vanilla game files in {melee.FILES_VAR}')
def test_peachs_metal_places_cannot_draw_the_build():
    files = Path(os.environ[melee.FILES_VAR])
    tables = melee_dobjs.fighter_tables(files / 'PlPe.dat')
    slots = melee_dobjs.layout(files / 'PlPeNr.dat', fighters.fighter('Peach').symbol)
    metal = {52, 53, 54, 81, 84}
    assert metal <= tables.elsewhere
    assert not metal.intersection(tables.low)
    assert slots[52].joint == 67 and 52 in tables.body
    with pytest.raises(Failed, match='joint 67.*0 places'):
        melee_dobjs.choose_places(slots, tables, [(67, 1)])


def test_metal_set_is_excluded_but_the_separate_metal_list_is_not(tmp_path):
    path = _fighter_file(tmp_path / 'PlTs.dat', [], {0: [0, 1], 1: [2, 3], 2: [0, 2], 3: [1, 3]})
    tables = melee_dobjs.fighter_tables(path)
    slots = [Slot(0, i) for i in range(4)]
    assert melee_dobjs.choose_places(slots, tables, [(0, 0)]) == [0]
    assert tables.low == (2,)


def test_join_counts_planned_details_but_needs_no_temporary_body_places(monkeypatch):
    """The pure placement part of the join can be checked without starting Blender or mutating a mesh."""
    import sys
    from types import SimpleNamespace as Obj
    from rigswap.plugins.melee.prep.join import join_for_places

    class Material:
        node_tree = None

        def __init__(self, name):
            self.name = name

    monkeypatch.setitem(sys.modules, 'bpy', Obj())  # Any attempted Blender operation fails this test.
    meshes = [Obj(name=name, material_slots=[Obj(material=Material(name))], data=Obj(polygons=[Obj(material_index=0)]))
              for name in ('body', 'left', 'right')]
    slots = [Slot(0, n) for n in range(6)]
    tables = FighterTables(body=(0,), low=(3, 4, 5))
    with pytest.raises(Failed, match='3 DObjs.*1 places'):
        join_for_places(meshes, slots, tables, keep=('left', 'right'))
    record = join_for_places(meshes, slots, tables, keep=('left', 'right'), deferred=('left', 'right'))
    assert record['dobjs'] == 3 and record['kept_apart'] == ['left', 'right']
    assert record['looks'] == {'before': 1, 'after': 1}


def test_a_form_is_the_place_its_option_lists_where_option_0_lists_the_object():
    """Kirby's model 0: option 0 is 3-5 and option 1, the jump's, 0-2, in each detail; a model's kept part (6) is an
    object of its own, and so is a form that would stand for one (7, which would draw 6). The metal sets are not."""
    tables = FighterTables(reserved=frozenset((6,)), options={
        (0, 0, 0): (3, 4, 5), (0, 0, 1): (0, 1, 2), (1, 0, 0): (11, 12, 13), (1, 0, 1): (8, 9, 10),
        (0, 1, 0): (6,), (0, 1, 1): (7,), (0, 2, 0): (14,), (0, 2, 1): (6,), (2, 0, 1): (15,), (3, 0, 1): (16,),
        (melee_dobjs.HULL, 0, 0): (17,), (melee_dobjs.HULL, 0, 1): (18,)})
    assert melee_dobjs.forms(tables) == {0: 3, 1: 4, 2: 5, 8: 11, 9: 12, 10: 13, 18: 17}


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlKb.dat').is_file(),
                    reason=f'needs the vanilla game files in {melee.FILES_VAR}')
def test_the_forms_the_games_files_list():
    """What a form stands for is on its joint, with its TObjs, in every fighter, but for the last of Mr. Game & Watch's
    heads, on joint 20: no build fills what it stands for (facts/flat.py). Kirby's body and his head are the only models
    with forms the costume does not keep."""
    files = Path(os.environ[melee.FILES_VAR])
    found, moved = {}, set()
    for f in fighters.FIGHTERS:
        if not (files / f'Pl{f.code}.dat').is_file():
            continue
        tables = facts_parts.reserve(melee_dobjs.fighter_tables(files / f'Pl{f.code}.dat'), f.keep)
        slots = melee_dobjs.layout(files / f'Pl{f.code}Nr.dat', f.symbol)
        forms = melee_dobjs.forms(tables)
        for form, place in forms.items():
            if slots[form].joint != slots[place].joint:
                moved.add((f.name, form, place))
            assert len(slots[form].textures) == len(slots[place].textures), (f.name, form, place)
        if forms:
            found[f.name] = forms
    assert found['Kirby'] == {0: 3, 1: 4, 2: 5, 8: 11, 9: 12, 10: 13}
    assert set(found) == {'Kirby', 'GameAndWatch'}
    assert moved == {('GameAndWatch', 107, 3), ('GameAndWatch', 108, 17), ('GameAndWatch', 109, 25)}


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlGw.dat').is_file(),
                    reason=f'needs the vanilla game files in {melee.FILES_VAR}')
def test_mr_game_and_watchs_hull_is_read_as_a_set_of_its_own():
    """The outline his body is drawn with: PlGw.dat's items[10], the lookup shape of the visibility sets, which leaves
    a way to fill it (forms covers its options)."""
    options = melee_dobjs.fighter_tables(Path(os.environ[melee.FILES_VAR]) / 'PlGw.dat').options
    assert options[melee_dobjs.HULL, 0, 0] == (1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14)
    assert (melee_dobjs.HULL, 3, 0) in options and options[melee_dobjs.HULL, 3, 0] == (9,)


def test_a_padded_material_is_a_copy_with_pass_through_tobjs_after_its_own(tmp_path):
    """The material a place keeps more TObjs for: a copy, its own TObjs first (copied, pointing at the same images),
    then blank ones passing colour and alpha through on the next texture maps, the render mode told of each; the
    original, drawn elsewhere with its own count, is untouched."""
    import struct
    from rigswap.plugins.melee.dat import Dat
    from tests.plugins.melee.test_dat import _DatBuilder
    b = _DatBuilder()
    image = b.add(bytes(0x18))
    own = b.add(struct.pack('>3I', 0, 0, 0) + bytes(0x5C - 12))
    b.point(own + 0x4C, image)
    mobj = b.add(struct.pack('>2I', 0, melee_dobjs.RENDER_DIFFUSE | melee_dobjs.RENDER_TEX0) + bytes(0x10))
    b.point(mobj + 0x08, own)
    b.save(tmp_path / 'm.dat', [('mobj', mobj)])
    d = Dat.read(tmp_path / 'm.dat')
    blank = d.add(bytes(0x18))
    before = bytes(d.data[mobj:mobj + 0x18]), bytes(d.data[own:own + 0x5C])
    cache = {}
    padded = melee_dobjs.padded_material(d, mobj, 3, blank, cache)
    assert melee_dobjs.padded_material(d, mobj, 3, blank, cache) == padded
    tobjs = d.chain(d.ptr(padded + 0x08), 0x04)
    assert len(tobjs) == 3 and tobjs[0] != own and d.ptr(tobjs[0] + 0x4C) == image
    assert [d.u32(t + 0x08) for t in tobjs] == [0, 1, 2]
    assert [d.u32(t + 0x40) for t in tobjs[1:]] == [melee_dobjs.TOBJ_PASS] * 2
    assert [d.ptr(t + 0x4C) for t in tobjs[1:]] == [blank] * 2
    assert d.u32(padded + 0x04) == melee_dobjs.RENDER_DIFFUSE | sum(melee_dobjs.RENDER_TEX0 << i for i in range(3))
    assert (bytes(d.data[mobj:mobj + 0x18]), bytes(d.data[own:own + 0x5C])) == before
    assert d.chain(d.ptr(mobj + 0x08), 0x04) == [own]
    untextured = d.add(struct.pack('>2I', 0, melee_dobjs.RENDER_DIFFUSE) + bytes(0x10))
    padded = melee_dobjs.padded_material(d, untextured, 1, blank, cache)
    assert len(d.chain(d.ptr(padded + 0x08), 0x04)) == 1 and d.ptr(untextured + 0x08) is None
