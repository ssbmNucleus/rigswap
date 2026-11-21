"""The fighters: every fighter with a Pl<code>Nr.dat costume in the game, its game files and the numbers prep and export
need, each with where it was read or measured (FIGHTERS); ``fighter`` and ``in_scope`` find them."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from ... import Failed
from .parts import KEPT


@dataclass(frozen=True)
class Fighter:
    name: str          # plain, no spaces: the target's file names
    code: str          # Pl<code>Nr.dat is the vanilla costume, Pl<code>.dat the fighter's data
    symbol: str        # the costume's root joint (the root symbol in Pl<code>Nr.dat)
    triangles: int     # the triangles the vanilla costume draws; prep's triangle budget is at least this
    kept: int          # bytes of the costume file that are not the model, which the import keeps
    preloaded: tuple   # bytes of the files the game preloads with the costume: Pl<code>.dat, the effect file
    in_scope: bool = True
    why_out: str = ''
    shown: tuple = (0,)  # the models whose option 0 the game shows when costume 0 spawns (the rest show none)
    keep: tuple = ()     # vanilla DObj places required by the fighter's equipment and replacement states
    scale: float = 1.        # ftData common attributes +0x8C (model_scaling)
    motions: str = ''        # the code of the Pl<code>AJ.dat that holds its common animations, when not its own
    flat: bool = False       # the game flattens his model and draws its outline (facts/flat.py)

    @property
    def max_influences(self):
        """Target.max_influences: the most joints any vanilla costume weights one vertex to (MAX_INFLUENCES), for every
        fighter in scope; 0 for one out of scope."""
        return MAX_INFLUENCES if self.in_scope else 0


# The most joints one envelope of a vanilla costume weights a vertex to, across the 25 fighters' neutral v1.02 costumes
# (Kirby's; every other fighter's own is 2 to 4), read again by dat.vanilla_max_influences in the tests. The game walks
# every joint in an envelope (src/sysdolphin/baselib/pobj.c:1154) and HSDRawViewer draws up to 6 (RenderDObj.cs
# MAX_WEIGHTS), so a costume may weight a vertex to as many joints as the game's own costumes do, whatever its fighter.
# A fighter's own maximum was the old importer's policy, not a limit of the game, and a cap of 2 tore the skin of a
# model whose vertices sit among more parts than that.
MAX_INFLUENCES = 6


# Every fighter with a Pl<code>Nr.dat costume in the game, and the costume's root joint as its root table names it.
# Zelda and Sheik are two targets: two costume files, two skeletons. So are Popo and Nana: a pair is two models, one on
# each fighter. Left out:
#   Master Hand, Crazy Hand, Giga Bowser: bosses, not on the character select screen.
#   Male and Female Wireframe, Sandbag: single-player and minigame opponents, not on the character select screen.
# The numbers prep needs (0 for fighters that are not targets):
#   triangles  drawn in play when costume 0 spawns: the triangles with three different corners in the display lists of
#              the high-detail DObjs (set 0, option 0 of each model in ``shown``) of the vanilla v1.02 Pl<code>Nr.dat,
#              counted by the HSDLib fork's tools/check_export.py; the target's .dae has the same, Ganondorf and Ness
#              within 3 (2026-10-10). The numbers before these were every DObj of a modded file.
#   kept       the blink and other material animations and the file's tables: the bytes of the file not reached from
#              the model's root joint, in the vanilla v1.02 Pl<code>Nr.dat and in the fighter's own high-detail model
#              (the target's .dae) imported back into it by HSDRawViewer, the larger (2026-10-10). Peach keeps 397,478
#              bytes; Captain Falcon, Ganondorf, Samus and Game & Watch, whose files are all model, 63 to 67 (the
#              header and root table). The export's blank DObjs and materials add a few KB to this.
#   preloaded  the vanilla v1.02 sizes (the disc's file table) of the fighter's Pl<code>.dat and the effect file the
#              game loads for it (clones share their original's; Game & Watch and the single-player fighters have
#              none); a fighter that brings another (Zelda and Sheik, Popo and Nana) also preloads that one's
#              Pl<code>.dat and largest vanilla costume. (randall shared/modellab/preload_heap.py VANILLA_FIGHTER_FILES,
#              PARTNERS)
#   scale      Pl<code>.dat common attributes, 2026-10-10, read again by model_scale (dat.py) in the tests. Source:
#              untangle randall/model_import/native_rig/fighter_scale.py:MODEL_SCALING, scale_offset.
#   motions    Nana's action table (PlNn.dat ftData +0x0C) gives only her eight partner specials an animation of their
#              own, which PlNnAJ.dat holds. A row with none plays Popo's (ft/ftdata.c ftData_80085FD4), in PlPpAJ.dat.
FIGHTERS = (
    Fighter('Mario', 'Mr', 'PlyMario5K_Share_joint', 4718, 227_546, (194_992, 78_654), scale=1.1),
    Fighter('DrMario', 'Dr', 'PlyDrmario5K_Share_joint', 4597, 227_934, (251_170, 78_654), scale=1.1),
    Fighter('Luigi', 'Lg', 'PlyLuigi5K_Share_joint', 4759, 113_362, (172_308, 76_186), scale=1.25),
    Fighter('Peach', 'Pe', 'PlyPeach5K_Share_joint', 4984, 397_478, (269_820, 5_094), shown=(0, 1, 2, 4, 6), keep=KEPT['Peach'], scale=1.15),
    Fighter('Bowser', 'Kp', 'PlyKoopa5K_Share_joint', 4960, 32_182, (195_840, 96_650), keep=KEPT['Bowser'], scale=0.69),
    Fighter('Yoshi', 'Ys', 'PlyYoshi5K_Share_joint', 4911, 27_482, (287_136, 23_754), keep=KEPT['Yoshi'], scale=1.05),
    Fighter('DonkeyKong', 'Dk', 'PlyDonkey5K_Share_joint', 4810, 120_140, (172_229, 64_311), scale=1.0),
    Fighter('CaptainFalcon', 'Ca', 'PlyCaptain5K_Share_joint', 4995, 65, (161_726, 162_876), scale=0.97),
    Fighter('Ganondorf', 'Gn', 'PlyGanon5K_Share_joint', 4939, 63, (201_400, 149_158), keep=KEPT['Ganondorf'], scale=1.08),
    Fighter('Falco', 'Fc', 'PlyFalco5K_Share_joint', 4150, 27_270, (237_784, 123_588), scale=1.1),
    Fighter('Fox', 'Fx', 'PlyFox5K_Share_joint', 4966, 27_790, (259_850, 123_588), scale=0.96),
    Fighter('Ness', 'Ns', 'PlyNess5K_Share_joint', 5002, 52_076, (356_055, 101_157), scale=1.0),
    Fighter('Kirby', 'Kb', 'PlyKirby5K_Share_joint', 2622, 86_950, (198_866, 65_702), shown=(0, 1), keep=KEPT['Kirby'], scale=0.92),
    Fighter('Samus', 'Ss', 'PlySamus5K_Share_joint', 4980, 63, (319_219, 82_986), keep=KEPT['Samus'], scale=0.88),
    Fighter('Zelda', 'Zd', 'PlyZelda5K_Share_joint', 4616, 85_630, (187_304, 86_954, 166_907, 380_180), shown=(0, 1), scale=1.26),
    Fighter('Sheik', 'Sk', 'PlySeak5K_Share_joint', 4943, 134_612, (166_907, 86_954, 187_304, 350_082), keep=KEPT['Sheik'], scale=1.4),
    Fighter('Link', 'Lk', 'PlyLink5K_Share_joint', 4949, 206_680, (239_099, 34_809), shown=(0, 1, 2), keep=KEPT['Link'], scale=1.22),
    Fighter('YoungLink', 'Cl', 'PlyClink5K_Share_joint', 4974, 206_010, (289_480, 34_809), shown=(0, 1, 2), keep=KEPT['YoungLink'], scale=0.96),
    Fighter('Pichu', 'Pc', 'PlyPichu5K_Share_joint', 2884, 59_414, (240_228, 114_472), scale=0.5),
    Fighter('Pikachu', 'Pk', 'PlyPikachu5K_Share_joint', 3806, 40_522, (216_410, 114_472), shown=(0, 1), scale=0.9),
    Fighter('Jigglypuff', 'Pr', 'PlyPurin5K_Share_joint', 3756, 154_374, (114_528, 26_834), scale=0.94),
    Fighter('Mewtwo', 'Mt', 'PlyMewtwo5K_Share_joint', 5008, 76_796, (197_653, 59_859), scale=1.0),
    Fighter('GameAndWatch', 'Gw', 'PlyGamewatch5K_Share_joint', 8286, 67, (260_320,), shown=(0, 2, 3), keep=KEPT['GameAndWatch'], scale=1.02, flat=True),
    Fighter('Marth', 'Ms', 'PlyMars5K_Share_joint', 4997, 173_452, (175_023, 23_977), shown=(0, 1), keep=KEPT['Marth'], scale=1.15),
    Fighter('Roy', 'Fe', 'PlyEmblem5K_Share_joint', 4904, 173_464, (268_085, 26_843), shown=(0, 1), keep=KEPT['Roy'], scale=1.08),
    Fighter('Popo', 'Pp', 'PlyPopo5K_Share_joint', 2524, 133_304, (109_674, 51_363, 94_630, 350_748), shown=(0, 1),
            keep=KEPT['Popo'], scale=1.15),
    Fighter('Nana', 'Nn', 'PlyNana5K_Share_joint', 2533, 133_304, (94_630, 51_363, 109_674, 350_744), shown=(0, 1),
            keep=KEPT['Nana'], scale=1.15, motions='Pp'),
    Fighter('MasterHand', 'Mh', 'PlyMasterhand_Share_joint', 0, 0, (53_509,), False, 'a boss, not selectable'),
    Fighter('CrazyHand', 'Ch', 'PlyCrazyhand_Share_joint', 0, 0, (54_572,), False, 'a boss, not selectable'),
    Fighter('GigaBowser', 'Gk', 'PlyGkoopa5K_Share_joint', 0, 0, (197_037, 96_650), False, 'a boss, not selectable'),
    Fighter('MaleWireframe', 'Bo', 'PlyBoy_Share_joint', 0, 0, (86_838,), False,
            'a single-player opponent, not selectable'),
    Fighter('FemaleWireframe', 'Gl', 'PlyGirl_Share_joint', 0, 0, (102_603,), False,
            'a single-player opponent, not selectable'),
    Fighter('Sandbag', 'Sb', 'PlySandbag_Share_joint', 0, 0, (17_010,), False,
            'the Home-Run Contest bag, not selectable'),
)


def _key(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def fighter(name):
    """The in-scope fighter called ``name`` (any case, spaces and dots ignored) or with that file code."""
    for f in FIGHTERS:
        if _key(name) in (_key(f.name), _key(f.code)):
            if not f.in_scope:
                raise Failed(f'{f.name} is not a target: {f.why_out}')
            return f
    raise Failed(f'no Melee fighter called {name!r}; the fighters are '
                 + ', '.join(f.name for f in FIGHTERS if f.in_scope))


def in_scope():
    return [f for f in FIGHTERS if f.in_scope]


def _build_fighter(build_dir, fighter_name):
    """The fighter ``fighter_name``, else the one ``<build_dir>/target.json`` ({"name": ...}) names."""
    if fighter_name is None:
        named = Path(build_dir) / 'target.json'
        if not named.is_file():
            raise Failed(f'which fighter? pass fighter_name or write {named}')
        fighter_name = json.loads(named.read_text(encoding='utf-8'))['name']
    return fighter(fighter_name)
