"""The Melee plugin's export (``run``): ``<build>/build.blend`` written into the fighter's vanilla costume
(Pl<code>Nr.dat) and read back, with what it goes through:

    write.py    HSDRawViewer (hsdraw, export_dae, import_dae), on copies of the vanilla files (vanilla_copies); the
                build to a .glb and a .glb to Collada through Blender 4 (blend_to_glb, glb_to_dae); the texture formats
                and alpha modes the import is told; the vanilla DObj list put back around the build's DObjs
                (fill_vanilla_dobjs), the kept vanilla parts copied in (copy_parts)
    dobjs.py    the costume's DObjs as the game counts them: the vanilla order and places, the PObj chains the game can
                load, the low-detail set, an object's other forms
    eyes.py     the build's eyes as the fighter's own blinking eyes (its apply runs when the build readied the eyes)
    checks.py   every read-back check, and require_checks

Collada is read the same way whoever wrote it (dae.py: read_dae). See the package's docstring for where the game's
files and HSDRawViewer are."""
from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import tempfile

from ... import Failed
from .. import DETAILS, FILES_VAR, _folder
from ..dae import read_dae
from ..dat import dat_joints
from ..facts import parts as melee_parts
from ..facts.fighters import _build_fighter
from . import dobjs as melee_dobjs
from . import physics
from .checks import (VERTEX_TOLERANCE, alpha_mode_check, check_costume_size, check_palettes,
                     compare_joints, compare_models, dobj_count_check, dobj_image_check, forms_check,
                     image_count_check, material_count_check,
                     position_check, preservation_checks, require_checks, texture_format_check, tree_differs,
                     without_kept)
from .flat import drawn_dae, hull_of
from .write import (NORMALS_REPAIRED, _formats_by_dae_id, blend_to_glb, export_dae, fill_forms, fill_vanilla_dobjs,
                    glb_to_dae, hsdraw, import_dae, plan_alpha_modes, preserve_samplers, preserve_sidedness,
                    vanilla_copies, wraps_written, written_alike)


# ---- the export ------------------------------------------------------------------------------------------------

def run(build_dir, out, fighter_name=None, files_dir=None):
    """Write the costume: ``<build_dir>/build.blend`` (a skinned model on the fighter's skeleton, bones named JOBJ_<n>
    as in the fighter's .glb) into the fighter's vanilla Pl<code>Nr.dat, as ``out``, with the vanilla costume's DObjs
    in number and order around the model's (dobjs.py: the game finds the textures it animates, and the parts it
    shows and hides, by their place; the low-detail DObjs draw the model again, for shadows and the magnifier, and every
    other form of the model's object draws it too). Then
    read it back: export the new file's high-detail and low-detail DObjs and check every joint is where the vanilla
    costume has it, the triangles and vertices are the model's in both, and the DObjs and the animated TObjs are where
    the vanilla costume has them. Failed with the worst joint, or the numbers, when the read-back differs. When prep has
    written ``<build_dir>/textures.json``, each material's texture is written in the format it names. A weapon of the
    fighter's whose slot a held item of the build took (``<build_dir>/weapons_replaced.json``) is not kept, nor are its
    other forms: their places are blank (facts/parts.py: kept_places, hidden_forms), and readback.json names them
    (``weapons_replaced``, ``weapon_forms_hidden``).

    The fighter is ``fighter_name``, else the name in ``<build_dir>/target.json`` ({"name": ...}). Working files go in
    a folder of their own under ``<build_dir>/export/``, with the export's own copies of the vanilla files
    (``vanilla_copies``: two exports of one fighter can run at once); ``readback.json`` there has every number of the
    read-back."""
    build_dir = Path(build_dir).resolve()
    f = _build_fighter(build_dir, fighter_name)
    files = _folder(files_dir, FILES_VAR, 'vanilla game files')
    for needed in (files / f'Pl{f.code}Nr.dat', files / f'Pl{f.code}.dat'):
        if not needed.is_file():
            raise Failed(f'{needed.name} is not in {files}')
    out = Path(out).resolve().with_suffix('.dat')
    work, numbers = write_build(build_dir, out, f, files)
    _save_readback(work, numbers)
    return out


def write_build(build_dir, out, f, files):
    """One build (``<build_dir>/build.blend``) written as ``out`` and checked (see ``run``), with a folder of working
    files of its own under ``<build_dir>/export/``. Returns (that folder, readback.json's numbers)."""
    dat, fighter_dat = files / f'Pl{f.code}Nr.dat', files / f'Pl{f.code}.dat'
    work = build_dir / 'export'
    work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=work))
    dat, fighter_dat = vanilla_copies(work, dat, fighter_dat)
    if (files / 'PlCo.dat').is_file():
        vanilla_copies(work, files / 'PlCo.dat')  # dynamics part indices -> costume joints
    keep, replaced = melee_parts.kept_places(f, build_dir)

    vanilla = read_dae(export_dae(dat, f.symbol, work / 'vanilla' / 'vanilla.dae'))
    body, accessories, empty_chains = physics.split(build_dir, work, f)
    glb = blend_to_glb(body, work / 'model.glb', dat, fighter_dat, f.symbol, keep, build_dir)
    joins_path = work / 'joins.json'
    joins = json.loads(joins_path.read_text(encoding='utf-8')) if joins_path.is_file() else {}
    dae = work / 'dae' / 'model.dae'
    try:
        glb_to_dae(glb, dae)
    except Failed:
        if dae.is_file():
            _save_readback(work, {'checks': {'images': image_count_check(glb, dae)}})
        raise
    checks = {'images': image_count_check(glb, dae), 'dae_materials': material_count_check(glb, dae)}
    repaired = dae.parent / NORMALS_REPAIRED       # corners Blender 4 wrote with no normal, given one (write.py)
    if repaired.is_file():
        checks['normals_repaired'] = {'corners': json.loads(repaired.read_text(encoding='utf-8')), 'pass': True}
    _save_readback(work, {'checks': checks})
    require_checks(checks, dae.name)
    model = read_dae(drawn_dae(dae, build_dir))     # what the checks hold the costume's drawn DObjs to
    differs = tree_differs(vanilla.joints, vanilla.parents, model.joints, model.parents)
    if differs:
        raise Failed(f"the build's skeleton is not {f.name}'s: {differs}")
    # The import replaces the DAE's joint transforms with the vanilla DAT's before binding (import_dae requires
    # its preserving-original-bones path). Blender's target armature is an approximation of those transforms;
    # comparing its local frames here mistakes that conversion for a change to the game's skeleton. Check the
    # written DAT below, exactly, including rotations and scales, rather than the discarded DAE transforms.
    textures = build_dir / 'textures.json'
    if accessories and textures.is_file():
        # Materials used only by accessories leave the body; their images were
        # written beside their SMDs from prep's already sized textures.
        import xml.etree.ElementTree as ET
        from ..dae import NS
        body_materials = {m.get('name') for m in ET.parse(dae).getroot().iter(NS + 'material')}
        asked = json.loads(textures.read_text(encoding='utf-8'))
        aliases = joins.get('material_aliases', {})
        textures = work / 'body-textures.json'
        textures.write_text(json.dumps({n: fmt for n, fmt in asked.items()
                                        if aliases.get(n, n) in body_materials}, indent=1), encoding='utf-8')
    if textures.is_file() and (joins.get('material_aliases') or joins.get('unused_materials')
                               or joins.get('atlases')):
        asked, mapped = json.loads(textures.read_text(encoding='utf-8')), {}
        # an atlas is written in the format it was packed for, which holds each of its images (prep/atlas.py)
        atlased = {name: a['format'] for a in joins.get('atlases', []) for name in a['materials']}
        for name, fmt in asked.items():
            if name in joins.get('unused_materials', []):
                continue
            name = joins['material_aliases'].get(name, name)
            fmt = atlased.get(name, fmt)
            if name in mapped and mapped[name] != fmt:
                raise Failed(f'joined material {name} requests different texture formats')
            mapped[name] = fmt
        textures = work / 'textures.json'
        textures.write_text(json.dumps(mapped, indent=1), encoding='utf-8')
    formats = _formats_by_dae_id(textures, work / 'dae' / 'model.dae')
    alpha_modes, alpha_plan = plan_alpha_modes(glb, dae, model, formats, joins.get('one_place', {}))
    # a texture the build draws opaque that the game would draw nothing of (a download fault, written as the build
    # draws it where it can be): said, not failed (write.py: _drawn_opaque)
    transparent = {key: row['fully_transparent'] for key, row in alpha_plan.items() if row.get('fully_transparent')}
    notes = [row['said'] for row in transparent.values()]
    if transparent:
        checks['transparent_textures'] = {'materials': transparent, 'pass': True}
    imported = import_dae(dat, f.symbol, work / 'dae' / 'model.dae', work / 'imported.dat', formats, alpha_modes)
    joins['texture_samplers'] = preserve_samplers(glb, dae, imported, f.symbol)
    joins['sidedness'] = preserve_sidedness(glb, dae, imported, f.symbol)
    if 'dobjs' in joins:        # the DObjs the join planned places for, against the ones the import made
        checks['dobj_count'] = dobj_count_check(joins['dobjs'], imported, f.symbol)
        _save_readback(work, {'checks': checks, 'joins': joins})
        require_checks(checks, imported.name)
    out.unlink(missing_ok=True)
    eye_plan = joins.get('eye_plan', {})
    hulls, deferred = [], []
    if eye_plan.get('destinations') or hull_of(build_dir):
        from .eyes import dae_dobjs
        made = dae_dobjs(dae)
        hulls = [n for n, (name, _) in enumerate(made) if name in hull_of(build_dir)]
        deferred = hulls + [n for n, (name, _) in enumerate(made) if name in eye_plan.get('destinations', ())]
    dobjs = fill_vanilla_dobjs(imported, dat, fighter_dat, f.symbol, out, keep=keep, deferred=deferred, high_only=hulls)
    if eye_plan:
        dobjs['eye_plan'] = eye_plan
    for name in DETAILS:    # each detail the build readied (rigswap.core.eyes), made the game's by this plugin's module
        if (Path(build_dir) / f'{name}.json').is_file():        # of its name (eyes.py: the eyes made to blink)
            dobjs[name] = import_module(f'{__package__}.{name}').apply(
                build_dir, out, dat, fighter_dat, f.symbol, dobjs, work / 'dae' / 'model.dae',
                work / 'vanilla' / 'vanilla.dae')
    tables = melee_parts.reserve(melee_dobjs.fighter_tables(fighter_dat), keep)
    dobjs['forms'] = fill_forms(out, tables, f.symbol)

    emitted = physics.emit(out, accessories, empty_chains, work)
    _save_readback(work, {'physics': emitted, 'checks': checks})
    memory = check_costume_size(out, f)
    palettes = check_palettes(out, f.symbol)

    # The skeleton is read from the files themselves: HSDRawViewer's .dae gives a joint where the mesh was bound to it,
    # and the import binds the mesh again at the skeleton's rest, so in 18 of the 25 vanilla costumes a few joints'
    # bind places differ from the vanilla's by up to 0.08 units though no joint moved (see compare_models' joints).
    check = compare_joints(dat_joints(dat, f.symbol, local=True), dat_joints(out, f.symbol, local=True))
    animated = melee_dobjs.fighter_tables(fighter_dat).animated
    dobjs_differ = melee_dobjs.layout_differs(melee_dobjs.layout(dat, f.symbol), melee_dobjs.layout(out, f.symbol),
                                              animated)

    def read_back(which, folder):       # the DObjs the game draws in play (high) or for shadows and the magnifier (low)
        dae = work / folder / 'readback.dae'
        dae.parent.mkdir(parents=True, exist_ok=True)
        hsdraw('--model', 'export', out, f.symbol, dae, '--names', 'place', '--dobjs', which, '--fighter', fighter_dat,
               '--show', ','.join(f'{m}:0' for m in f.shown))
        return read_dae(dae)
    drawn, low = read_back('high', 'readback'), read_back('low', 'readback-low')
    # the read-back knows no hull set, so it shows the hull with the body; the model the checks hold it to has none
    hulls = list(dobjs.get('flat', {}).get('hull', {}).values())
    back, low = (without_kept(m, [*keep, *hulls]) for m in (drawn, low))
    numbers = {**({'notes': notes} if notes else {}),
               'memory': memory, 'physics': emitted, 'skeleton': str(check),
               'dobjs': dict(dobjs, differs=dobjs_differ), 'joins': joins,
               'model_to_readback': compare_models(model, back), 'model_to_low_detail': compare_models(model, low),
               'vanilla_to_readback': compare_models(vanilla, back),
               'texture_formats': texture_format_check(formats, model, back)}
    if replaced:
        numbers['weapons_replaced'] = replaced
        numbers['weapon_forms_hidden'] = list(melee_parts.hidden_forms(f.name, replaced))
    checks.update(preservation_checks(model, back, numbers['model_to_readback'],
                                      eyes=dobjs.get('eyes', {}).get('moved', []),
                                      alike=written_alike(model, alpha_plan,
                                                          wraps_written(joins['texture_samplers'])),
                                      padded=[int(p) for p in dobjs.get('padded_textures', {})]))
    checks['skeleton'] = {'matrix_change': check.matrix_error, 'tolerance': 0,
                          'pass': not check.moved(tolerance=0)}
    checks['positions'] = position_check(numbers['model_to_readback'])
    checks['low_positions'] = position_check(numbers['model_to_low_detail'])
    checks['texture_formats'] = numbers['texture_formats']
    checks['palettes'] = palettes
    if 'kept_parts' in dobjs:
        checks['kept_parts'] = dobjs['kept_parts']
    checks['alpha_modes'] = alpha_mode_check(out, f.symbol, model, back, alpha_plan)
    checks['dobj_images'] = dobj_image_check(out, f.symbol, model, back, low)
    checks['forms'] = forms_check(out, f.symbol, tables)
    numbers['checks'] = checks
    _save_readback(work, numbers)
    require_checks(checks, out.name)
    if dobjs_differ:
        raise Failed(f'read-back of {out.name}: its DObjs are not where the game looks for them: {dobjs_differ}')
    if back.triangles != model.triangles:
        raise Failed(f'read-back of {out.name} draws {back.triangles} triangles; the model has {model.triangles}')
    if not dobjs['low_detail']['left_out'] and low.triangles != model.triangles:
        raise Failed(f'read-back of {out.name}: its low-detail DObjs draw {low.triangles} triangles; the model has '
                     f'{model.triangles}')
    gap = numbers['model_to_readback'].get('vertex_gap_max')
    if gap is None or gap > VERTEX_TOLERANCE:
        raise Failed(f'read-back of {out.name}: a vertex is {gap} from any of the model\'s '
                     f'(tolerance {VERTEX_TOLERANCE})')
    return work, numbers



def _save_readback(work, numbers):
    numbers['export_directory'] = str(work)
    text = json.dumps(numbers, indent=1)
    (work / 'readback.json').write_text(text, encoding='utf-8')
    (work.parent / 'readback.json').write_text(text, encoding='utf-8')
