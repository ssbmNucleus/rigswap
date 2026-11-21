"""What a joint's places can draw: a place is one material and a chain of PObjs (export/write.py: fill_vanilla_dobjs),
so a joint needs one place for each look, whatever its meshes, and when the build has more looks on a joint than the
game gives it places, those looks share one image (join_for_places, atlas.py). Not a step of prep's ``run``: export runs
it on its copy of the build (export/write.py: blend_to_glb), when the places are known. No texture is shrunk to make it
fit."""
from __future__ import annotations

from ... import Failed
from ..export.dobjs import choose_places


def _material_settings(m, image_key=lambda image: image, *, ignore=()):
    """Exact settings of a plain image/Principled material, but for the ones named in ``ignore``; unfamiliar node
    graphs keep their own identity."""
    if m is None:
        return None
    tree = m.node_tree
    if tree is None or getattr(m, 'animation_data', None) or tree.animation_data or any(
                          n.type not in {'BSDF_PRINCIPLED', 'OUTPUT_MATERIAL', 'TEX_IMAGE', 'MIX', 'VERTEX_COLOR'}
                          for n in tree.nodes):
        return m

    def value(v):
        if isinstance(v, str):
            return v
        try:
            return tuple(v)
        except TypeError:
            return v
    nodes = list(tree.nodes)
    settings = tuple((n.type, tuple((i.identifier, value(i.default_value)) for i in n.inputs
                                    if hasattr(i, 'default_value')),
                      tuple((p, image_key(n.image) if p == 'image' else value(getattr(n, p))) for p in
                            ('image', 'interpolation', 'extension', 'projection', 'projection_blend',
                             'distribution', 'subsurface_method', 'target', 'is_active_output', 'mute',
                             'data_type', 'blend_type', 'clamp_factor', 'clamp_result', 'factor_mode', 'layer_name')
                            if hasattr(n, p) and p not in ignore))
                     for n in nodes)
    links = tuple(sorted((nodes.index(l.from_node), l.from_socket.identifier,
                          nodes.index(l.to_node), l.to_socket.identifier) for l in tree.links))
    return (m.get('kind'), tuple((p, value(getattr(m, p))) for p in
                  ('diffuse_color', 'metallic', 'roughness', 'use_backface_culling', 'surface_render_method',
                   'blend_method', 'alpha_threshold') if hasattr(m, p) and p not in ignore), settings, links)


def join_for_places(meshes, vanilla, tables, keep=(), deferred=(), formats=None):
    """Make every joint's looks fit its places, and name the materials that will share one.

    Two materials differing only in culling are one look, since a PObj holds its own culling (write.py:
    fill_vanilla_dobjs chains them, as it does the meshes of one material). While the looks on a joint outnumber the
    places the game gives it (``choose_places`` says, for any joint and any fighter), two of them share an image
    (atlas.merge_images, its UVs moved to match), until the planner fits them or no two can. It fails with the planner's
    own count when none is left, or when a look has no image to share.

    The objects in ``keep`` (a detail's own, the eyes) keep their materials. Only objects with destinations already
    decided by the detail exporter (``deferred``) need no ordinary high-detail place. ``dobjs`` in the record counts
    every DObj the import will make, one for each mesh and material, which export holds the import to
    (export/__init__.py, the ``dobj_count`` check); ``one_place`` names, for each material sharing a look with an earlier
    one, that one (write.py: plan_alpha_modes). Low detail needs nothing from here: its meshes are high detail's, and
    choose_low_places shares its places out by look.
    """
    from .textures import _base_color_image
    from .atlas import image_identity, merge_images
    formats = dict(formats or {})
    images = {}

    def image_key(image):
        if image not in images:
            images[image] = image_identity(image)
        return images[image]
    # Objects in ``keep`` (the eyes the blink needs as their own DObjs) keep their materials; their DObjs still count,
    # even when their final animated destinations need no temporary body places.
    kept = [o for o in meshes if o.name in set(keep)]
    meshes = [o for o in meshes if o.name not in set(keep)]

    def materials(o):
        return {o.material_slots[p.material_index].material if p.material_index < len(o.material_slots) else None
                for p in o.data.polygons}

    def joint(o):
        if 'SINGLE' in o.name:
            groups = {o.vertex_groups[g.group].name for v in o.data.vertices for g in v.groups if g.weight > 0}
            if len(groups) == 1 and next(iter(groups)).startswith('JOBJ_'):
                return int(next(iter(groups))[5:])
        return 0

    def look(m):
        return _material_settings(m, image_key, ignore={'use_backface_culling'}), formats.get(m.name if m else None)

    def textures(m):
        return int(m is not None and m.node_tree is not None and _base_color_image(m.node_tree) is not None)

    def needs():
        """[(joint, TObjs)] for each look on each joint the build's places have to hold."""
        return list({(joint(o), look(m)): (joint(o), textures(m)) for o in meshes + kept if o.name not in deferred
                     for m in materials(o)}.values())

    used = {m for o in meshes + kept for m in materials(o)}
    report = {'material_aliases': {}, 'unused_materials': sorted({s.material.name for o in meshes + kept
               for s in o.material_slots if s.material is not None and s.material not in used})}
    if kept:
        report['kept_apart'] = sorted(o.name for o in kept)
    protected = {m for o in kept for m in materials(o)}

    # The importer shares identical materials. Give them one identity before establishing the GLB baseline, retaining
    # every old name for texture formats.
    canonical = {}
    for o in meshes:
        for slot in o.material_slots:
            m = slot.material
            if m in protected:
                continue
            key = (_material_settings(m, image_key), formats.get(m.name) if m else None)
            first = canonical.setdefault(key, m)
            if m != first:
                report['material_aliases'][m.name] = first.name
                slot.material = first
    report['looks'] = {'before': len(needs())}
    while True:
        try:
            choose_places(vanilla, tables, needs())
            break
        except Failed:
            atlas = merge_images(meshes, protected, materials, joint, formats, _material_settings)
            if not atlas:
                raise
            formats.update({name: atlas['format'] for name in atlas['materials']})
            report.setdefault('atlases', []).append(atlas)
    report['looks']['after'] = len(needs())
    first = {}
    report['one_place'] = {m.name: first.setdefault(look(m), m).name for m in
                           sorted({m for o in meshes + kept for m in materials(o) if m is not None},
                                  key=lambda m: m.name) if first.setdefault(look(m), m) is not m}
    report['dobjs'] = sum(len(materials(o)) for o in meshes + kept)
    return report
