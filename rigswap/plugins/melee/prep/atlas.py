"""Capacity-only atlases: images of one shader sharing an image, original texels and affine UVs, in a GX format drawing
each image as its own did (shared_formats). Each material keeps its culling, a PObj's own in the file.

Only plain, single-image materials using the render UV layer qualify. The finite UV range actually drawn is laid out
as its wrap mode draws it (repeated, mirrored or clamped), with a GX block of padding for filtering, so the atlas needs
no wrap of its own. No resizing, shader baking or geometry changes.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

from .textures import GX_FORMATS, alpha_class, gx_bytes, read_pixels, write_pixels
from ..facts.alpha import choose, draws_nothing, gate


# The longest side GX takes (GXTexture.c GXInitTexObj: the assert on width and height). TEXTURE_MAX_SIDE is prep's
# limit on a model's own image, for memory; an atlas holds such images at their own size, so only this binds it.
GX_MAX_SIDE = 1024


def image_identity(image):
    """Identical decoded PNG texels and interpretation, even when imported as different image datablocks."""
    if image is None or image.source not in {'FILE', 'GENERATED'} or not all(image.size):
        return image
    return (tuple(image.size), image.colorspace_settings.name, image.alpha_mode,
            hashlib.sha256(read_pixels(image).tobytes()).digest())


def wrapped(texels, side, extension):
    """Where a wrap mode reads ``texels`` (an integer array) on a side of ``side`` texels."""
    if extension == 'REPEAT':
        return texels % side
    if extension == 'MIRROR':
        return np.minimum(texels % (2 * side), 2 * side - 1 - texels % (2 * side))
    return np.clip(texels, 0, side - 1)


def pack(images, bounds, extensions, fmt):
    """Two RGBA arrays, UV bounds in bottom-up coordinates, their wrap modes -> (pixels, UV transforms), or no
    full-resolution fit.

    A transform is (scale, offset): uv * scale + offset. Integer texel translations keep compression blocks aligned.
    CI8 is only combined when the union fits its 256-entry palette, so merging cannot force colour reduction.
    """
    code = fmt.split(':')[0]
    if code not in GX_FORMATS:
        return None
    if len({alpha_class(p) for p in images}) != 1:
        return None
    if code == 'CI8' and len(np.unique(np.concatenate([p.reshape(-1, 4) for p in images]), axis=0)) > 256:
        return None
    block = np.array(GX_FORMATS[code][1:])
    tiles, origins, sizes = [], [], []
    for pixels, (lo, hi), extension in zip(images, bounds, extensions):
        lo, hi = np.asarray(lo), np.asarray(hi)
        if not np.isfinite([lo, hi]).all() or np.any(hi < lo):
            return None
        size = np.array(pixels.shape[1::-1])
        # Align the entire sampled region to compression blocks, not just to UV tiles. This also handles tiny
        # images and UVs outside [0, 1], without modulo-wrapping vertices (which would break interpolation).
        start = np.floor(lo * size / block) * block - block
        end = np.ceil(hi * size / block) * block + block
        extent = end - start
        if np.any(extent > GX_MAX_SIDE):
            return None
        start, extent = start.astype(int), extent.astype(int)
        x, y = [wrapped(np.arange(s, s + e), side, extension) for s, e, side in zip(start, extent, size)]
        tiles.append(pixels[::-1][y[:, None], x])  # work bottom-up, like Blender's UVs
        origins.append(start)
        sizes.append(size)
    a, b = [np.array(t.shape[1::-1]) for t in tiles]
    layouts = []
    for axis in (0, 1):
        offset = np.zeros(2, dtype=int)
        offset[axis] = a[axis]
        needed = np.maximum(a, offset + b)
        size = np.array([1 << (int(n) - 1).bit_length() for n in needed])
        if max(size) <= GX_MAX_SIDE:
            layouts.append((int(np.prod(size)), axis, size, offset))
    if not layouts:
        return None
    _, _, size, offset = min(layouts, key=lambda row: row[:2])
    out = np.empty((size[1], size[0], 4), dtype=np.uint8)
    out[:] = images[0][0, 0]
    transforms = []
    for tile, origin, source_size, at in zip(tiles, origins, sizes, (np.zeros(2, dtype=int), offset)):
        h, w = tile.shape[:2]
        out[at[1]:at[1] + h, at[0]:at[0] + w] = tile
        transforms.append((source_size / size, (at - origin) / size))
    out = out[::-1].copy()
    # Padding/cropping changes alpha proportions. Use the export's existing rules, so neither its blend decision
    # nor its fully-transparent repair can change when two images become one (even with the same alpha class).
    def alpha_settings(p):
        written = gate([p], fmt)
        return (choose({'alpha_mode': 'BLEND'}, p[..., 3])['mode'], written['format'], written['transparent'],
                draws_nothing([p], fmt, 'Test'), draws_nothing([p], fmt, 'Blend'))
    signatures = [alpha_settings(p) for p in [*images, out]]
    if any(None in s for s in signatures) or len(set(signatures)) != 1:
        return None
    return out, transforms


def shared_formats(formats, images):
    """The GX formats two images sharing an atlas may be written in, drawing each as well as its own format did,
    in order of preference: their own, when they share one; else CI8 where both may be (their alpha class has a
    palette) and their colours together fit its 256 exactly; else RGBA8 where either is, which holds anything; else
    the direct colours of their alpha class, RGB565 opaque and RGB5A3 with alpha, within a step of 5 bits of every
    texel, as CI8's palettes and CMPR's end colours are. (Waddle Dee on Kirby, who has one place for a body: a CI8
    face, a CMPR bandana and spear.) None at all where their alpha classes differ, which pack refuses too."""
    if len(set(formats)) == 1:
        return [formats[0]]
    classes = {alpha_class(p) for p in images}
    if len(classes) != 1:
        return []
    kind = classes.pop()
    if any(f.split(':')[0] == 'RGBA8' for f in formats):
        return ['RGBA8']
    out = []
    palette = {'none': 'RGB565', 'cutout': 'RGB5A3'}.get(kind)
    if palette and len(np.unique(np.concatenate([p.reshape(-1, 4) for p in images]), axis=0)) <= 256:
        out.append(f'CI8:{palette}')
    return out + ['RGB565' if kind == 'none' else 'RGB5A3']


def _texture(material):
    tree = material.node_tree if material else None
    if tree is None or tree.animation_data or any(
            n.type not in {'BSDF_PRINCIPLED', 'OUTPUT_MATERIAL', 'TEX_IMAGE', 'MIX', 'VERTEX_COLOR'}
            for n in tree.nodes):
        return None
    textures = [n for n in tree.nodes if n.type == 'TEX_IMAGE']
    if len(textures) != 1:
        return None
    node = textures[0]
    if (node.image is None or node.image.source not in {'FILE', 'GENERATED'} or not all(node.image.size)
            or node.projection != 'FLAT' or node.inputs['Vector'].is_linked
            or node.interpolation not in {'Linear', 'Closest'}):
        return None
    # glTF's vertex-colour multiply is a MIX between the image and Principled. Its graph/settings and colour
    # layer must match too; remapping the one image's UVs leaves these per-vertex colours untouched.
    todo, seen, reaches_colour = [node], set(), False
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        for link in (l for l in tree.links if l.from_node == current):
            if link.to_node.type == 'MIX':
                todo.append(link.to_node)
            elif link.to_node.type == 'BSDF_PRINCIPLED' and link.to_socket.name in {'Base Color', 'Alpha'}:
                reaches_colour = True
            else:
                return None
    return node if reaches_colour else None


def merge_images(meshes, protected, materials, joint, formats, settings):
    """Put two looks on one image, or return None. The caller rechecks the actual slot planner.

    A look is the materials drawing one image with one wrap mode and shader; they may differ in culling, and each
    keeps its own. Two looks can share an image when their shaders agree but for wrap mode, which the atlas lays out
    (``pack``), on a joint both are on, in a format holding both (shared_formats). A pair keeping both formats goes
    first; then the smallest. Every material of the two then draws the atlas, clamped, each UV inside its image's tile."""
    import bpy
    used = sorted({m for o in meshes for m in materials(o) if m is not None and m not in protected},
                  key=lambda m: m.name)
    looks = {}
    for m in used:
        node = _texture(m)
        if node is None or m.name not in formats:
            continue
        uses, uv, joints = [], [], set()
        for o in meshes:
            if m not in materials(o):
                continue
            # Shared mesh datablocks would be transformed twice; let those keep their own materials.
            layer = next((l for l in o.data.uv_layers if l.active_render), None)
            if layer is None or o.data.users != 1:
                break
            faces = [p for p in o.data.polygons if o.material_slots[p.material_index].material == m]
            loops = [i for p in faces for i in p.loop_indices]
            uv.extend(tuple(layer.data[i].uv) for i in loops)
            uses.append((o, layer, loops))
            joints.add(joint(o))
        else:
            if uv:
                shader = (settings(m, lambda _: None, ignore={'use_backface_culling', 'extension'}),
                          node.image.colorspace_settings.name, node.image.alpha_mode)
                look = looks.setdefault((shader, image_identity(node.image), node.extension),
                                        {'shader': shader, 'node': node, 'materials': [], 'uses': [], 'uv': [],
                                         'joints': set()})
                look['materials'].append(m)
                look['uses'] += uses
                look['uv'] += uv
                look['joints'] |= joints
    candidates = []
    for a, b in combinations(looks.values(), 2):
        if a['shader'] != b['shader'] or not a['joints'] & b['joints']:
            continue
        for look in (a, b):
            if 'pixels' not in look:
                look['pixels'] = read_pixels(look['node'].image)
                look['format'] = formats[look['materials'][0].name]
        for fmt in shared_formats([a['format'], b['format']], [a['pixels'], b['pixels']]):
            packed = pack([a['pixels'], b['pixels']],
                          [(np.min(look['uv'], axis=0), np.max(look['uv'], axis=0)) for look in (a, b)],
                          [a['node'].extension, b['node'].extension], fmt)
            if packed:
                changes = (fmt != a['format']) + (fmt != b['format'])
                candidates.append((changes, packed[0].size, a['materials'][0].name, b['materials'][0].name, a, b,
                                   packed, fmt))
                break
    if not candidates:
        return None
    _, _, _, _, a, b, (pixels_out, transforms), fmt = min(candidates, key=lambda row: row[:4])
    h, w = pixels_out.shape[:2]
    first = a['node']
    image = bpy.data.images.new(a['materials'][0].name + '-atlas', width=w, height=h, alpha=True)
    image.colorspace_settings.name = first.image.colorspace_settings.name
    image.alpha_mode = first.image.alpha_mode
    write_pixels(image, pixels_out)
    for look, (scale, offset) in zip((a, b), transforms):
        for o, layer, loops in look['uses']:
            for i in loops:
                layer.data[i].uv = np.asarray(layer.data[i].uv) * scale + offset
        for m in look['materials']:
            node = _texture(m)
            node.image = image
            node.extension = 'EXTEND'
    record = {'materials': [m.name for m in a['materials'] + b['materials']], 'format': fmt, 'size': [w, h],
              'source_sizes': [list(look['pixels'].shape[1::-1]) for look in (a, b)],
              'image_bytes': gx_bytes(fmt.split(':')[0], w, h),
              'source_image_bytes': [gx_bytes(look['format'].split(':')[0], *look['pixels'].shape[1::-1])
                                     for look in (a, b)],
              'loss': 'Separate images and unused source texels. No resampling, geometry or weight removal; '
                      're-encoding and filtering can change appearance and require an in-game check.'}
    if a['format'] != b['format']:
        record['source_formats'] = [a['format'], b['format']]
    return record
