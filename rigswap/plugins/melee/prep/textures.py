"""1. textures: what the game never draws taken out of each material (drawn_only); a material that draws one flat
colour given it as a one-texel image (colour_textures); every image a material uses made a power of two on each
side, at most TEXTURE_MAX_SIDE, and each image the game draws given the GX format HSDRawViewer will write it in
(limit_textures); texture_formats is the map of them that export passes to the import."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


# The longest side a texture is written at, in pixels. Vanilla fighters' largest are 256 (Mario, Bowser, Peach), and
# large textures are what take a costume over its memory (memory.py). (randall model_import/runner.py MAX_TEXTURE)
TEXTURE_MAX_SIDE = 256
# Every side is also made the nearest power of two. GX repeats a texture only along a side that is a power of two, and
# export preserves the source sampler after HSDRawViewer's import (the .dae it reads loses wrap modes). Vanilla costumes have
# other sizes (Mario a 190x190, Peach a 110x220) but clamp along every such side: all 191 such textures in the 25
# vanilla Pl<code>Nr.dat files, 2026-10-10.
# The memory step never halves a texture below this; vanilla fighters' largest textures are 128 or 256.
# (shared/modellab/native_import.py TEXTURE_FLOOR_MIN)
TEXTURE_MIN_SIDE = 128

# GX texture formats: (bits per pixel, block width, block height). An image is stored in whole blocks of 32 bytes (64
# for RGBA8). CI8 also stores a palette of 256 colours of 16 bits. (HSDRaw GXTexFmt; randall shared/rendering/
# datprobe.py gx_image_size)
GX_FORMATS = {'CMPR': (4, 8, 8), 'CI8': (8, 8, 4), 'RGB565': (16, 4, 4), 'RGB5A3': (16, 4, 4), 'RGBA8': (32, 4, 4)}
CI8_PALETTE_BYTES = 512

# Left alone, HSDRawViewer's import writes a texture as CMPR (4 bits a pixel, two colours and one bit of alpha per 4x4
# block), or as RGB5A3 when it has partial alpha: some alpha above 8 and below 248, looking at every 64th of each side.
# prep decides the same way, so the memory estimate is the import's. (HSDRawViewer/Converters/ModelImporter.cs
# HasPartialAlpha)
ALPHA_SAMPLES = 64
ALPHA_PARTIAL = (8, 248)

# CI8 (256 colours) draws flat, hard-edged art exactly where CMPR steps its edges; vanilla fighters' eyes and faces are
# CI8. A texture is written as CI8 when its alpha class may be (ALPHA_CLASSES: not partial alpha), it is at most
# CI8_MAX_SIDE, CMPR gets at least CI8_MIN_BAD of its pixels visibly wrong and CI8 at most a CI8_GAIN-th as many, and
# the CI8 textures add at most CI8_MAX_EXTRA bytes over CMPR in all. They go back to CMPR first when the costume is over
# its memory. A pixel is visibly wrong when a channel is off by more than VISIBLE_ERROR of 255.
# (shared/modellab/texture_formats.py AUTO_*, BAD)
CI8_MAX_SIDE = 256
CI8_MIN_BAD = 0.025
CI8_GAIN = 3.0
CI8_MAX_EXTRA = 256 * 1024
VISIBLE_ERROR = 16


# The alpha classes (alpha_class), one row each, with what follows from each: the formats it may take, and how a
# resize keeps it in its class.
#   own               its format unless it is written as CI8: the import's own for it, CMPR, or RGB5A3 for partial
#                     alpha, which CMPR drops
#   why               why it has its own format (prep.json), when CI8 is not weighed
#   ci8_palette       the palette it is written with as CI8, for hard-edged art (CI8_*; choose_format weighs CI8
#                     against CMPR); None: never CI8
#   cut_after_resize  resized (resample), its alpha is cut here, at or above opaque and below clear, so that it stays
#                     in its class; None: kept as it comes. A resized cutout gets soft edges, which the import would
#                     write as RGB5A3 at four times the bytes, so it is cut again at 128; an opaque image is cut at 0,
#                     every pixel opaque. (shared/modellab/preload_heap.py resample_texture: Game & Watch's costume
#                     grew when cutouts were halved)
@dataclass(frozen=True)
class AlphaClass:
    own: str
    why: str
    ci8_palette: str | None
    cut_after_resize: int | None


ALPHA_CLASSES = {
    'none': AlphaClass(own='CMPR', why='opaque or cutout', ci8_palette='RGB565', cut_after_resize=0),
    'cutout': AlphaClass(own='CMPR', why='opaque or cutout', ci8_palette='RGB5A3', cut_after_resize=128),
    'partial': AlphaClass(own='RGB5A3', why='partial alpha, which CMPR drops', ci8_palette=None, cut_after_resize=None),
}


# ---- 1. textures -----------------------------------------------------------------------------------------------------

def drop_missing_images(meshes):
    """A material whose alpha comes from an image the source lacks is not drawn: its cutout is unknowable.
    Delete its faces and say so in prep's notes. Blender holds a named missing image with no size. Missing images
    used only for colour are unlinked, leaving the material's own colour (colour_textures). -> record per image"""
    import bmesh
    materials = {o.material_slots[p.material_index].material for o in meshes for p in o.data.polygons
                 if p.material_index < len(o.material_slots) and o.material_slots[p.material_index].material is not None}
    record, hidden = [], set()
    for m in sorted(materials, key=lambda m: m.name):
        alpha_nodes, pending = set(), []
        for shader in (m.node_tree.nodes if m.node_tree else []):
            if shader.type == 'BSDF_PRINCIPLED':
                pending.extend(link.from_node for link in shader.inputs['Alpha'].links)
        while pending:
            node = pending.pop()
            if node not in alpha_nodes:
                alpha_nodes.add(node)
                pending.extend(link.from_node for socket in node.inputs for link in socket.links)
        for node in (m.node_tree.nodes if m.node_tree else []):
            if node.type != 'TEX_IMAGE' or node.image is None or (node.image.size[0] and node.image.size[1]):
                continue
            links = [link for output in node.outputs for link in output.links]
            if links:
                row = {'material': m.name, 'image': node.image.name,
                       'unlinked': sorted({f'{link.to_node.name}: {link.to_socket.name}' for link in links})}
                if node in alpha_nodes:
                    hidden.add(m)
                    row.update(drawn=False, said=f'material {m.name} is not drawn: its alpha image '
                               f'{node.image.name} is missing from the source')
                record.append(row)
            for link in links:
                m.node_tree.links.remove(link)
    for ob in meshes:
        slots = {i for i, slot in enumerate(ob.material_slots) if slot.material in hidden}
        faces = [p.index for p in ob.data.polygons if p.material_index in slots]
        if faces:
            bm = bmesh.new()
            bm.from_mesh(ob.data)
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[bm.faces[i] for i in faces], context='FACES')
            bm.to_mesh(ob.data)
            bm.free()
            ob.data.update()
    return record


DRAWN_INPUTS = ('Base Color', 'Alpha', 'Emission Color', 'Emission Strength')


def drawn_only(meshes):
    """Leave each material of ``meshes`` what the game draws of it, so materials that draw the same compare the same
    (join.py's identity, atlas.py's one image). The import draws a Principled BSDF's Base Color and Alpha, and its
    emission where the base is black (export/write.py _emission_to_diffuse): a node reaching the shader only through
    another input (a normal map, roughness, specular) goes. An Alpha taken from a second image node showing the Base
    Color's own alpha, sampled the same way (mia-Samus: the same picture read again as non-colour data), is taken from
    the Base Color's node instead, and the copy goes. A material whose output is not a Principled BSDF, or nothing,
    is left as it is. -> record, one row per material changed"""
    materials = {o.material_slots[p.material_index].material for o in meshes for p in o.data.polygons
                 if p.material_index < len(o.material_slots) and o.material_slots[p.material_index].material is not None}
    record = []
    for m in sorted(materials, key=lambda m: m.name):
        tree = m.node_tree
        outputs = [n for n in (tree.nodes if tree else []) if n.type == 'OUTPUT_MATERIAL']
        if not outputs or any(link.from_node.type != 'BSDF_PRINCIPLED'
                              for n in outputs for link in n.inputs['Surface'].links):
            continue
        shaders = {link.from_node for n in outputs for link in n.inputs['Surface'].links}
        if not shaders:
            continue
        row = {}
        for shader in sorted(shaders, key=lambda n: n.name):
            colour, alpha = _linked_image(shader, 'Base Color', 'Color'), _linked_image(shader, 'Alpha', 'Alpha')
            if colour is not None and alpha is not None and colour != alpha and _same_alpha(colour, alpha):
                tree.links.new(colour.outputs['Alpha'], shader.inputs['Alpha'])
                row['alpha_from_base_color'] = alpha.image.name
        keep = set(outputs) | shaders
        todo = [link.from_node for s in shaders for name in DRAWN_INPUTS if name in s.inputs
                for link in s.inputs[name].links]
        while todo:
            node = todo.pop()
            if node not in keep:
                keep.add(node)
                todo.extend(link.from_node for socket in node.inputs for link in socket.links)
        gone = sorted((n for n in tree.nodes if n not in keep and n.type != 'FRAME'), key=lambda n: n.name)
        if gone:
            row['nodes'] = [f'{n.name} ({n.type})' for n in gone]
            for n in gone:
                tree.nodes.remove(n)
        for link in [link for link in tree.links if link.to_node in keep and
                     (link.to_node in shaders and link.to_socket.name not in DRAWN_INPUTS or
                      link.to_node.type == 'OUTPUT_MATERIAL' and link.to_socket.name != 'Surface')]:
            tree.links.remove(link)
        if row:
            record.append({'material': m.name, **row})
    return record


def _linked_image(shader, socket, output):
    """The image node linked straight into ``socket`` from its ``output``, or None."""
    links = shader.inputs[socket].links
    node = links[0].from_node if links else None
    if node is None or node.type != 'TEX_IMAGE' or node.image is None or links[0].from_socket.name != output:
        return None
    return node


def _same_alpha(a, b):
    """Two image nodes drawing the same alpha: the same texels' alpha, sampled the same way through the same UVs."""
    if tuple(a.image.size) != tuple(b.image.size) or not all(a.image.size):
        return False
    if any(getattr(a, p) != getattr(b, p) for p in ('extension', 'interpolation', 'projection')):
        return False
    via = [[(link.from_node, link.from_socket.identifier) for link in n.inputs['Vector'].links] for n in (a, b)]
    return via[0] == via[1] and np.array_equal(read_pixels(a.image)[..., 3], read_pixels(b.image)[..., 3])


def colour_textures(meshes):
    """Give each material of ``meshes`` that draws one flat colour that colour as a one-texel image on Base Color.

    HSDRawViewer's import makes a material from its diffuse texture alone (ModelImporter.cs GenerateMaterial, with
    ImportMaterialInfo off): a material with no image gets the import's white, whatever colour the .dae gives it, so a
    model coloured by its materials (Super Mario 64's rips: red cap, blue overalls) comes out white. A Principled BSDF
    whose Base Color is not linked draws its socket's colour everywhere; the same colour as one opaque sRGB texel
    (load/materials.py: colour_texel) draws it in the game, whatever the UVs. White, the import's own, stays as it is.
    Left as they are: a Base Color linked to anything (an image, vertex colours, a procedural), a shader whose Alpha or
    Emission Color is linked (its image is what the export draws: export/write.py _emission_to_diffuse), and a
    material with no Principled BSDF. Materials of one colour share one image. -> record, one row per material"""
    import bpy
    from ....load.materials import colour_texel
    materials = {o.material_slots[p.material_index].material for o in meshes for p in o.data.polygons
                 if p.material_index < len(o.material_slots) and o.material_slots[p.material_index].material is not None}
    images, record, coloured = {}, [], set()
    for m in sorted(materials, key=lambda m: m.name):
        shaders = [n for n in (m.node_tree.nodes if m.node_tree else []) if n.type == 'BSDF_PRINCIPLED']
        for shader in shaders:
            base = shader.inputs['Base Color']
            if base.is_linked or shader.inputs['Alpha'].is_linked or shader.inputs['Emission Color'].is_linked:
                continue
            texel = colour_texel(tuple(base.default_value))
            if texel[:3] == (255, 255, 255):
                continue
            if texel not in images:
                image = bpy.data.images.new('colour-{:02x}{:02x}{:02x}'.format(*texel), width=1, height=1, alpha=True)
                image.colorspace_settings.name = 'sRGB'
                image.pixels.foreach_set(np.asarray(texel, dtype=np.float32) / 255)
                image.update()
                image.file_format = 'PNG'
                image.pack()
                images[texel] = image
            node = m.node_tree.nodes.new('ShaderNodeTexImage')
            node.image = images[texel]
            m.node_tree.links.new(node.outputs['Color'], base)
            coloured.add(m)
            record.append({'material': m.name, 'colour': [round(float(c), 4) for c in base.default_value[:3]],
                           'texel': list(texel[:3]), 'image': images[texel].name})
    for o in meshes:        # a texture is drawn through UVs: a mesh with none gets a layer (one texel reads the same)
        if not o.data.uv_layers and any(s.material in coloured for s in o.material_slots):
            o.data.uv_layers.new()
    return record


@dataclass
class Texture:
    """One image the game draws, and how it will be written."""
    image: object               # the bpy image
    materials: list             # names of the materials that draw it
    pixels: np.ndarray          # (h, w, 4) uint8, top row first, as the build had it
    alpha: str                  # its alpha class (alpha_class): a key of ALPHA_CLASSES
    size: tuple                 # (w, h) as it will be written
    format: str = 'CMPR'
    why: str = ''
    automatic: bool = False     # CI8 picked for hard edges; given back for CMPR first when memory is short

    @property
    def name(self):
        return self.image.name

    def bytes(self):
        return gx_bytes(self.format, *self.size)

    def code(self):
        """The format as --texture-formats takes it: CI8 names its palette, its alpha class's."""
        if self.format == 'CI8':
            return 'CI8:' + ALPHA_CLASSES[self.alpha].ci8_palette
        return self.format

    def record(self):
        h, w = self.pixels.shape[:2]
        return {'image': self.name, 'materials': self.materials, 'size_in_build': [w, h], 'size': list(self.size),
                'alpha': self.alpha, 'format': self.code(), 'why': self.why, 'bytes': self.bytes()}


def gx_bytes(fmt, width, height):
    """The bytes GX stores a ``width`` x ``height`` image in, in format ``fmt``."""
    bits, bw, bh = GX_FORMATS[fmt]
    blocks = -(-int(width) // bw) * -(-int(height) // bh)
    return blocks * bw * bh * bits // 8 + (CI8_PALETTE_BYTES if fmt == 'CI8' else 0)


def limit_textures(meshes, max_side=TEXTURE_MAX_SIDE):
    """Every image a material of ``meshes`` uses, made a power of two on each side and at most ``max_side`` on its
    long side (see game_size), and for each image the game draws (what feeds a material's Base Color, which the import
    takes as the texture) its GX format. -> ([Texture], record)"""
    drawn, used = _material_images(meshes)
    record = {'max_side': max_side, 'resized': [], 'not_drawn': [], 'no_pixels': []}
    textures, extra = [], 0
    for image in sorted(used, key=lambda i: i.name):
        if not (image.size[0] and image.size[1]):
            record['no_pixels'].append(image.name)
            continue
        pixels = read_pixels(image)
        alpha = alpha_class(pixels)
        h, w = pixels.shape[:2]
        size = game_size((w, h), max_side)
        if size != (w, h):
            write_pixels(image, resample(pixels, size, alpha))
            record['resized'].append({'image': image.name, 'from': [w, h], 'to': list(size)})
        if image not in drawn:
            record['not_drawn'].append(image.name)
            continue
        t = Texture(image, sorted(drawn[image]), pixels, alpha, size)
        extra += choose_format(t, extra)
        textures.append(t)
    record['formats'] = dict(Counter(t.code() for t in textures))
    return textures, record


def _material_images(meshes):
    """({image: names of the materials that draw it}, every image a material of ``meshes`` uses)."""
    drawn, used = {}, set()
    materials = {o.material_slots[p.material_index].material for o in meshes for p in o.data.polygons
                 if p.material_index < len(o.material_slots) and o.material_slots[p.material_index].material is not None}
    for m in sorted(materials, key=lambda m: m.name):
        if m.node_tree is None:
            continue
        nodes = _all_nodes(m.node_tree)
        used.update(n.image for n in nodes if n.type == 'TEX_IMAGE' and n.image is not None)
        image = _base_color_image(m.node_tree)
        if image is not None:
            drawn.setdefault(image, []).append(m.name)
    return drawn, used


def _all_nodes(tree, seen=None):
    seen = set() if seen is None else seen
    out = []
    for n in tree.nodes:
        out.append(n)
        if n.type == 'GROUP' and n.node_tree is not None and n.node_tree.name not in seen:
            seen.add(n.node_tree.name)
            out += _all_nodes(n.node_tree, seen)
    return out


def _base_color_image(tree):
    """The nearest image upstream of a Principled BSDF's Base Color, or of the material's only image node."""
    for n in tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            todo = [link.from_node for link in n.inputs['Base Color'].links]
            seen = set()
            while todo:
                node = todo.pop(0)
                if node.name in seen:
                    continue
                seen.add(node.name)
                if node.type == 'TEX_IMAGE' and node.image is not None:
                    return node.image
                todo += [link.from_node for i in node.inputs for link in i.links]
    images = [n.image for n in tree.nodes if n.type == 'TEX_IMAGE' and n.image is not None]
    return images[0] if len(images) == 1 else None


def read_pixels(image):
    """An image's pixels as (h, w, 4) uint8, top row first."""
    w, h = image.size
    values = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(values)
    return np.round(np.clip(values, 0, 1) * 255).astype(np.uint8).reshape(h, w, 4)[::-1].copy()


def write_pixels(image, pixels):
    """Put (h, w, 4) uint8 ``pixels``, top row first, into ``image`` and pack it into the .blend as a PNG."""
    h, w = pixels.shape[:2]
    if tuple(image.size) != (w, h):
        image.scale(w, h)
    image.pixels.foreach_set((pixels[::-1].astype(np.float32) / 255).ravel())
    image.update()
    image.file_format = 'PNG'
    image.pack()


def alpha_class(pixels):
    """'none' when every pixel is opaque, 'partial' when the import would call it partial alpha, else 'cutout': its row
    of ALPHA_CLASSES."""
    a = pixels[..., 3]
    if (a >= 255).all():
        return 'none'
    h, w = a.shape
    sample = a[::max(1, h // ALPHA_SAMPLES), ::max(1, w // ALPHA_SAMPLES)]
    return 'partial' if ((sample > ALPHA_PARTIAL[0]) & (sample < ALPHA_PARTIAL[1])).any() else 'cutout'


def fit_inside(size, longest):
    """``size`` made at most ``longest`` on its long side, aspect kept; never larger."""
    w, h = size
    if max(w, h) <= longest:
        return (w, h)
    return (max(1, w * longest // max(w, h)), max(1, h * longest // max(w, h)))


def game_size(size, longest=TEXTURE_MAX_SIDE):
    """The size a texture is written at: fit inside ``longest``, then each side the nearest power of two (190 becomes
    256, 88 becomes 64), which keeps the aspect as near as powers of two can."""
    return tuple(min(longest, 2 ** round(math.log2(side))) for side in fit_inside(size, longest))


def resample(pixels, size, alpha):
    """``pixels`` resized to ``size`` (Lanczos), keeping the alpha class ``alpha``: its alpha cut where its row of
    ALPHA_CLASSES says."""
    from PIL import Image
    out = np.asarray(Image.fromarray(pixels, 'RGBA').resize(tuple(size), Image.LANCZOS)).copy()
    cut = ALPHA_CLASSES[alpha].cut_after_resize
    if cut is not None:
        out[..., 3] = np.where(out[..., 3] >= cut, 255, 0)
    return out


def choose_format(t, extra_so_far):
    """Set ``t``'s format and why: CI8 for hard-edged art (see CI8_*) when its alpha class may be CI8, else the class's
    own (ALPHA_CLASSES). -> the bytes CI8 adds over CMPR, or 0."""
    row = ALPHA_CLASSES[t.alpha]
    if row.ci8_palette is None or max(t.size) > CI8_MAX_SIDE:
        t.format, t.why = row.own, row.why
        return 0
    pixels = resample(t.pixels, t.size, t.alpha) if tuple(t.size) != t.pixels.shape[1::-1] else t.pixels
    shown = pixels[..., 3] >= 128
    if not shown.any():
        shown[:] = True
    cmpr, ci8 = cmpr_error(pixels)[shown], ci8_error(pixels)[shown]
    cmpr_bad, ci8_bad = float((cmpr > VISIBLE_ERROR).mean()), float((ci8 > VISIBLE_ERROR).mean())
    added = gx_bytes('CI8', *t.size) - gx_bytes('CMPR', *t.size)
    if cmpr_bad >= CI8_MIN_BAD and ci8_bad * CI8_GAIN <= cmpr_bad and extra_so_far + added <= CI8_MAX_EXTRA:
        t.format, t.automatic = 'CI8', True
        t.why = f'hard edges: {cmpr_bad:.1%} of pixels visibly off as CMPR, {ci8_bad:.1%} as CI8'
        return added
    t.format = 'CMPR'
    t.why = (f'CMPR draws it about as well ({cmpr_bad:.1%} of pixels visibly off, CI8 {ci8_bad:.1%})'
             if cmpr_bad < CI8_MIN_BAD or ci8_bad * CI8_GAIN > cmpr_bad
             else f'hard edges, but CI8 textures already add {extra_so_far:,} bytes')
    return 0


def cmpr_error(pixels):
    """Per pixel, the largest channel error CMPR (DXT1) would make: each 4x4 block drawn with two end colours in RGB565
    on the line its colours spread along most, and the two colours a third of the way between them."""
    h, w = pixels.shape[:2]
    rgb = np.pad(pixels[..., :3].astype(np.float32), ((0, -h % 4), (0, -w % 4), (0, 0)), mode='edge')
    rows, cols = rgb.shape[0] // 4, rgb.shape[1] // 4
    blocks = rgb.reshape(rows, 4, cols, 4, 3).swapaxes(1, 2).reshape(-1, 16, 3)
    centre = blocks.mean(axis=1, keepdims=True)
    spread = blocks - centre
    _, vectors = np.linalg.eigh(np.einsum('bki,bkj->bij', spread, spread))
    line = vectors[:, :, -1]                                         # the direction the colours spread along most
    along = np.einsum('bki,bi->bk', spread, line)
    ends = [centre[:, 0] + along.min(axis=1)[:, None] * line, centre[:, 0] + along.max(axis=1)[:, None] * line]
    a, b = (np.round(np.clip(e, 0, 255) / 255 * [31, 63, 31]) / [31, 63, 31] * 255 for e in ends)
    palette = np.stack([a, b, (2 * a + b) / 3, (a + 2 * b) / 3], axis=1)
    error = np.abs(blocks[:, :, None, :] - palette[:, None, :, :]).max(axis=-1).min(axis=-1)
    return error.reshape(rows, cols, 4, 4).swapaxes(1, 2).reshape(rows * 4, cols * 4)[:h, :w]


def ci8_error(pixels):
    """Per pixel, the largest channel error of a 256-colour palette (median cut, no dithering)."""
    from PIL import Image
    rgb = Image.fromarray(np.ascontiguousarray(pixels[..., :3]), 'RGB')
    quantised = rgb.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB')
    return np.abs(pixels[..., :3].astype(np.int16) - np.asarray(quantised, np.int16)).max(axis=-1)


def texture_formats(textures, meshes=None):
    """{material: format} for every material that draws a texture: the --texture-formats map. With ``meshes``, only
    the materials still on one of their faces: step 5 (normals) can take every face of a material out (a material
    drawn only on specks), and export fails on a format named for a material the file no longer has."""
    used = None if meshes is None else {o.material_slots[p.material_index].material.name for o in meshes
                                        for p in o.data.polygons if p.material_index < len(o.material_slots)
                                        and o.material_slots[p.material_index].material}
    return {m: t.code() for t in textures for m in t.materials if used is None or m in used}
