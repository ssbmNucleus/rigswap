"""Convert an empty colour texture to the material's own colour, before any intermediate export loses it."""
from __future__ import annotations

import numpy as np


def colour_texel(colour):
    """A shader's linear RGB as one opaque sRGB texel. Material opacity stays on the shader's Alpha input."""
    rgb = np.clip(np.asarray(colour[:3], dtype=float), 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return tuple(int(v) for v in np.rint(srgb * 255)) + (255,)


def convert():
    """Imported Principled materials: an all-zero-alpha image linked directly to Base Color, with no shader Alpha
    link, contributes no visible texels. Use the Base Color socket's authored default instead. A texture used for
    transparency, any image with nonzero alpha, and composed colour graphs keep their meaning.

    Keep the colour as a packed one-texel image: texture-only consumers must receive the same colour as Blender.
    Replace only this colour link, never the shared image or its other uses. Record the source and linear colour.
    """
    import bpy
    empty, record = {}, []
    for material in bpy.data.materials:
        tree = material.node_tree
        for shader in list(tree.nodes) if tree else []:
            if shader.type != 'BSDF_PRINCIPLED' or shader.inputs['Alpha'].is_linked:
                continue
            base = shader.inputs['Base Color']
            if not base.is_linked:
                continue
            link = base.links[0]
            node = link.from_node
            if node.type != 'TEX_IMAGE' or node.image is None or link.from_socket.name != 'Color':
                continue
            image = node.image
            if image not in empty:
                pixels = np.empty(len(image.pixels), dtype=np.float32)
                image.pixels.foreach_get(pixels)
                empty[image] = bool(len(pixels) and np.all(pixels[3::4] == 0))
            if not empty[image]:
                continue
            colour = tuple(base.default_value)
            texel = colour_texel(colour)
            solid = bpy.data.images.new(material.name + '-colour', width=1, height=1, alpha=True)
            solid.colorspace_settings.name = 'sRGB'
            solid.pixels.foreach_set(np.asarray(texel, dtype=np.float32) / 255)
            solid.update()
            solid.file_format = 'PNG'
            solid.pack()
            replacement = tree.nodes.new('ShaderNodeTexImage')
            replacement.image = solid
            tree.links.new(replacement.outputs['Color'], base)
            record.append({'material': material.name, 'image': image.name, 'colour': list(colour),
                           'texture': solid.name, 'texel': list(texel)})
    return record
