"""Shared Workbench scene preparation and labels for the core's visual questions."""
from pathlib import Path

from PIL import ImageFont


TILE = 520                     # px, square, one view in a model's picture


# The label font, by the name each system gives it: Windows' Arial, then macOS's (the same face under its own file
# name: without it a Mac drew every label in Pillow's built-in regular face), then DejaVu, which Linux desktops have.
FONTS = {False: ('arial.ttf', 'Arial.ttf', 'DejaVuSans.ttf'),
         True: ('arialbd.ttf', 'Arial Bold.ttf', 'DejaVuSans-Bold.ttf')}


def _font(size, bold=False):
    for name in FONTS[bool(bold)]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _base_color_image(mat):
    """The image texture node that feeds the material's surface color, followed back through any nodes between."""
    tree = mat.node_tree
    if tree is None:
        return None
    out = next((n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None) or \
        next((n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if out is None:
        return None
    seen = set()

    def back(node):
        if node.name in seen:
            return None
        seen.add(node.name)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node
        inputs = list(node.inputs)
        # a shader's color input first, so a normal map's image is not taken for the color
        inputs.sort(key=lambda s: 0 if s.name in ('Base Color', 'Color', 'Color1', 'A') else 1)
        for sock in inputs:
            if sock.name in ('Normal', 'Tangent'):
                continue
            for link in sock.links:
                found = back(link.from_node)
                if found:
                    return found
        return None

    return back(out)


def _prepare_scene():
    """Every mesh shows with its materials; nothing else renders. Returns the meshes."""
    import bpy
    scene = bpy.context.scene
    meshes = []
    for ob in scene.objects:
        if ob.type == 'MESH':
            ob.hide_render = False
            ob.hide_viewport = False
            ob.hide_set(False)
            meshes.append(ob)
        else:
            ob.hide_render = True
    # Workbench colours a face from its material's active image node; make that the image feeding the surface colour.
    for mat in bpy.data.materials:
        node = _base_color_image(mat)
        if node is not None:
            mat.node_tree.nodes.active = node

    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'TEXTURE'
    shading.show_backface_culling = False
    shading.show_object_outline = False
    shading.show_cavity = False
    shading.show_shadows = False
    scene.display.render_aa = '8'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.film_transparent = True
    scene.render.resolution_x = TILE
    scene.render.resolution_y = TILE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    return [ob for ob in meshes if ob.visible_get()]


def _world_points(meshes):
    """Every vertex of every shown mesh, posed as the file has it, in world space."""
    import bpy
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for ob in meshes:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            n = len(me.vertices)
            if n:
                co = np.empty(n * 3, dtype=np.float64)
                me.vertices.foreach_get('co', co)
                co = co.reshape(n, 3)
                mw = np.array(ev.matrix_world)
                chunks.append(co @ mw[:3, :3].T + mw[:3, 3])
        finally:
            ev.to_mesh_clear()
    return np.concatenate(chunks) if chunks else None


def _draws_set(mat):
    """Whether ``mat``'s surface colour takes a colour set of its mesh: a colour attribute node upstream of its
    shader's Base Color (glTF's importer wires one; load.open._draw_set wires one for an OBJ's or an FBX's)."""
    tree = mat.node_tree if mat is not None else None
    if tree is None:
        return False
    todo = [l.from_node for n in tree.nodes if n.type == 'BSDF_PRINCIPLED' for l in n.inputs['Base Color'].links]
    seen = set()
    while todo:
        node = todo.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == 'VERTEX_COLOR':
            return True
        todo.extend(l.from_node for s in node.inputs if s.name not in ('Normal', 'Tangent') for l in s.links)
    return False


def _linear(c):
    import numpy as np
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _display(c):
    import numpy as np
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - 0.055)


def still():
    """Render the still the scene is set up for, to ``scene.render.filepath``, each face in the colours the costume
    will draw it in. Workbench draws a face in its material's image, or its colour where it has none, and never a
    colour set; the costume draws the set, times the material's colour, on every face whose material draws one
    (load.open.colours). So where a face shown under such a material is seen, its picture is multiplied by the set,
    drawn flat in a pass of its own and laid in through a second pass that says which pixels those faces are. A face
    under another material (an image, a piece greyed or painted for the question) is as Workbench drew it, as is one
    whose set is white there, and a scene whose shown materials draw no set renders once, as before.

    Taz (an OBJ whose fur is a set, 2026-11-19) stood grey in every ask's picture: on his own tile of the look he was
    grey on a grey kart, and the answer left him out with it."""
    import bpy
    import numpy as np
    from PIL import Image
    scene = bpy.context.scene
    bpy.ops.render.render(write_still=True)
    shown = [ob for ob in scene.objects if ob.type == 'MESH' and not ob.hide_render]
    sets = {slot.material for ob in shown for slot in ob.material_slots if _draws_set(slot.material)}
    if not sets:
        return
    path = Path(bpy.path.abspath(scene.render.filepath))
    shading = scene.display.shading
    before = (shading.light, shading.color_type, scene.render.film_transparent, scene.render.filepath, scene.world,
              scene.render.dither_intensity)
    colours = {m: tuple(m.diffuse_color) for m in bpy.data.materials}
    drawn = {}                              # a mesh's set as the costume draws it: the one it renders with
    for me in {ob.data for ob in shown}:
        if len(me.color_attributes):
            drawn[me] = me.color_attributes.active_color_index
            me.color_attributes.active_color_index = me.color_attributes.render_color_index
    behind = bpy.data.worlds.new('core_still')      # which faces, on black; their sets, on white: an edge fades to none
    passes = {}
    try:
        shading.light, scene.render.film_transparent, scene.world = 'FLAT', False, behind
        scene.render.dither_intensity = 0.
        for m in bpy.data.materials:
            m.diffuse_color = (1., 1., 1., 1.) if m in sets else (0., 0., 0., 1.)
        for name, kind, back in (('where', 'MATERIAL', 0.), ('set', 'VERTEX', 1.)):
            shading.color_type, behind.color = kind, (back, back, back)
            scene.render.filepath = str(path.with_name(f'{path.stem}-{name}{path.suffix}'))
            bpy.ops.render.render(write_still=True)
            with Image.open(scene.render.filepath) as img:
                passes[name] = np.asarray(img.convert('RGB'), dtype=float) / 255
            Path(scene.render.filepath).unlink()
    finally:
        (shading.light, shading.color_type, scene.render.film_transparent, scene.render.filepath, scene.world,
         scene.render.dither_intensity) = before
        bpy.data.worlds.remove(behind)
        for m, c in colours.items():
            m.diffuse_color = c
        for me, index in drawn.items():
            me.color_attributes.active_color_index = index
    where = passes['where'][..., :1]
    times = 1 - where + where * _linear(passes['set'])
    coloured = np.abs(times - 1).max(axis=2) > 1 / 255      # a set white there says nothing: the pixel is as it was
    if not coloured.any():
        return
    with Image.open(path) as img:
        mode = img.mode
        out = np.asarray(img.convert('RGBA')).copy()
    plain = out[coloured, :3] / 255
    out[coloured, :3] = np.round(_display(_linear(plain) * times[coloured]) * 255).clip(0, 255)
    Image.fromarray(out, 'RGBA').convert(mode).save(path)
