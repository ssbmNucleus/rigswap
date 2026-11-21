"""load.texture: a texture transform (KHR_texture_transform) is part of a material's UV mapping, and without it the
file's UVs mean nothing. Blender's importers keep it as a Mapping node, which nothing after load reads, so load bakes it
in."""

DRAWN = ('Base Color', 'Alpha', 'Emission Color')       # what a material draws in the game (melee prep.textures)


def _mapping(tree):
    """The Mapping node of mesh UVs behind what this material draws, if any. A face holds one set of UVs, so it's the
    first drawn input's. A transform on anything else (a normal map) moves nothing the game draws."""
    for name in DRAWN:
        todo = [link.from_node for shader in tree.nodes if shader.type == 'BSDF_PRINCIPLED'
                for link in shader.inputs[name].links]
        seen = set()
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            links = node.inputs['Vector'].links if node.type == 'MAPPING' else ()
            if links and links[0].from_socket.name == 'UV':
                return node
            todo.extend(link.from_node for socket in node.inputs for link in socket.links)
    return None


def bake():
    """A material's transform goes into the active UVs of the faces drawing it, the ones the build reads, and its
    Mapping node goes. The image then reads those UVs. -> [{'material', 'node', 'meshes'}]."""
    import bpy
    from mathutils import Euler, Matrix, Vector
    meshes = {o.data for o in bpy.context.scene.objects if o.type == 'MESH' and o.data.uv_layers}
    record = []
    for material in bpy.data.materials:
        node = _mapping(material.node_tree) if material.node_tree else None
        if node is None:
            continue
        source = getattr(node.inputs['Vector'].links[0].from_node, 'uv_map', '')
        location, rotation, scale = (node.inputs[n].default_value for n in ('Location', 'Rotation', 'Scale'))
        matrix = Matrix.LocRotScale(Vector(location), Euler(rotation), Vector(scale))
        if node.vector_type == 'TEXTURE':       # the transform of the texture, not of its coordinates
            matrix.invert()
        users = [me for me in meshes if material in me.materials[:]]
        for me in users:
            uvs = me.uv_layers.get(source) or me.uv_layers.active
            for face in me.polygons:
                if me.materials[face.material_index] == material:
                    for i in face.loop_indices:
                        me.uv_layers.active.data[i].uv = (matrix @ Vector((*uvs.data[i].uv, 0))).to_2d()
        record.append({'material': material.name, 'node': node.name, 'meshes': sorted(me.name for me in users)})
        material.node_tree.nodes.remove(node)
    return record
