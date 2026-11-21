"""prep/textures.py: texture sizes and formats."""
import pytest
import numpy as np

from rigswap.plugins.melee.prep import memory as prep_memory, textures as prep_textures
from tests.plugins.melee.prep.test_prep import _empty_scene, _flat, _grid, _material, _mesh


@pytest.mark.blender
def test_an_oversize_texture_is_resized_and_each_drawn_texture_gets_its_format():
    _empty_scene()
    gradient = _flat(512, 256)
    gradient[..., 0] = np.linspace(0, 255, 512).astype(np.uint8)[None, :]
    soft = _flat(32, 32)
    soft[..., 3] = 128                                                   # partial alpha
    y, x = np.mgrid[0:64, 0:64]
    stripes = np.array([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)], np.uint8)[(x + 2 * y) % 3]
    materials = [_material('big', gradient), _material('soft', soft), _material('stripes', stripes)]
    ob = _grid('body', 2, materials)

    textures, record = prep_textures.limit_textures([ob])
    by_name = {t.name: t for t in textures}
    assert by_name['big'].size == (256, 128) and tuple(by_name['big'].image.size) == (256, 128)
    assert by_name['big'].image.packed_file is not None
    assert record['resized'] == [{'image': 'big', 'from': [512, 256], 'to': [256, 128]}]
    assert by_name['big'].code() == 'CMPR'
    assert by_name['soft'].code() == 'RGB5A3' and by_name['soft'].alpha == 'partial'
    assert by_name['stripes'].code() == 'CI8:RGB565'                    # three colours CMPR cannot draw in one block
    assert prep_textures.texture_formats(textures) == {'big': 'CMPR', 'soft': 'RGB5A3', 'stripes': 'CI8:RGB565'}


@pytest.mark.blender
def test_a_material_of_one_flat_colour_draws_it_through_a_one_texel_image():
    """The import draws a material's texture only (HSDRawViewer ImportMaterialInfo off): a Base Color left at a
    colour reaches the game as one texel of that colour in sRGB; white, the import's own, and linked colours stay."""
    bpy = _empty_scene()

    def plain(name, colour):
        m = bpy.data.materials.new(name)
        next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = colour
        return m
    cap, cap_again, glove = plain('cap', (1, 0, 0, 1)), plain('cap again', (1, 0, 0, 1)), plain('glove', (1, 1, 1, 1))
    skin = plain('skin', (0.991, 0.533, 0.191, 1))
    painted = _material('painted', _flat(4, 4))
    body = _grid('body', 2, [cap, glove, painted, cap_again])
    bare = _mesh('bare', [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], [skin])     # no UVs

    record = prep_textures.colour_textures([body, bare])
    assert [(r['material'], r['texel'], r['image']) for r in record] == [
        ('cap', [255, 0, 0], 'colour-ff0000'), ('cap again', [255, 0, 0], 'colour-ff0000'),
        ('skin', [254, 193, 121], 'colour-fec179')]
    image = prep_textures._base_color_image(cap.node_tree)
    assert image is prep_textures._base_color_image(cap_again.node_tree) and tuple(image.size) == (1, 1)
    assert image.packed_file is not None and prep_textures.read_pixels(image).tolist() == [[[255, 0, 0, 255]]]
    assert prep_textures._base_color_image(glove.node_tree) is None
    assert prep_textures._base_color_image(painted.node_tree).name == 'painted'
    assert len(bare.data.uv_layers) == 1 and len(body.data.uv_layers) == 1


def test_every_texture_side_becomes_the_nearest_power_of_two_within_the_limit():
    assert prep_textures.game_size((190, 190)) == (256, 256)
    assert prep_textures.game_size((88, 40)) == (64, 32)
    assert prep_textures.game_size((2048, 1024)) == (256, 128)
    assert prep_textures.game_size((1024, 300)) == (256, 64)               # 256 x 75, then powers of two
    assert prep_textures.game_size((64, 64)) == (64, 64)
    assert prep_memory.halved((256, 128)) == (128, 64) and prep_memory.halved((128, 256)) == (64, 128)
    assert prep_memory.halved((128, 64)) is None


@pytest.mark.blender
def test_a_material_keeps_only_what_the_game_draws():
    """mia-Samus: a Normal Map node on the shader and the Alpha read again from the Base Color's own picture (as
    non-colour data) made every material a graph join.py doesn't know, its own identity. The import draws neither."""
    from rigswap.plugins.melee.prep import atlas as prep_atlas
    from rigswap.plugins.melee.prep.join import _material_settings
    bpy = _empty_scene()
    pixels = _flat(4, 4)
    pixels[0, 0, 3] = 0

    def image(name, rgba, space='sRGB'):
        made = bpy.data.images.new(name, 4, 4, alpha=True)
        made.colorspace_settings.name = space          # first: a generated image set to another space starts over
        made.pixels.foreach_set((rgba[::-1].astype(np.float32) / 255).ravel())
        return made

    def mia(name):
        m = _material(name, pixels)
        tree = m.node_tree
        bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
        colour = next(n for n in tree.nodes if n.type == 'TEX_IMAGE')
        again = tree.nodes.new('ShaderNodeTexImage')
        again.image = image(name + ' again', pixels, 'Non-Color')
        tree.links.new(again.outputs['Alpha'], bsdf.inputs['Alpha'])
        tree.links.new(tree.nodes.new('ShaderNodeNormalMap').outputs['Normal'], bsdf.inputs['Normal'])
        tree.links.new(colour.outputs['Color'], bsdf.inputs['Specular Tint'])
        return m, bsdf, colour
    skin, bsdf, colour = mia('skin')
    opened = skin.copy()                                # sidedness's two-sided copy of it
    opened.use_backface_culling = False
    cut, cut_bsdf, _ = mia('cut')                       # its Alpha another picture's: stays
    other = next(n for n in cut.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image.name == 'cut again')
    other.image = image('mask', _flat(4, 4, (0, 0, 0, 0)))
    mixed = _material('mixed', pixels)                  # not a Principled BSDF's output: left
    tree = mixed.node_tree
    mix = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(mix.outputs['Shader'], next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])
    tree.nodes.new('ShaderNodeNormalMap')
    body = _grid('body', 2, [skin, opened, cut, mixed])

    record = prep_textures.drawn_only([body])
    assert [r['material'] for r in record] == ['cut', 'skin', 'skin.001']
    assert record[1]['alpha_from_base_color'] == 'skin again' and 'alpha_from_base_color' not in record[0]
    assert sorted(n.type for n in skin.node_tree.nodes) == ['BSDF_PRINCIPLED', 'OUTPUT_MATERIAL', 'TEX_IMAGE']
    assert bsdf.inputs['Alpha'].links[0].from_node == colour and bsdf.inputs['Alpha'].links[0].from_socket.name == 'Alpha'
    assert not bsdf.inputs['Normal'].is_linked and not bsdf.inputs['Specular Tint'].is_linked
    assert prep_atlas._texture(skin) == colour
    assert (_material_settings(skin, ignore={'use_backface_culling'})
            == _material_settings(opened, ignore={'use_backface_culling'})) != opened
    assert sorted(n.type for n in cut.node_tree.nodes) == ['BSDF_PRINCIPLED', 'OUTPUT_MATERIAL', 'TEX_IMAGE',
                                                           'TEX_IMAGE']
    assert cut_bsdf.inputs['Alpha'].links[0].from_node == other
    assert 'NORMAL_MAP' in {n.type for n in mixed.node_tree.nodes}
