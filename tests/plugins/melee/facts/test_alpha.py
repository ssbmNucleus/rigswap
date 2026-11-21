"""facts/alpha.py: the materials' alpha rules, and the import's gate."""
import numpy as np
import pytest

from rigswap.plugins.melee.facts import alpha as facts_alpha


def test_alpha_facts_distinguish_glows_soft_overlays_and_cutout_rims():
    cutout = np.array([0, 255] * 49 + [128, 128], dtype=np.uint8)
    soft = np.array([0] * 90 + [128] * 10, dtype=np.uint8)
    assert facts_alpha.choose({'kind': 'glow'}, cutout)['mode'] == 'Blend'
    assert facts_alpha.choose({'alpha_mode': 'BLEND'}, cutout)['mode'] == 'Test'
    assert facts_alpha.choose({'alpha_mode': 'BLEND'}, soft)['mode'] == 'Blend'
    assert facts_alpha.choose({'alpha_mode': 'MASK'}, soft)['mode'] == 'Test'
    assert facts_alpha.choose({'alpha_mode': 'BLEND'}, np.array([8, 248]))['mode'] == 'Test'


def test_an_alpha_never_clear_nor_opaque_where_it_draws_is_translucency_and_blends():
    """Pikmin 2's glass helmet: black, alpha 53 to 158 over every texel, its source OPAQUE. Test cut it to the few
    texels above 0x80; its alpha is a level of translucency, and it blends. An eyelash's soft edges leave its clear and
    solid texels: still a cutout. A MASK material is a cutout by its source's word."""
    glass = np.array([53, 64, 70, 100, 158] * 20, dtype=np.uint8)
    lashes = np.array([0] * 20 + [255] * 10 + [128] * 70, dtype=np.uint8)
    assert facts_alpha.choose({'alpha_mode': 'OPAQUE'}, glass)['mode'] == 'Blend'
    assert facts_alpha.choose({}, glass)['mode'] == 'Blend'                    # no mode said is OPAQUE
    assert facts_alpha.choose({'alpha_mode': 'OPAQUE'}, lashes)['mode'] == 'Test'
    assert facts_alpha.choose({'alpha_mode': 'MASK'}, glass)['mode'] == 'Test'
    assert facts_alpha.choose({'alpha_mode': 'BLEND'}, glass)['mode'] == 'Blend'
    assert facts_alpha.choose({'alpha_mode': 'OPAQUE'}, np.array([], np.uint8))['mode'] == 'Test'


def _image(alpha):
    alpha = np.asarray(alpha, np.uint8)
    rgba = np.full(alpha.shape + (4,), 128, np.uint8)
    rgba[..., 3] = alpha
    return rgba


def test_the_format_the_import_writes():
    opaque, cutout = np.full((256, 256), 255, np.uint8), np.full((256, 256), 255, np.uint8)
    cutout[:8] = 0
    soft = cutout.copy()
    soft[0, 0] = 100                        # on the import's sample, every 4th texel of each side here
    unsampled = cutout.copy()
    unsampled[1, 1] = 100                   # between its samples
    assert facts_alpha.written_format('CMPR', opaque) == 'CMP'
    assert facts_alpha.written_format(None, cutout) == 'CMP'
    assert facts_alpha.written_format('CMPR', soft) == 'RGB5A3'
    assert facts_alpha.written_format('CMPR', unsampled) == 'CMP'
    assert facts_alpha.written_format('CI8', soft) == 'CI8:RGB565'
    assert facts_alpha.written_format('CI8:RGB5A3', soft) == 'CI8:RGB5A3'
    assert facts_alpha.written_format('RGB565', soft) == 'RGB565'


def test_rgb5a3_keeps_three_bits_of_alpha():
    # Cammy on Mario (runs/pilot4): Tex_0010 and Tex_0012 have no alpha below 242; RGB5A3 writes them opaque.
    alpha = facts_alpha.encoded_alpha(_image([[242, 248, 249, 254, 255, 224, 223, 0]]), 'RGB5A3')
    assert alpha.tolist() == [[255, 255, 255, 255, 255, 255, 218, 0]]


def test_cmpr_keeps_alpha_below_8_where_its_block_has_a_clear_colour():
    a = np.full((8, 8), 255, np.uint8)
    a[3, 3] = 0                             # the last texel of the top left block: the encoder never looks at it
    a[0, 4] = 5                             # the first of the top right: below 8, clear
    a[4, 0], a[4, 1] = 12, 7                # bottom left: 12 gives the block its clear colour; 12 stays, 7 is clear
    a[4, 4] = 200                           # bottom right: partial alpha CMPR drops
    out = facts_alpha.encoded_alpha(_image(a), 'CMP')
    clear = {tuple(p) for p in np.argwhere(out < 255)}
    assert clear == {(0, 4), (4, 1)} and set(np.unique(out)) == {0, 255}


@pytest.mark.parametrize('fmt,transparent', [('RGB565', False), ('CI8:RGB565', False), ('RGBA8', True),
                                             ('CI8:RGB5A3', None)])
def test_other_formats(fmt, transparent):
    gate = facts_alpha.gate([_image([[255, 255], [255, 254]])], fmt)
    assert gate['transparent'] is transparent
    assert gate['below_255'] == .25 and gate['format'] == fmt


def test_the_gate_reads_the_texels_the_import_writes_not_the_png():
    rgba = _image(np.full((256, 256), 255))
    rgba[0, :4, 3] = 242                    # cammy's Tex_0012: a few texels just under opaque
    gate = facts_alpha.gate([rgba], 'RGB5A3')
    assert gate == {'format': 'RGB5A3', 'below_255': 4 / 65536, 'transparent_texels': 0., 'transparent': False}
    rgba[0, 0, 3] = 0
    gate = facts_alpha.gate([rgba], 'RGB5A3')
    assert gate['transparent'] and gate['transparent_texels'] == 1 / 65536
    assert facts_alpha.gate([], None) == {'format': None, 'below_255': 0., 'transparent_texels': 0.,
                                          'transparent': False}
