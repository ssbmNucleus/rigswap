"""Melee material alpha facts, from untangle shared/modellab/alpha_modes.py:plan
and native_import.py:import_arguments (read-only reference, 2026-10-10); and the
import's gate, from HSDRawViewer (the ssbmNucleus HSDLib fork, read 2026-10-11).

The table applies to materials, never to costumes. GLB alphaMode and emissive
texture metadata carry the source's intent across the Blender/Collada boundary.
The rules read encoded alpha where the material's triangles draw it, not PNG alpha.

The gate. The import gives a material's texture the alpha operation of the mode it
was told, and a Test material its pixel engine compare, only when the texture as the
import wrote it is transparent (Converters/ModelImporter.cs ConfigureAlpha, behind
Extensions/TOBJExtensions.cs IsTransparent: the TObj's image decoded again, and some
texel's alpha not 255). Every other material gets no alpha operation, whatever mode it
was told. So whether a material comes out transparent is a matter of the texels the
import wrote, in the format it wrote them (written_format), not of the PNG it read:
RGB5A3 keeps an alpha of 224 to 254 as opaque, CMPR keeps only alpha below 8, in a
block that has room for it (encoded_alpha). An RGB5A3 texture whose lowest
alpha is 242 comes out opaque, rightly.
"""
from dataclasses import dataclass

import numpy as np


SOFT_ALPHA = (8, 248)  # exclusive byte bounds, alpha_modes.py:SOFT
SOFT_SHARE = .10      # alpha_modes.py:TRANSLUCENT


# A material's rule is the first whose conditions its meaning (plan: the source material's kind and glTF alpha mode)
# and its image's alpha meet; a condition left None asks nothing, so the last rule takes every material.
#   mode            what the import is told for the material (--alpha-modes)
#   kind            the material's kind must be this
#   alpha_mode      its source alpha mode must be this
#   soft_at_least   at least this share of its image's texels must be soft (SOFT_ALPHA)
@dataclass(frozen=True)
class Rule:
    name: str
    mode: str
    why: str
    kind: str | None = None
    alpha_mode: str | None = None
    soft_at_least: float | None = None

    def applies(self, meaning, soft):
        return ((self.kind is None or meaning.get('kind') == self.kind)
                and (self.alpha_mode is None or meaning.get('alpha_mode', '').upper() == self.alpha_mode)
                and (self.soft_at_least is None or soft >= self.soft_at_least))


# A material whose source says nothing of its alpha (OPAQUE: the .dae's or .mtl's material has no transparency, so
# Blender leaves the image's alpha unwired) still draws its image's alpha in the game: Test cuts a cutout's clear
# texels, which a rip that lost the wiring needs (hair cards, lashes). A cutout's alpha is clear somewhere and opaque
# somewhere. An alpha that is neither anywhere the material draws is a level of translucency, not a mask, and Test
# would decide it by the 0x80 threshold alone: Pikmin 2's glass helmet (an IA8 texture, black, alpha 53 to 158 over
# every texel) was cut down to a few specks of highlight, and the lining behind it showed as an opaque grey disc. Such
# a material is blended, as its alpha says. Every soft texel, SOFT_ALPHA, where it draws: an eyelash's soft edges
# leave its clear and solid texels, and it stays a cutout. A MASK material is a cutout by its source's word.
TRANSLUCENT_SHARE = 1.0

RULES = (
    Rule('glow', 'Blend', 'an emissive image with a soft halo', kind='glow'),
    Rule('soft overlay', 'Blend', 'source BLEND and at least 10% soft alpha texels', alpha_mode='BLEND',
         soft_at_least=SOFT_SHARE),
    Rule('translucent', 'Blend', 'source OPAQUE, and its alpha is soft wherever it draws: never clear, never opaque, a '
         'level of translucency and not a cutout', alpha_mode='OPAQUE', soft_at_least=TRANSLUCENT_SHARE),
    Rule('cutout or opaque', 'Test', 'keep depth writes; discard alpha below 0x80'),
)


def choose(meaning, alpha):
    """The material facts, evaluated on drawn, encoded alpha samples: the mode the import is told. Whether the import
    makes the material transparent at all is its gate's (gate). ``meaning``: {'alpha_mode': the source's glTF alpha
    mode (none: OPAQUE), 'kind': 'glow' for an emissive image}."""
    alpha = np.asarray(alpha)
    meaning = dict(meaning, alpha_mode=meaning.get('alpha_mode') or 'OPAQUE')
    soft = float(((alpha > SOFT_ALPHA[0]) & (alpha < SOFT_ALPHA[1])).mean()) if alpha.size else 0.
    rule = next(r for r in RULES if r.applies(meaning, soft))
    return {'mode': rule.mode, 'why': rule.why, 'soft_alpha': soft}


# ---- the import's gate --------------------------------------------------------------------------------------------

# The format the import writes a texture in, when --texture-formats names none for its material: CMP
# (ModelImportSettings.cs TextureFormat), turned to RGB5A3 when the image has partial alpha, some alpha above 8 and
# below 248 looking at every 64th texel of each side (ModelImporter.cs HasPartialAlpha), since CMP keeps one bit.
IMPORT_FORMAT = 'CMP'
PARTIAL_ALPHA = (8, 248)
PARTIAL_SAMPLES = 64
PALETTED = ('CI4', 'CI8', 'CI14X2')


def written_format(code, alpha):
    """The format the import writes a texture of alpha bytes ``alpha`` ((h, w)) in, FORMAT or FORMAT:PALETTE: the
    --texture-formats ``code`` (None: the import's own), CMPR read as CMP and a palette format's palette RGB565 when the
    code names none (ModelImportArgs.cs ReadTextureFormats), CMP turned to RGB5A3 for partial alpha."""
    fmt, _, palette = (code or IMPORT_FORMAT).upper().partition(':')
    fmt = 'CMP' if fmt == 'CMPR' else fmt
    if fmt in PALETTED:
        return f'{fmt}:{palette or "RGB565"}'
    h, w = alpha.shape
    sample = alpha[::max(1, h // PARTIAL_SAMPLES), ::max(1, w // PARTIAL_SAMPLES)]
    if fmt == 'CMP' and ((sample > PARTIAL_ALPHA[0]) & (sample < PARTIAL_ALPHA[1])).any():
        return 'RGB5A3'
    return fmt


def encoded_alpha(rgba, fmt):
    """The alpha of each texel of ``rgba`` ((h, w, 4) uint8) once the import has written it in ``fmt`` (as
    written_format names it) and HSDRaw decodes it again, which is what IsTransparent reads: (h, w) uint8. The
    encoders are HSDRaw's Tools/GXImageConverter.cs and Tools/Textures/CMPREncode.cs. None where this does not mirror
    the encoding: a palette with alpha (CI with an RGB5A3 or IA8 palette), whose colours the import's quantizer picks
    (ExoQuantSharp.cs), or a format prep never asks for; the file tells (export/checks.py: alpha_mode_check)."""
    base, _, palette = fmt.partition(':')
    a = rgba[..., 3]
    if base == 'RGB565' or (base in PALETTED and palette == 'RGB565'):
        return np.full(a.shape, 255, np.uint8)
    if base in ('RGBA8', 'IA8'):
        return a.copy()
    if base == 'RGB5A3':    # EncodeRGBA3: below 255, three bits of alpha (a / 32); DecodeRGBA3: a3 * 255 / 7
        return np.where(a == 255, 255, (a // 32).astype(np.int32) * 255 // 7).astype(np.uint8)
    if base == 'CMP':
        return _cmp_alpha(a)
    return None


def _cmp_alpha(a):
    """CMPREncode.cs: each 4x4 block of an 8x8 tile (texels past the image's edge zero) has a clear colour when one of
    its first fifteen texels has alpha below 16 (the loop stops short of the last), and then a texel with alpha below
    8 takes it; every other texel is opaque (GXImageConverter.cs fromCMP)."""
    h, w = a.shape
    H, W = -(-h // 8) * 8, -(-w // 8) * 8
    padded = np.zeros((H, W), np.uint8)
    padded[:h, :w] = a
    blocks = padded.reshape(H // 4, 4, W // 4, 4).swapaxes(1, 2).reshape(H // 4, W // 4, 16)
    clear = (blocks[..., :15] < 16).any(-1)[..., None] & (blocks < 8)
    out = np.where(clear, 0, 255).astype(np.uint8)
    return out.reshape(H // 4, W // 4, 4, 4).swapaxes(1, 2).reshape(H, W)[:h, :w]


# The alpha test a Test material gets (ModelImporter.cs ConfigureAlpha, its PEDesc): a texel is drawn when its alpha is
# greater than AlphaRef0, 0x80 (AlphaComp0 Greater, AlphaOp And, AlphaComp1 Always). Another mode draws a texel of any
# alpha above 0.
ALPHA_TEST_REF = 0x80


def draws_nothing(images, code, mode):
    """Whether a material drawing ``images`` ((h, w, 4) uint8 each), written in --texture-formats ``code`` and told
    ``mode``, draws no texel at all: every texel's alpha as the import writes it (encoded_alpha) fails the material's
    alpha test (Test: ALPHA_TEST_REF; any other mode: 0). False with no images; None where only the file can tell."""
    if not images:
        return False
    floor = ALPHA_TEST_REF if mode == 'Test' else 0
    for rgba in images:
        alpha = encoded_alpha(rgba, written_format(code, rgba[..., 3]))
        if alpha is None:
            return None
        if (alpha > floor).any():
            return False
    return True


def gate(images, code):
    """The import's gate for a material drawing ``images`` ((h, w, 4) uint8 each; the import writes its diffuse
    image), asked in --texture-formats ``code``: {format: as the import writes it, below_255: the share of the images'
    texels whose alpha is below 255, transparent_texels: the share still below 255 as the import wrote them,
    transparent: whether ConfigureAlpha gives the material an alpha operation}. The last two are None where only the
    file can tell (encoded_alpha); an untextured material gets no alpha operation."""
    if not images:
        return {'format': None, 'below_255': 0., 'transparent_texels': 0., 'transparent': False}
    formats, below, written = [], [], []
    for rgba in images:
        fmt = written_format(code, rgba[..., 3])
        formats.append(fmt)
        below.append(rgba[..., 3].ravel() < 255)
        alpha = encoded_alpha(rgba, fmt)
        written.append(None if alpha is None else alpha.ravel() < 255)
    share = None if any(w is None for w in written) else float(np.concatenate(written).mean())
    return {'format': formats[0] if len(set(formats)) == 1 else formats,
            'below_255': float(np.concatenate(below).mean()), 'transparent_texels': share,
            'transparent': None if share is None else share > 0}
