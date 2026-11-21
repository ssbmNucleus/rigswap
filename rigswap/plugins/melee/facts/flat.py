"""A flat fighter: what the game does with Game & Watch's file. He is an ordinary skinned biped of 53 joints that the game
draws differently, so nothing about his rig needs a path of its own; these are the facts a build has to meet
(prep/flat.py makes it meet them, export/flat.py writes what the file has to say). Fighter.flat says who is flat. From
the decomp (the doldecomp melee project, read 2026-11-01): ftGw_Init.c, fighter.c, ftmaterial.c, ftdrawcommon.c.

Flat. ftGw_Init_OnLoad puts his width, 0.01, in x34_scale.z, and Fighter_UpdateModelScale writes it to the root joint's
scale X, so after the joints move model X is a hundredth of itself and what shows is an outline in YZ, seen from -X
when he faces right and from +X when he faces left. A joint that turns about up brings its own X into that plane: his
waist stands nearly square to the camera at rest and swings from none of its width in the plane to all of it from one
move to the next (Run none, Wait1 .88, nair all of it); his head stands in profile but for three moves (usmash, dsmash,
throw forward). So the broad side of a model lies in YZ, and a body has depth: what a sheet shows goes edge-on.

Fill. ftMaterial_800BFB4C overwrites every material's diffuse with the slot's fill: black vanilla, white in the slots
the app makes (its core/gaw_colors.py). A black texture draws black under any fill. Only a REPLACE texture on a CONSTANT
material ignores the fill, so that's what shows its own image.

Outline. The hull (dobjs.py: HULL) is drawn first with no depth written, then the body, then the hull again with depth
alone (ftDrawCommon_80080E18), so it shows only outside the body. Its colour is the material's blended with his outline
colour by that colour's alpha (ftMaterial_800BF6BC): half white over black, grey 128. A hull with no texture would go
white in a white slot; one with the body's black texture is that grey in every slot.

Head. His head is model 2, its options at places 17, 16, 42 and 108, one of which the moves swap in. The last is on
joint 20, and an envelope PObj on a DObj under a non-root joint is transformed twice, so a build's head can't stand
there: it stays in the body, in model 0 on the root, which every move shows, and all four stay blank.
"""

# How far his outline stands off his body, in his units: where the vertices of each vanilla hull DObj lie from the body
# DObj they outline. It is not one number: the medians run from .21 (arms) through .25 (torso) and .29 (head, legs) to
# .36 (feet), each hull drawn by hand, with no share of the piece's size in it. What the eye reads is one line, so this
# is the middle of them.
LINE = .29

RENDER_CONSTANT = 1         # a material's render mode: drawn in its diffuse colour, not lit
RENDER_LIGHT = 0b1111       # the four bits saying how a material is coloured: constant, vertex, diffuse, specular
TOBJ_COLOUR = 15 << 16      # a TObj's flags: how it colours what it's on
COLOUR_REPLACE = 5 << 16    # with its own image alone
