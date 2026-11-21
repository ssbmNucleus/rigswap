"""Moves as -> constraint: one entry per answer to "how does this piece move", what it becomes inside weights.solve
(docs/pipeline.html, "Moves as -> constraint": "What each piece's answer becomes inside weights.solve. One row per
answer. A new way of moving is a new row."). And which pieces are layers, rendering layers and held items, defined here
once: the vocabulary's Piece asks here (is_layer, is_rendering_layer, is_held), and so does every step.

A rule (Rule):
    says                the constraint, as the design page's table says it
    rows                how weights.solve makes a piece's candidate rows (weights/constraints.py has one maker for
                        each)
    at_seam             what it is where pieces share a vertex (a seam, or the copies a file splits one vertex into):
                        'holds' (pinned: its candidates alone count, since it can't bend at its own edge), 'meets' (the
                        mean of the pieces that meet there), or 'follows' (it hangs from the body: where it shares a
                        vertex with a piece that does not follow, that piece decides, so no vertex of the body takes a
                        hanging chain's bone, which the game may simulate, through a seam). SEAM_ORDER is their order
    at_boundary         what it is to a body or layer vertex within BAND of its surface (weights.combine): 'meets' (the
                        two meet halfway at the seam), 'holds' (the vertex takes its weights there: it can't bend), or
                        None (it pulls on nothing)
    reads_piece_under   its candidates are those of the piece under it, read at the nearest point of that one's surface

The pieces each rule takes (rule_of): a layer, a held item, else a piece's own moves_as when it has a rule, else body.

    a layer             it moves as a layer (a sleeve over an arm, a skirt over the legs), or it is a rendering layer of
                        a piece it names: "Rendering layers (an edge-line pass over the body) are kept as layers of their
                        piece" (the design page, drop)
    a rendering layer   it belongs layer_of: a second surface with another piece's shape drawn over it (an edge-line or
                        shading pass). A plugin whose game can't draw one drops it (Melee's prep)
    a thin layer        a layer that is not a rendering layer: what a plugin's prep gives back faces (layers.json)
    a held item         it belongs held or moves as held: "Held items stay, tied to their one hand", "a held item is
                        pinned to its hand" (the design page). Every held item is sized and placed by place_held:
                        by fit's measured target/source hand length ratio, or, in a weapon slot of the target's,
                        to the length of the target's own weapon

And which chains a body piece follows: only a chain inside its part (INSIDE, body_follows), as weights.solve reads it
for a model's own chains and for the chains made from round-4 joints; any other hangs, and only a piece that hangs
follows it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    says: str
    rows: str
    at_seam: str
    at_boundary: str | None
    reads_piece_under: bool = False


# The chain kinds inside the part they grow from: a finger is in its hand (the body type's hand runs to the end of its
# longest finger), so a body piece on the hand moves with its fingers. Every other kind (hair, a tail, a skirt flap)
# hangs from the body, and only a piece that hangs follows it: a body piece's weights on one go to the part it hangs
# from, whether the chain is the model's own bones or the temporary bones bind.temp laid along a chain made from
# round-4 joints (a tail a model without tail bones has).
INSIDE = ('finger',)

# The piece kinds (a moves_as answer) that ride the part they are anchored to: a body piece bends with it and a rigid
# one is pinned to it, so what they cover is where that part is. A piece that hangs, is held or is a layer is not
# where its part is, and says nothing of it (fit.height: the pieces a download's head top is found among).
RIDE_THE_PART = ('body', 'rigid')


def body_follows(group):
    """Whether a body piece's weights follow the chain ``group``: only a chain inside its part (moves_as.INSIDE: a
    finger, inside its hand), the model's own bones and the temporary bones bind.temp laid along a chain made from
    round-4 joints alike. Any other chain (hair, a tail, a skirt flap) hangs from the body and moves only a piece that
    hangs (_hangs)."""
    return group['kind'] in INSIDE


# where pieces share a vertex, the first of these any of them is decides it (Rule.at_seam)
SEAM_ORDER = ('holds', 'meets', 'follows')

RULES = {
    'body': Rule("the model's own skin (its rig's, or its bind's) carried bone to bone onto the target's: it "
                 "follows its parts and the fingers inside them, never a chain that hangs from them; boundary shared "
                 "with neighbours", rows='the skin', at_seam='meets', at_boundary='meets'),
    'rigid': Rule("one weight vector: the skin under the place it is attached, else that part's bone; attached on "
                  "both sides of a joint, not rigid there: its own skin, as a body piece",
                  rows='the part bone', at_seam='holds', at_boundary='holds'),
    'layer': Rule('the same weights as the part under it', rows='the piece under it', at_seam='meets',
                  at_boundary=None, reads_piece_under=True),
    'held': Rule("pinned to that hand's bone", rows='the hand bone', at_seam='holds', at_boundary=None),
    'hangs': Rule('bone to bone along the chain, when the target has a matching chain; else frozen, pinned at its '
                  "root part's bone; where it shares a vertex with the body, it follows the body there",
                  rows='the chain', at_seam='follows', at_boundary=None),
}


def _field(piece, name):
    """A field of ``piece``: a vocabulary.Piece, or a piece as description.json holds it (a dict; a field it lacks is
    None)."""
    return piece.get(name) if isinstance(piece, dict) else getattr(piece, name)


def is_rendering_layer(piece):
    """Whether ``piece`` (a vocabulary.Piece, or its dict in description.json) is a rendering layer of another piece: it
    belongs layer_of."""
    return _field(piece, 'belongs') == 'layer_of'


def is_layer(piece):
    """Whether ``piece`` is solved as a layer: it moves as a layer, or it is a rendering layer of a piece it names."""
    return _field(piece, 'moves_as') == 'layer' or (is_rendering_layer(piece) and _field(piece, 'layer_of') is not None)


def is_thin_layer(piece):
    """Whether ``piece`` is a thin layer, which a plugin's prep gives back faces: a layer, not a rendering layer."""
    return is_layer(piece) and not is_rendering_layer(piece)


def is_held(piece):
    """Whether ``piece`` is a held item: it belongs held or moves as held."""
    return _field(piece, 'belongs') == 'held' or _field(piece, 'moves_as') == 'held'


def rule_of(piece):
    """The name of the rule (RULES) ``piece`` is solved by."""
    if is_layer(piece):
        return 'layer'
    if is_held(piece):
        return 'held'
    return _field(piece, 'moves_as') if _field(piece, 'moves_as') in RULES else 'body'
