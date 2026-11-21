"""Pieces: the answers round 2 can give to what a piece belongs to, and what a twin of a piece can be, defined here once.
One entry per answer, with what the AI is asked and what the description makes of it;
a new kind of piece is a new entry. The vocabulary's BELONGS is these names, round 2 asks them (describe.pieces.
round_2) and drop carries them out (describe.pieces.dropping, copies); which of them are layers or held items is
library.moves_as's.

BELONGS, one entry per ``Piece.belongs``:
    names_a_piece   the field naming the other piece the answer is about (layer_of, copy_of, alternate_of), or None
    dropped         whether drop removes a piece with this answer
    piece_required  whether the answer needs that piece named (a rendering layer needs the piece under it; a copy can
                    be dropped without knowing which piece it copies)
    asked           False when round 2 does not ask it of each piece (the twins ask gives alternate_of)
    question        what the answer means, as the question says it
TWIN_KINDS, one entry per kind of twin the twins ask can answer for a piece other than the one seen in play:
    belongs         the belongs it takes, naming the piece seen in play
    question        what it is, as the question says it"""
from __future__ import annotations

BELONGS = {
    # part of the model that stays and is built
    'character': {'names_a_piece': None, 'dropped': False, 'question': 'part of the character.'},
    # an item the character holds: stays, tied to its hand (build.place_held)
    'held': {'names_a_piece': None, 'dropped': False,
             'question': 'a thing the character holds, such as the sword lying beside it.'},
    # a glow, a shadow, a trail or a flash, not the character: dropped
    'effect': {'names_a_piece': None, 'dropped': True,
               'question': 'a visual effect, not the character: a glow, a shadow under it, a trail, a flash.'},
    # a display base the character stands on, not the character: dropped
    'stand': {'names_a_piece': None, 'dropped': True,
              'question': 'a display stand or base the character stands on.'},
    # a level-of-detail copy of a piece that stays: dropped
    'copy': {'names_a_piece': 'copy_of', 'dropped': True, 'piece_required': False,
             'piece_question': 'when belongs is copy, the number of the piece it is a copy of, when you can tell; '
                               'otherwise null.',
             'question': 'a lower-detail copy of part of the character (a level-of-detail mesh): the same part drawn '
                         'again in the same place, with fewer triangles and a coarser shape. It and the piece it '
                         'copies each lie mostly on the other; a small piece lying on part of a larger one is not '
                         'a copy of it. Give the piece it copies in copy_of when you can tell. Copies are removed; '
                         'the full-detail piece stays.'},
    # a second surface over another piece, following it: kept as that piece's layer
    'layer_of': {'names_a_piece': 'layer_of', 'dropped': False, 'piece_required': True,
                 'piece_question': 'when belongs is layer_of, the number of the piece it is a layer of; '
                                   'otherwise null.',
                 'question': "a rendering layer of another piece: a second surface with that piece's shape and "
                             'triangles, drawn over it, such as an edge-line (outline) pass or a shading pass. '
                             'An edge-line pass over the body is layer_of that body piece. Give that piece\'s '
                             'number in layer_of.'},
    # the same surface as the piece seen in play, in another form: dropped. Not in the list asked of each piece: a piece belongs alternate_of by the twins ask's answer (TWIN_KINDS).
    'alternate_of': {'names_a_piece': 'alternate_of', 'dropped': True, 'piece_required': True, 'asked': False,
                     'question': 'an alternate form of another piece, the one seen in play: the same surface with '
                                 'another texture, shown at other moments.'},
}


# What a piece of a twin set other than the one seen in play is (the twins ask's answer, twins_question), as the question
# says it, and what the description makes of it (drop_plan): the belongs it takes, naming the piece seen in play. Each
# is dropped: a twin lies on that piece triangle for triangle, so as a layer it could only fight it for the same
# pixels (Blender's outline shells are not inverted hulls: the same winding, places and image as the body).
TWIN_KINDS = {
    'shell': {'belongs': 'copy', 'question': 'a shell or outline of the surface: the surface drawn again for an edge '
                                             'line or shading, often in one flat or smeared colour'},
    'alternate': {'belongs': 'alternate_of', 'question': 'an alternate form of it: the surface with another texture, '
                                                          'shown at other moments (another costume, a transformation)'},
    'copy': {'belongs': 'copy', 'question': 'the surface drawn again, looking the same'},
}
