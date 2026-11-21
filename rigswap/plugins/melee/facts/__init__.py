"""The Melee facts: what the game has, as tables, one row for each fighter (or part, or material), each with where it
was read or measured, and the small lookups that read them.

    fighters.py   every fighter with a costume: its file code, root joint, influence cap, in-game scale, the triangles
                  its costume draws, the bytes it keeps, what the game preloads with it (FIGHTERS); fighter, in_scope
    parts.py      the vanilla parts a costume keeps (PARTS: weapons, stowed forms, replacement bodies), their
                  triangles (PART_TRIANGLES), and what each weapon a fighter fights with is and its other forms
                  (WEAPONS); kept_places, hidden_forms, not_kept
    chains.py     the chains of each fighter's joints, the game's physics strands among them, and the roles a joint's
                  name would give wrongly (CHAINS, OWN_ROLES)
    alpha.py      the materials' alpha: the mode each one is imported with (RULES, choose)
"""
