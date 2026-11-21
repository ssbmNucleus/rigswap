"""Making a target from a vanilla costume, once for each fighter (the plugin's ``prepare``, then ``describe_targets``):

    prepare.py   the DObjs the game draws in play, exported by HSDRawViewer and made a .glb with every joint, each
                 texture as the export's texture map names it (prepare, dae_to_glb); the game camera beside it
    camera.py    the game camera for a fighter (camera)
    bones.py     the description from the game's own bone tables: PlCo.dat's joint names to roles, the chains
                 (describe_from_tables, describe_targets)
    weapons.py   the weapons the fighter fights with, measured on its vanilla costume (write_weapons)
    frames.py    the fighter's own animations at the frames the export's gate poses a costume in (write, read)
"""
