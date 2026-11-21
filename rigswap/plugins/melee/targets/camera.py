"""The game camera for a fighter (camera): ``prepare`` writes it beside the fighter's .glb as <Name>.camera.json, and
the result's sheet looks at the build through it (plugins/__init__.py says what the sheet reads of it)."""


# The game camera, from the old importer (randall model_import/studio/look/game.py, `game_camera`, measured against
# in-game captures): a perspective camera with a 35 degree vertical field of view, on the stage plane in front of the
# fighter, aimed 8 game units above the fighter's origin, at max(42, 3 x the fighter's largest extent) units; the
# fighter faces right on screen. The game draws a fighter's model `model_scale` times larger, so in the model's own
# space (the .glb's: y up, the fighter facing +z) the camera sits on the -x side at distance / scale, height 8 / scale,
# looking along +x. The game renders 640x480 and shows it at 4:3 with a projection aspect of 1.2173333, so the scene
# is drawn 1.0953 times wider than square pixels would draw it. The camera follows the fighter's root joint.
CAMERA_FOV_Y = 35.0
CAMERA_HEIGHT = 8.0
CAMERA_DISTANCE_MIN = 42.0
CAMERA_DISTANCE_PER_EXTENT = 3.0
CAMERA_RESOLUTION = (640, 480)
CAMERA_ASPECT = 1.2173333


def camera(scale, extent, facing=1):
    """The game camera for a fighter of in-game ``scale`` whose model spans ``extent`` units at most. ``facing`` 1:
    facing right on screen, -1 left. Its projection and numbers are what the sheet reads (plugins/__init__.py); the
    rest records how it was made. (Fighters prepared before 2026-10-10 also have ``'kind': 'melee'``, read by
    nothing now.)"""
    distance = round(max(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_PER_EXTENT * float(extent)), 3)
    side = 1. if facing >= 0 else -1.
    lift = CAMERA_HEIGHT / scale
    return {'projection': 'perspective', 'fov_y': CAMERA_FOV_Y,
            'eye': [-side * distance / scale, lift, 0.], 'target': [0., lift, 0.], 'up': '+y', 'front': '+z',
            'resolution': list(CAMERA_RESOLUTION), 'stretch_x': round((4 / 3) / CAMERA_ASPECT, 5),
            'distance': distance, 'model_scale': round(scale, 4), 'facing': int(side),
            'follows': 'the root joint', 'space': "the target .glb's (y up, the fighter facing +z)"}
