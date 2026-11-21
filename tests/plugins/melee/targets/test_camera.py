"""targets/camera.py: the game camera."""
from rigswap.plugins.melee.targets import camera as targets_camera


def test_camera_is_the_game_camera_in_model_space():
    c = targets_camera.camera(scale=2.0, extent=10.0)
    assert c['distance'] == 42.0                       # never nearer than 42
    assert c['eye'] == [-21.0, 4.0, 0.0] and c['target'] == [0.0, 4.0, 0.0]
    assert targets_camera.camera(scale=1.0, extent=20.0)['distance'] == 60.0
    assert targets_camera.camera(1.0, 1.0, facing=-1)['eye'][0] > 0
