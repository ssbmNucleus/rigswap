import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import random

from PIL import Image
import pytest

from rigswap import ai, pictures
from rigswap.library import asks




def test_shared_picture_is_safe_across_routes_redraws_and_concurrent_asks(tmp_path):
    original = tmp_path / 'view.png'
    Image.new('RGB', (2000, 1500), 'red').save(original)
    with ThreadPoolExecutor(8) as pool:
        copies = list(pool.map(lambda _: pictures.save(original, 'describe round 1'), range(16)))
    assert len(set(copies)) == 1
    old = Path(copies[0]).read_bytes()
    close = pictures.save(original, 'build pick')
    with Image.open(close) as im:
        assert im.size == (1024, 768)
    assert close != copies[0] and Path(copies[0]).read_bytes() == old
    Image.new('RGB', (2000, 1500), 'blue').save(original)
    assert pictures.save(original, 'describe round 1') != copies[0]
    assert Path(copies[0]).read_bytes() == old


def test_route_settings_apply_to_closeups_png_and_quality(tmp_path):
    original = tmp_path / 'view.png'
    Image.frombytes('RGB', (1800, 900), random.Random(0).randbytes(1800 * 900 * 3)).save(original)
    whole, close = [pictures.save(original, 'describe round 3', i) for i in (0, 3)]
    for path, size in ((whole, (1024, 512)), (close, (1280, 640))):   # the library's round-3 edges
        with Image.open(path) as im:
            assert im.format == 'JPEG' and im.size == size
    how = replace(asks.DEFAULT, picture=asks.Picture(400, 'JPEG', 92))
    with Image.open(pictures.save(original, how)) as im:
        assert im.size == (400, 200) and im.format == 'JPEG'
    how = replace(how, picture=asks.Picture(400, 'PNG'))
    with Image.open(pictures.save(original, how)) as im:
        assert im.size == (400, 200) and im.format == 'PNG'




@pytest.mark.parametrize('route', asks.ASKS)
def test_tiny_png_wins_over_larger_encoding_and_is_attached_as_png(tmp_path, route):
    original = tmp_path / 'tiny.png'
    Image.new('RGBA', (20, 10), (0, 0, 0, 0)).save(original)
    source = original.read_bytes()
    sent = Path(pictures.save(original, route, folder=tmp_path / 'copies'))
    assert sent != original and sent.suffix == '.png'
    assert sent.read_bytes() == source
    assert original.read_bytes() == source
    assert pictures.record_path(sent) == str(original)
    assert pictures.save(original, route, folder=tmp_path / 'copies') == str(sent)
    part = ai.request_body('q', [sent], {}, route=route)['messages'][1]['content'][1]
    assert part['image_url']['url'] == 'data:image/png;base64,' + base64.b64encode(source).decode()


def test_jpeg_wins_without_upscaling_a_picture_below_the_cap(tmp_path):
    original = tmp_path / 'small.png'
    Image.frombytes('RGB', (80, 40), random.Random(0).randbytes(80 * 40 * 3)).save(original)
    sent = Path(pictures.save(original, 'describe round 2'))
    assert sent.stat().st_size < original.stat().st_size
    with Image.open(sent) as image:
        assert image.format == 'JPEG' and image.size == (80, 40)


def test_old_oversized_cached_jpeg_is_not_reused(tmp_path):
    import hashlib

    original = tmp_path / 'view.png'
    Image.new('RGB', (20, 10), 'red').save(original)
    source = original.read_bytes()
    identity = hashlib.sha256(source).hexdigest()[:20]
    old = tmp_path / f'sent-1280-jpeg-85-{identity}' / 'view.sent.jpg'
    old.parent.mkdir()
    Image.open(original).save(old, format='JPEG', quality=85, optimize=True)
    assert old.stat().st_size > len(source)
    sent = Path(pictures.save(original, 'describe round 1'))
    assert sent != old and sent.read_bytes() == source
