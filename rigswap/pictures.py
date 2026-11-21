"""Save the pictures an ask sends, before recording, replay matching or AI caching.

The drawings stay as the run's record. Sent copies live in a settings-named folder beside them, named
<name>.sent.jpg (or the original format when its bytes are smaller). The folder keeps the same drawing safe to ask with at
different sizes. Every route's encoding and edges come from library.asks, never from a drawing's layout.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
import threading

from PIL import Image

from .library import asks

_lock = threading.Lock()
_records = {}       # prepared paths made in this run, including the eyes' further-shrunk copies


def record_path(path):
    """The drawing behind a prepared copy, for the watch page; an existing sent file is its own record."""
    return _records.get(str(path), str(path))


def save(path, route, index=0, *, folder=None, max_edge=None):
    """Return a sent copy without changing ``path``. ``max_edge`` may tighten an eyes request's byte budget.

    The encoded copy never upscales, and the original bytes win whenever encoding would make them larger.
    Already prepared pictures pass through unchanged, so watch and ai can both guard their own boundary. When
    ``folder`` is supplied (a picture outside the run), neither the source nor its directory is written to.
    """
    path = Path(path)
    how = asks.entry(route).picture
    if path.stem.endswith('.sent') and max_edge is None:
        return str(path)
    edges = how.max_edge
    edge = edges[min(index, len(edges) - 1)] if isinstance(edges, tuple) else edges
    if max_edge is not None:
        edge = min(edge, max_edge)
    suffix = 'jpg' if how.format == 'JPEG' else 'png'
    record = record_path(path)
    source = Path(record).read_bytes()
    # The source digest also prevents stale copies after a redraw, and gives external files with equal names
    # separate destinations. Published copies are immutable: concurrent asks/replay hashes see identical bytes.
    identity = hashlib.sha256(source).hexdigest()[:20]
    dest = Path(folder) if folder is not None else path.parent
    # A new policy namespace prevents reuse of older encoded copies that could exceed the original's bytes.
    dest = dest / f'sent-smallest-{edge}-{how.format.lower()}-{how.quality}-{identity}' / f'{path.stem.removesuffix(".sent")}.sent.{suffix}'
    original_dest = dest.with_suffix(Path(record).suffix.lower())
    with _lock:
        if original_dest.is_file():
            dest = original_dest
        if not dest.is_file():
            with Image.open(BytesIO(source)) as original:
                pic = original.convert('RGBA')
                # Alpha must be composited, not dropped: transparent texels can have black RGB underneath.
                background = Image.new('RGBA', pic.size, 'white')
                background.alpha_composite(pic)
                pic = background.convert('RGB')
                pic.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                encoded = BytesIO()
                pic.save(encoded, format=how.format, optimize=True,
                         **({'quality': how.quality} if suffix == 'jpg' else {}))
                payload = encoded.getvalue()
                if len(source) <= len(payload):
                    payload, dest = source, original_dest
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        stream.write(payload)
                    except BaseException:
                        stream.close()
                        temporary.unlink(missing_ok=True)
                        raise
                try:
                    os.replace(temporary, dest)
                finally:
                    temporary.unlink(missing_ok=True)
        _records[str(dest)] = record
    return str(dest)


def prepare(paths, route, *, folder=None):
    return [save(p, route, i, folder=folder) for i, p in enumerate(paths)]
