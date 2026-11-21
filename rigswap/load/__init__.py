"""Step 1, load: open every model file in the download, as it came, standing in the idle it names (stance) with its
texture transforms in its UVs (texture). Nothing is judged, and nothing is dropped but what the file itself declares:
surfaces its idle collapses, and the coarser levels of detail its mesh names declare (open.finest_levels)
(docs/pipeline.html, step 1, "Open every model file in the download").

    open      unpack the download, let Blender import every model file in it, match textures (open.py)
    convert   a .dae becomes a .glb through an installed Blender 4.x, since Blender 5 has no Collada importer
              (convert.py; open uses it for every .dae)

``run(download, out)`` is the step: it writes ``<out>/models/<n>-<name>.blend`` per model file and ``<out>/models.json``
listing them with counts."""
from .open import run

__all__ = ['run']
