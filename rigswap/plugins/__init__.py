"""The Melee plugin and its target-description helpers."""
from importlib import import_module
from pathlib import Path

from ..vocabulary import Description


class Failed(Exception):
    """A step that can't go on, with its reason in plain words and numbers."""


def description_path(glb):
    """Where a target's description is: ``<file name>.description.json`` beside its .glb. ``rigswap describe-target``
    writes it."""
    return Path(glb).with_suffix('.description.json')


def read_description(glb):
    """The description of the target in ``glb``, with its bone graph's path made whole: the file keeps it relative
    to itself. Failed when the target has not been described."""
    path = description_path(glb)
    if not path.is_file():
        raise Failed(f'{Path(glb).stem} must be described first: no {path.name} in {path.parent} '
                     f'(rigswap describe-target writes it)')
    desc = Description.load(path)
    if desc.bones.get('graph'):
        desc.bones = dict(desc.bones, graph=str((path.parent / desc.bones['graph']).resolve()))
    return desc


def load(name):
    if name != 'melee':
        raise ValueError(f'unknown plugin: {name}')
    return import_module('rigswap.plugins.melee')


def names():
    """The supported game."""
    return ['melee']
