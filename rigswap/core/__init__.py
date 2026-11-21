"""The new middle: put the fighter's skeleton inside the model, let the model's own structure say which part of the mesh
follows which fighter bone, then move the whole mesh into the fighter's rest with one skinning deformation.

Design: docs/DESIGN.md (agreed 2026-10-25). Plumbing contract: rigswap-judge/redesign/contracts.md (export needs only
build.blend with the fighter's armature at rest and meshes skinned to its JOBJ names, plus target.json, then prep).
"""
