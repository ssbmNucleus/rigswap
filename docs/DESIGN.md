# The new middle: what has to be true for a swap to work (draft, 2026-10-25)

## The mechanism in one sentence
Put the fighter's skeleton **inside the model**, in the model's own pose and proportions, let the **model's own
structure** decide which part of the mesh follows which fighter bone, then move the whole mesh into the fighter's rest
pose with **one continuous skinning deformation**. Export stays as it is.

## Why that works emergently (and where it can't fail)
A model's author already solved "how does this mesh deform": a rigged rip carries its skin weights; a segmented rip
(N64, Dreamcast, OBJ statues in parts) was built to animate as rigid pieces. If each source bone (or rigid piece) is
handed to the fighter bone that plays its role, the fighter's animations deform the mesh the way its author intended.
Tears can only come from a broken correspondence: two neighbouring source bones handed to fighter bones that aren't
neighbours, or one source bone split unevenly. That is a property we can check directly in code. There are no bands,
no per-part scaling and no seam rules: one map moves everything, so no seam opens unless the map is wrong.

## The pieces
1. **Upright and front.** Rigged: the rig gives up (root to head) and the mirror axis. The front lies perpendicular
   to the mirror axis. Unrigged: the mesh's symmetry plane gives the mirror axis. The front's *sign* is semantic: one
   AI choice between the two fronts the facts allow. Scale to the fighter's height.
2. **The model's own skeleton.** Rigged: its armature at rest (from the bind matrices) and its skin, used as they
   are. Segmented: rigid pieces and where they touch form the skeleton (a piece is a segment, a contact is a joint).
   One-piece unrigged meshes are out of scope at first.
3. **Labels.** Code finds the chains, mirror pairs (by tree shape and symmetry) and the root. Names help when they
   exist ("L_Thigh"). The AI does one labelling pass on a colour-coded picture: each chain is spine, head, arm, leg,
   tail, wing, hair or other, plus the body plan (biped, quadruped, ball). That's about 2–3 asks per import, all of
   them semantics.
4. **Correspondence.** Each fighter part goes to a model chain segment, and each fighter joint gets a point inside the
   model: the matching model joint, or the proportional point along the chain when the counts differ. Model parts
   with no fighter counterpart (a tail on Mario) ride the bone they hang from. Fighter joints with no model part
   (fingers) sit with their parent. This step owns the one correctness condition: neighbours stay neighbours.
5. **Skin.** Rigged: the model's own weights, renamed through the correspondence (one weight source, never mixed per
   part). Segmented: each piece rides its bone rigidly, as its author built it. At most 6 per vertex, normalized.
6. **Re-pose.** Each fighter bone gets one transform from where it sits inside the model to the fighter's rest
   (turning it, and stretching it along its length to the fighter's bone length). The mesh follows by linear blend
   skinning with step 5's weights. This one step un-poses a posed download and re-proportions it, continuously.
7. **Hand-off.** Write `build.blend` (the fighter's armature at rest, meshes skinned to JOBJ names) and
   `target.json`, then run prep. That's the whole export contract (`contracts.md` §2).

Later and optional, fed through the existing readers once the core works: hanging chains as physics accessories,
eyes (blink), held items.

## Fixed by the game, not chosen
The fighter's skeleton (names, parents, rest positions) stays vanilla: hurtboxes and hitboxes hang off its bones, so
moving joints would change gameplay. So proportions must come from re-posing the mesh, not from editing the
skeleton.

## Open questions to settle by trying, not by arguing
- How far to stretch along a bone (all the way to the fighter's length, or partly, keeping the model's look)?
  Decide by looking at a few in game.
- One-piece unrigged meshes: a geometric skeleton, or out of scope?
- Posed glTF rigs: is the rest from the bind matrices always right?

## How we work on it
- Build it as a new package beside the old middle, using the plumbing as is.
- First loop: rigged bipeds onto biped fighters, end to end, about 10 downloads, run and shot in game in minutes.
- Then quadrupeds and balls, then segmented unrigged, then the optional parts.
- Look for trends, dig into the failures to find the piece that's wrong, fix the piece.
- Keep it small; there is no score to beat.
