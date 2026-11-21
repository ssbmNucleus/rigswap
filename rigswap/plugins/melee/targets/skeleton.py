"""Read target bone roles and the starts of their body parts."""
from ....library.body_types import of, part_ends
from ....vocabulary import BODY_TYPES, CANT_TELL, CHAIN_KINDS, roles_by_number


def skeleton(graph, roles, body_type, joints=None):
    """A bone graph as a plain list: [{'name', 'parent' (a name or None), 'head', 'tail' (None when the graph has
    none), 'role', 'parts'}], in the graph's order. ``graph``: bones.json's dict or its 'bones' list (numbered bones,
    parents by number). ``roles``: {number, or number as text: role} (roles_by_number), from the description; a bone
    with no role gets "can't tell". A role may list several parts, in chain order, for one bone across them (a
    one-bone leg: [left_thigh, left_shin]); a role given as text is a list of one. 'parts' is the whole list and
    'role' its first part, which is where the bone starts. A bone answered "can't tell" whose part the skeleton itself
    shows (placed_parts, for ``body_type``) has that part as its role, and 'placed' says why; everything that reads
    the skeleton reads it as that part's bone.

    ``joints``: the description's joints (round 4's, for the parts a rig lacks). A bone of a part that carries no
    vertices (unweighted) does not start that part when round 4 placed the joint the part starts at: 'unweighted'
    says why, and part_starts passes it by, so the part starts at the first of its bones that carries vertices (a
    foot's bone that carries the shoe, under an empty bone up in the hip), else, its rig lacking it, at the joint
    (bind_parts makes its bone from the joints). It keeps its part, so what hangs from it (hair, a helper) still moves
    with that part and does not fall through to the part above."""
    bones = graph['bones'] if isinstance(graph, dict) else graph
    name_of = {b['n']: b['name'] for b in bones}
    roles = roles_by_number(roles)
    out = []
    for b in bones:
        role = roles.get(int(b['n']), CANT_TELL)
        parts = [str(p) for p in role] if isinstance(role, (list, tuple)) else [role]
        parts = parts or [CANT_TELL]
        out.append({'n': b['n'], 'name': b['name'], 'parent': name_of.get(b['parent']),
                    'head': [float(v) for v in b['head']],
                    'tail': [float(v) for v in b['tail']] if b.get('tail') is not None else None,
                    'role': parts[0], 'parts': parts})
    placed = placed_parts(out, body_type)
    for b in out:
        if b['name'] in placed:
            part = placed[b['name']]['part']
            b.update(role=part, parts=[part], placed=placed[b['name']]['why'])
    carried = {b['name']: b.get('vertices') for b in bones}
    for b, why in unweighted(out, body_type, joints, carried).items():
        next(x for x in out if x['name'] == b)['unweighted'] = why
    return out


def unweighted(bones, body_type, joints, carried):
    """{bone: why} for each bone of a part that carries no vertices (``carried``: {bone: how many vertices it moves},
    bones.json's 'vertices'; a bone not counted is left as it is) while ``joints`` has the joint that part starts at
    (the body type's part_ends): such a bone starts nothing; the first of the part's bones that carries vertices starts
    it, else the joint (skeleton)."""
    known, ends = set(BODY_TYPES[body_type]), part_ends(body_type)
    out = {}
    for b in bones:
        parts = [p for p in b['parts'] if p in known]
        if not parts or carried.get(b['name']) != 0:
            continue
        start = ends[parts[0]][0]
        if start in (joints or {}):
            out[b['name']] = (f"a bone of the {', '.join(parts).replace('_', ' ')} that carries no vertices, and round "
                              f"4 placed the {start.replace('_', ' ')} joint, which starts the "
                              f"{parts[0].replace('_', ' ')}: the first of its bones that carries vertices starts it, "
                              f"else the joint")
    return out


def placed_parts(bones, body_type):
    """{bone name: {'part', 'why'}} for each bone answered "can't tell" whose part the skeleton itself shows. The
    nearest bone above it with a part has part P (its last, for a bone across several parts); the nearest bone with a
    part down each of its branches all start the same part Q; and Q is the part after P in the body (its chains: the
    next part of P's chain, or the first part of a chain that hangs from P: the neck's head, the forearm's hand, the
    hips' thighs). Such a bone sits at the joint between P and Q, so it is Q's first bone. A branch with no bone with a
    part below it (a hair chain, a helper) says nothing. When the branches disagree, Q does not follow P, or there is no
    bone with a part above it or below it, the bone is not placed, and moves with the nearest bone above it that has a
    part (fit.part_of_bone). Only the answers count, never a bone placed here, so the bones' order changes nothing.
    With no bone with a part above it, the bone is the rig's root: when the bones with a part right below it all hang
    from one part H in the body, past the parts the rig answers no bone for (the spine and a leg whose only bone is its
    foot both hang from the hips), H being the body's first part and no bone of the rig answered H, it is H's bone
    when it was answered "can't tell", or "none" with a chain hanging from it (a chain hangs from the body): what
    hangs from it (a robe's skirt) moves with the hips, not with nothing. Answered "none" with no chain, the answer
    stands, as above.

    A bone answered "none" or "can't tell" with no bone of a part anywhere below it, but a chain (hair, a tail, fingers)
    hanging below it, is what that chain hangs from, and a chain hangs from the body: when the part P above it is
    followed by one part Q alone (the neck by the head, the forearm by the hand) and no bone of the rig is answered Q,
    it is Q's bone. A rig whose head bone was answered "none", with the face on hair bones below it, keeps its face on
    the head instead of dropping it to the neck or the chest. When the rig answers some other bone Q, or P is followed
    by several parts (the chest), it stays as answered. ``bones``: [{'name', 'parent' (a name or None), 'parts'}], as
    skeleton has them before it places any."""
    known = set(BODY_TYPES[body_type])
    kinds = {k for k in CHAIN_KINDS if k not in known}
    by_name = {b['name']: b for b in bones}
    children = {}
    for b in bones:
        if b['parent'] is not None:
            children.setdefault(b['parent'], []).append(b)
    follows = {p: chain[i - 1] if i else parent                 # {part: the part it hangs from}
               for parent, chain in of(body_type).chains for i, p in enumerate(chain)}

    def parts(b):
        return [p for p in b['parts'] if p in known]
    answered = {p for b in bones for p in parts(b)}

    def hangs_from(q):              # the part q hangs from, past the parts the rig answers no bone for
        up = follows.get(q)
        while up is not None and up not in answered and follows.get(up) is not None:
            up = follows[up]
        return up

    out = {}
    for b in bones:
        said = list(b['parts'])
        if said not in ([CANT_TELL], ['none']):
            continue
        above = by_name.get(b['parent'])
        while above is not None and not parts(above):
            above = by_name.get(above['parent'])
        below, chained, look = [], False, list(children.get(b['name'], []))
        while look:                                    # down each branch to its nearest bone with a part
            c = look.pop(0)
            if parts(c):
                below.append(c)
            else:
                chained |= any(k in kinds for k in c['parts'])
                look += children.get(c['name'], [])
        if above is None:
            # the rig's root: the bone the body's line and its legs hang from, with no part answered above it
            root = {hangs_from(parts(c)[0]) for c in below}
            if ((said == [CANT_TELL] or chained) and len(root) == 1 and (h := root.pop()) is not None
                    and h not in answered and all(c['parent'] == b['name'] for c in below)):
                out[b['name']] = {'part': h, 'why': (
                    f'answered "{said[0]}"; no bone with a part is above it, and the bones with a part right below it '
                    f'({", ".join(str(c["name"]) for c in below)}) all hang from the {h}, which no bone of the rig '
                    f'is: it is the {h}')}
            continue
        p = parts(above)[-1]
        if not below:
            after = [q for q, up in follows.items() if up == p]
            if not chained or len(after) != 1 or after[0] in answered:
                continue
            out[b['name']] = {'part': after[0], 'why': (
                f'answered "{said[0]}"; it hangs from {above["name"]} ({p}), a chain hangs below it and no bone of a '
                f'part, and no bone of the rig is the {after[0]}, the one part after the {p}: the chain hangs from '
                f'the {after[0]}, so it is a bone of the {after[0]}')}
            continue
        starts = {parts(c)[0] for c in below}
        if said != [CANT_TELL] or len(starts) != 1:
            continue
        q = starts.pop()
        if follows.get(q) != p:
            continue
        names = [str(c['name']) for c in below]
        listed = ', '.join(names[:3]) + (f' and {len(names) - 3} more' if len(names) > 3 else '')
        out[b['name']] = {'part': q, 'why': (
            f'answered "{CANT_TELL}"; it hangs from {above["name"]} ({p}) and the nearest bones with a part below it, '
            f'on every branch, are all the {q} ({listed}), the part after the {p}: it sits at the joint between them, '
            f'so it is the first bone of the {q}')}
    return out


def part_starts(bones, body_type):
    """{part: {'start': head of the part's first bone, 'bone': its name, 'bones': every bone with that role}} for the
    parts of ``body_type`` that have a bone. A part's first bone is the bone with that role that has no ancestor with
    it; when several do, the one nearest the root (then the first in the graph) starts it, and 'starts' lists them.
    A bone skeleton reads as unweighted starts nothing and is not counted."""
    by_name = {b['name']: b for b in bones}

    def depth(b):
        d = 0
        while b['parent'] is not None:
            b, d = by_name[b['parent']], d + 1
        return d

    def has_ancestor_with(b, role):
        while b['parent'] is not None:
            b = by_name[b['parent']]
            if b['role'] == role and not b.get('unweighted'):
                return True
        return False

    out = {}
    for part in BODY_TYPES[body_type]:
        mine = [b for b in bones if b['role'] == part and not b.get('unweighted')]
        if not mine:
            continue
        firsts = [b for b in mine if not has_ancestor_with(b, part)]
        first = min(firsts, key=lambda b: (depth(b), bones.index(b)))
        rec = {'start': first['head'], 'bone': first['name'], 'bones': [b['name'] for b in mine]}
        if len(firsts) > 1:
            rec['starts'] = [b['name'] for b in firsts]
        out[part] = rec
    return out
