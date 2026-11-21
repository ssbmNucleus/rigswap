"""Body roles and consecutive chains in a target bone graph."""
from . import skeleton as correspond
from ....library.body_types import of
from ....vocabulary import BODY_TYPES, CANT_TELL, CHAIN_KINDS, Chain, role_parts, roles_by_number


NONE = 'none'               # a bone that is no part and in no chain: a root, a helper, a marker


def roles(body_type):
    """Every role a bone may be given for this body type: its parts, the chain kinds that aren't also one of its parts
    (a serpent's tail is a part), and none. "can't tell" comes on top in the schema."""
    parts = BODY_TYPES[body_type]
    return list(parts) + [k for k in CHAIN_KINDS if k not in parts] + [NONE]


def chain_kinds(body_type):
    parts = BODY_TYPES[body_type]
    return [k for k in CHAIN_KINDS if k not in parts]


def parts_of(role, body_type):
    """The parts of the body type a bone with this role covers: its one part, every part of a list in the order
    given, or none for a chain kind, "none" and "can't tell"."""
    parts = BODY_TYPES[body_type]
    if isinstance(role, list):
        return [p for p in role if p in parts]
    return [role] if role in parts else []


def chains(bones, roles_, body_type):
    """Chain bones joined into chains. Consecutive parent-child bones with the same chain kind are one chain; a chain
    ends where its last bone has no child of its kind, or more than one (each of those starts a chain of its own).
    A chain's root is the role of its nearest ancestor that covers a part of the body type, past any chain, none or
    "can't tell" bones on the way; "can't tell" when there is none. A "can't tell" bone whose part the skeleton itself
    shows (correspond.placed_parts, as every later step reads the skeleton) covers that part: hair on a head bone that
    couldn't be told, with only head bones below it, hangs from the head. When that ancestor covers several parts, the
    root is its whole list, as given: which of those parts the chain hangs from is not asked. ``roles_``: {bone
    number: role}, its keys numbers or numbers as text (a description's, read from JSON): roles_by_number reads both
    the same. Returns a list of Chain, by first bone number."""
    roles_ = roles_by_number(roles_)
    kinds = set(chain_kinds(body_type))
    by_n = {b['n']: b for b in bones}

    def answer(n):                                        # as a list, as correspond.skeleton reads it
        r = roles_.get(n, CANT_TELL)
        return list(r) if isinstance(r, list) else [r]
    placed = correspond.placed_parts([{'name': b['n'], 'parent': b['parent'], 'parts': answer(b['n'])}
                                      for b in bones], body_type)

    def kind(n):
        r = roles_.get(n)
        return r if isinstance(r, str) and r in kinds else None

    def same_kind_children(n, k):
        return [c for c in by_n[n]['children'] if kind(c) == k]

    out = []
    for b in sorted(bones, key=lambda b: b['n']):
        k = kind(b['n'])
        if k is None:
            continue
        p = b['parent']
        if p is not None and kind(p) == k and len(same_kind_children(p, k)) == 1:
            continue                                     # it carries on its parent's chain
        members, cur = [b['n']], b['n']
        while len(same_kind_children(cur, k)) == 1:
            cur = same_kind_children(cur, k)[0]
            members.append(cur)
        root, a = CANT_TELL, b['parent']
        while a is not None:
            if parts_of(roles_.get(a), body_type):
                root = roles_[a]
                break
            if a in placed:
                root = placed[a]['part']
                break
            a = by_n[a]['parent']
        out.append(Chain(kind=k, bones=members, root=root))
    return out
