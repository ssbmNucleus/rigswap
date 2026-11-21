"""Cap vertex influences after prep changes a mesh; identify its face pieces."""
import numpy as np


PIECE = 'rigswap_piece'     # the face attribute naming each face's piece in the weighted .blend files


TINY = 1e-4    # a weight under this share of its vertex is dropped (interpolation noise, invisible when posed)


FADE = 6       # the cap keeps of each weight w it keeps 1 - (d/w)**FADE, d the largest it drops: a bone kept just ahead


def normalize_and_cap(w, max_influences=None, tiny=TINY):
    """Rows normalized to 1, weights under ``tiny`` dropped, at most ``max_influences`` kept (the largest), normalized
    again. Each weight kept fades as the largest weight dropped (d) comes near it: w becomes w * (1 - (d/w)**FADE).
    A bone kept just ahead of one dropped carries almost nothing, so where two vertices side by side swap their last
    bone kept and their first dropped (the third and fourth bone of one about equal on both), neither carries much of
    the bone the other lacks, and the surface between them does not part when those two bones move apart; a hard cut
    would give each a quarter of a bone the other has none of. A bone well ahead of the first dropped is as it was.
    The weights come out the same whatever order the bones are in, and a row whose kept bones all equal the one
    dropped (no kept bone ahead of it) keeps them as they are.
    Returns (weights, number of rows that had more than the cap). A row with a weight that is not a finite number
    comes out all NaN: it is never made finite here, so the finite check fails on it."""
    w = np.asarray(w, float).clip(0.0, None)
    bad = ~np.isfinite(w).all(1)

    def norm(x):
        s = x.sum(1, keepdims=True)
        with np.errstate(invalid='ignore'):                     # inf / inf: those rows are NaN below anyway
            out = np.divide(x, s, out=np.zeros_like(x), where=s > 0)
        out[bad] = np.nan
        return out

    w = norm(w)
    w[w < tiny] = 0.0
    capped = 0
    if max_influences:
        over = (w > 0).sum(1) > max_influences
        capped = int(over.sum())
        if capped:
            rows = w[over]
            drop = np.argsort(-rows, axis=1, kind='stable')[:, max_influences:]
            d = np.take_along_axis(rows, drop[:, :1], axis=1)          # the largest weight dropped
            np.put_along_axis(rows, drop, 0.0, axis=1)
            kept = rows > 0
            faded = np.where(kept, rows * (1.0 - (d / np.where(kept, rows, 1.0)) ** FADE), 0.0).clip(0.0, None)
            ahead = faded.sum(1) > 0
            rows[ahead] = faded[ahead]
            w[over] = rows
    return norm(w), capped


def cap_groups(objects, max_influences=None):
    """Hold the vertex-group weights of the meshes ``objects`` to the rule the solve leaves them by (normalize_and_cap:
    each vertex's weights normalized, those under TINY dropped, at most ``max_influences`` kept, the largest, faded as
    the largest dropped comes near them), as a
    later step that changes them must: a decimation that merges vertices mixes their weights, and the merged vertex can
    end up on more bones than the cap, or not summing to 1. Every group is a bone's (weights.solve leaves no other).
    Returns {'cap', 'most_found': the most bones any vertex had, 'vertices_over': how many had more than the cap,
    'vertices_changed'}."""
    most = over = changed = 0
    for ob in objects:
        n, k = len(ob.data.vertices), len(ob.vertex_groups)
        if not n or not k:
            continue
        w = np.zeros((n, k))
        for v in ob.data.vertices:
            for g in v.groups:
                w[v.index, g.group] = g.weight
        most = max(most, int((w > 0).sum(1).max()))
        capped, count = normalize_and_cap(w, max_influences)
        over += count
        rows = np.flatnonzero(np.abs(capped - w).max(1) > 0)
        changed += len(rows)
        for v in rows:
            for c in np.flatnonzero((w[v] > 0) | (capped[v] > 0)):
                group = ob.vertex_groups[int(c)]
                if capped[v, c] > 0:
                    group.add([int(v)], float(capped[v, c]), 'REPLACE')
                else:
                    group.remove([int(v)])
    return {'cap': max_influences, 'most_found': most, 'vertices_over': over, 'vertices_changed': changed}
