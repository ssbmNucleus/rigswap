"""How a round sends its independent asks together: numbered first, in the order the round makes them, then sent at
most ``at_once()`` at a time; the answers come back in that order, whichever finished first.

``AT_ONCE`` is how many of one run's asks may be out at once, waiting for a turn to send or for the answer. What
was measured (pilot 6: 901 asks, six runs at once; and the line, 2026-10-12):
- The provider sets no limit near it. The OpenRouter key has no request limit, a paid model has none on OpenRouter's
  side, and the provider the asks library pins sent no 429 and no 5xx; its answers were no slower with 17 to 26 asks
  waiting on it than with 1 to 4 (round 2's median 11 to 15 s either way). Its four timeouts came with 5 to 9 waiting.
- The line out does. Every picture of every ask goes up a line of about 0.5 MB/s for one send and 0.65 to 0.8 MB/s for
  several, which ai.SENDING_AT_ONCE shares among every run on the machine: a round-3 ask (1.4 to 1.8 MB) waited 32 s,
  median, for its turn to send, against 2 s for its answer.
So a round of big pictures (round 3, round 4, the pick) goes at the line's pace past a few asks out, whatever the cap,
and a round of small pictures and long answers (round 2: 0.6 MB, 13 s) goes faster the more are out: 8 keep it as fast
as it went in pilot 6, where rounds 2, the twins and held already went 8 at a time (rounds 3 and 4 had no cap, and 69
of one run's round-3 asks waited for the line at once). In a batch the line is full anyway, and the cap keeps one run's
many asks from taking the turns to send while the other runs' rounds wait. A send's deadline (library/asks.py) starts
at its turn to send: neither the pool wait nor the wait for the line is on it. ``RIGSWAP_ASKS_AT_ONCE`` sets it for
a run (``rigswap swap --asks-at-once``; ``rigswap batch --asks-at-once`` passes it to each run's process).

An ask that raises (a bug, not a failed ask: ai.ask returns those as its default) fails the round, as before: every
other ask of the round still finishes and is recorded, then the first error in the round's order is raised."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import time
import os

AT_ONCE = 8
ENV = 'RIGSWAP_ASKS_AT_ONCE'
clock = ContextVar('ask_clock', default=None)


def made(ask, started=None):
    """Reserve the clock now, before dispatch; carry it into the worker without changing provider signatures."""
    started = time.monotonic() if started is None else started

    def call(*args, **kwargs):
        token = clock.set((started, time.monotonic() - started))
        try:
            return ask(*args, **kwargs)
        finally:
            clock.reset(token)
    return call


def at_once():
    """How many of this run's asks may be out at once: RIGSWAP_ASKS_AT_ONCE when set (a whole number, 1 or more),
    else AT_ONCE."""
    given = os.environ.get(ENV, '').strip()
    if not given:
        return AT_ONCE
    if not given.isdigit() or int(given) < 1:
        raise ValueError(f'{ENV} must be a whole number of 1 or more, not {given!r}')
    return int(given)


def ask_all(requests_, ask, effort=None):
    """Independent asks of one round at the same time, at_once() at most. ``requests_``: a list of (question,
    pictures, schema, default); ``ask``: how each is asked, watch.recorded's, which numbers them all, in list order,
    before the first is sent (its ``numbered``), so the run's asks/ folders and the order its page lists them in are
    the round's, never the order the answers came back in; or ``unrecorded(ask)``, said outright, where nothing is
    kept. Any other callable is refused: a wrapper that hid ``numbered`` would number them as they came back.
    ``effort``, when given, is passed to every ask; when None, nothing is passed. Returns [(answer, info)] in list
    order."""
    requests_ = list(requests_)
    ahead = getattr(ask, 'numbered', None)
    if ahead is None:
        raise TypeError(f'{ask!r} numbers nothing: pass watch.recorded(...), or asking.unrecorded(...) where nothing '
                        'is kept')
    askers = [ahead(question=r[0]) for r in requests_]
    how = {'effort': effort} if effort else {}
    if len(requests_) <= 1:
        return [one(*r, **how) for one, r in zip(askers, requests_)]
    with ThreadPoolExecutor(max_workers=min(at_once(), len(requests_))) as pool:
        futures = [pool.submit(one, *r, **how) for one, r in zip(askers, requests_)]
    return [f.result() for f in futures]


def unrecorded(ask):
    """``ask`` for ask_all where nothing is kept (a test's fake provider, a measurement outside a run): it has no
    numbers to give, and says so."""
    def asking(*a, **how):
        return ask(*a, **how)
    asking.numbered = lambda question: made(ask)
    return asking
