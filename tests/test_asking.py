import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rigswap import asking


class _Provider:
    """A fake provider: each question sleeps its own time, and how many are out at once is kept."""

    def __init__(self, sleeps, fail=None):
        self.sleeps, self.fail = sleeps, fail
        self.lock, self.out, self.most, self.order = threading.Lock(), 0, 0, []

    def __call__(self, question, pictures, schema, default, **how):
        with self.lock:
            self.out += 1
            self.most = max(self.most, self.out)
        try:
            time.sleep(self.sleeps[question])
            if question == self.fail:
                raise RuntimeError(f'{question} broke')
            return {'said': question}, {'source': 'ai', 'error': None, 'cost': 0.001,
                                        'seconds': self.sleeps[question], 'reply': {'said': question}}
        finally:
            with self.lock:
                self.out -= 1
                self.order.append(question)


def test_the_pool_size_bounds_how_many_are_out_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv(asking.ENV, '2')
    provider = _Provider({f'q{i}': 0.3 for i in range(6)})
    t0 = time.monotonic()
    asking.ask_all([(f'q{i}', [], {}, {}) for i in range(6)], asking.unrecorded(provider))
    assert provider.most == 2 and time.monotonic() - t0 >= 0.85             # three waves of two
    monkeypatch.setenv(asking.ENV, '1')
    assert asking.at_once() == 1
    monkeypatch.delenv(asking.ENV)
    assert asking.at_once() == asking.AT_ONCE
    for bad in ('0', 'four', '-2'):
        monkeypatch.setenv(asking.ENV, bad)
        with pytest.raises(ValueError, match=asking.ENV):
            asking.at_once()


def test_the_effort_goes_to_every_ask_only_when_given(tmp_path):
    seen = []
    asking.ask_all([('a', [], {}, {})], asking.unrecorded(lambda q, p, s, d: seen.append(q) or (d, {})))
    asking.ask_all([('b', [], {}, {}), ('c', [], {}, {})],
                   asking.unrecorded(lambda q, p, s, d, effort: seen.append(effort) or (d, {})),
                   effort='medium')
    assert seen == ['a', 'medium', 'medium']


def test_the_pool_wait_is_not_on_the_asks_deadline(tmp_path, monkeypatch):
    from rigswap import ai
    from rigswap.library.asks import Ask
    monkeypatch.setenv(asking.ENV, '1')
    monkeypatch.setattr(ai, 'key', lambda: 'test')
    sent = []

    def post(body, key, info, deadline_s):
        sent.append(deadline_s)
        return {'usage': {'cost': 0.001}, 'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(ai, '_post', post)

    def provider(q, p, s, d, **how):
        if q == 'first':
            time.sleep(0.1)
            return d, {}
        return ai.ask(q, p, s, d, route=Ask(efforts=('high',), deadline_s=0.05))
    result = asking.ask_all([('first', [], {}, {}), ('queued', [], {}, {})],
                            asking.unrecorded(provider))
    assert sent == [0.05] and result[1][1]['source'] == 'ai'      # waited longer than its deadline, then was sent
    assert result[1][1]['queued_s'] >= 0.1
