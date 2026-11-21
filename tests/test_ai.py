import json
import time

import pytest

from rigswap import ai
from rigswap.library import asks as library

SCHEMA = {'type': 'object', 'properties': {'up': {'enum': ['+z', '-z']}}, 'required': ['up']}
R = 'describe round 1'          # every ask names its round
DEFAULT = {'up': '+z'}


class FakeResponse:
    """What requests.post(stream=True) returns: keep-alive whitespace, then the body (or keep-alive forever)."""

    def __init__(self, body=None, every=0.0):
        self.body, self.every, self.closed = body, every, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        while self.body is None:
            time.sleep(self.every)
            yield b'\n   \n'
        yield b'\n   \n' + json.dumps(self.body).encode()


@pytest.fixture(autouse=True)
def own_turns(monkeypatch, tmp_path):
    # the turns to send are the test's own: another process's asks on this machine never hold them
    monkeypatch.setattr(ai, '_sending', ai.Turns(ai.SENDING_AT_ONCE, tmp_path / 'turns'))


@pytest.fixture
def post(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []

    def use(response):
        monkeypatch.setattr(ai.requests, 'post', lambda *a, **kw: sent.append(kw) or response)
        return sent
    return use


def _reply(text):
    return {'choices': [{'message': {'content': text}}], 'usage': {'cost': 0.001}}


def test_reply_is_read_after_keep_alive(post):
    sent = post(FakeResponse(_reply('Here: {"up": "-z"} done')))
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer == {'up': '-z'} and info['source'] == 'ai' and info['cost'] == 0.001
    assert sent[0]['stream'] is True and sent[0]['timeout'] == (ai.CONNECT_S, ai.READ_S)


def test_one_ask_may_set_its_own_effort(post, tmp_path):
    sent = post(FakeResponse(_reply('{"up": "-z"}')))
    how = library.entry(R)
    effort = how.effort()
    own_effort = next(e for e in library.EFFORTS if e != effort)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert sent[-1]['json']['reasoning'] == {'effort': effort}
    assert info['effort'] == effort
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, effort=own_effort, route=R)
    assert sent[-1]['json']['reasoning'] == {'effort': own_effort} and info['effort'] == own_effort
    assert library.entry(R).effort() == effort                    # the round's default is left as it was
    # the cache keeps the override apart from the round's default effort
    ai.ask('q', [], SCHEMA, DEFAULT, cache_dir=tmp_path, effort=own_effort, route=R)
    assert ai.ask('q', [], SCHEMA, DEFAULT, cache_dir=tmp_path, effort=own_effort, route=R)[1]['source'] == 'cache'
    assert ai.ask('q', [], SCHEMA, DEFAULT, cache_dir=tmp_path, route=R)[1]['source'] == 'ai'

def test_the_turns_to_send_hold_across_processes(tmp_path):
    # several runs go at once, each its own process: a turn another process holds is not free here either
    import subprocess
    import sys
    held = 1.5
    script = '\n'.join(['import time', 'from rigswap import ai', f'with ai.Turns(1, {str(tmp_path)!r}):',
                        f'    print("held", flush=True); time.sleep({held})'])
    child = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'held'
        t0 = time.monotonic()
        with ai.Turns(1, tmp_path):
            waited = time.monotonic() - t0
        with ai.Turns(2, tmp_path):                 # a second turn is free at once
            pass
    finally:
        child.wait(10)
    assert waited > held / 2
    t0 = time.monotonic()
    with ai.Turns(1, tmp_path):                     # the child is gone, and its turn with it
        assert time.monotonic() - t0 < 0.5


class _Slow(FakeResponse):
    """A reply whose body comes after ``wait`` seconds."""
    def __init__(self, body, wait):
        super().__init__(body)
        self.wait = wait

    def iter_content(self, chunk_size):
        time.sleep(self.wait)
        yield from super().iter_content(chunk_size)


def test_a_reply_without_a_json_object_takes_the_default(post):
    for text in ('no braces at all', None, '} backwards {'):
        post(FakeResponse(_reply(text)))
        answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
        assert answer is DEFAULT and info['source'] == 'default' and 'ValueError' in info['error']


def test_each_round_is_asked_as_its_entry_says(post, monkeypatch):
    sent = post(FakeResponse(_reply('{"up": "-z"}')))
    cheap = library.Ask(model='some/cheaper-model', provider='elsewhere', efforts=('low',), deadline_s=60)
    monkeypatch.setitem(library.ASKS, 'describe round 9', cheap)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route='describe round 9')
    body = sent[-1]['json']
    assert body['model'] == 'some/cheaper-model' and body['reasoning'] == {'effort': 'low'}
    assert body['provider'] == {'order': ['elsewhere'], 'allow_fallbacks': False}
    assert info['model'] == 'some/cheaper-model' and info['effort'] == 'low'
    ai.ask('q', [], SCHEMA, DEFAULT, route=cheap)                   # an Ask itself, as the lab tries one
    assert sent[-1]['json']['model'] == 'some/cheaper-model'
    ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert sent[-1]['json']['model'] == library.DEFAULT.model
    assert sent[-1]['json']['provider'] == {'order': ['together'], 'allow_fallbacks': False}
    for unnamed in ('a round with no entry', None):                 # every ask names its round
        with pytest.raises(ValueError, match='every ask names its round'):
            ai.ask('q', [], SCHEMA, DEFAULT, route=unnamed)


def test_every_round_the_pipeline_records_has_an_entry():
    names = {'core up', 'core front', 'core look', 'core parts', 'core labels', 'core chains',
             'core forks',
             'core eyes', 'core eyes iris', 'core paint', 'core layer', 'core paint way'}
    assert names <= set(library.ASKS)
    assert all(a.efforts and set(a.efforts) <= set(library.EFFORTS) and a.deadline_s > 0 for a in library.ASKS.values())


def test_the_deadline_is_wall_clock(post, monkeypatch):
    from types import SimpleNamespace
    now = [0.0]
    # Advance wall time between keep-alives without depending on a loaded machine's scheduler.
    # Patch only ai's clock, not the process-wide time module used by pytest and threads.
    monkeypatch.setattr(ai, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    response = FakeResponse()
    def keep_alive(chunk_size):
        for tick in range(1, 7):
            now[0] = tick * 0.05
            yield b'\n   \n'
        pytest.fail('keep-alive bytes must not extend the wall-clock deadline')
    monkeypatch.setattr(response, 'iter_content', keep_alive)
    post(response)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high',), deadline_s=0.2))
    assert now[0] == pytest.approx(0.25) and info['seconds'] == pytest.approx(0.25)
    assert answer is DEFAULT and info['source'] == 'default' and 'deadline' in info['error'] and response.closed
    assert 'attempts' not in info                # one effort listed: nothing lower to ask at


def test_a_deadline_passed_is_asked_once_more_at_the_next_lower_effort(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []

    def fake(*a, **kw):                          # the high ask never finishes; the medium one answers
        sent.append(kw['json']['reasoning']['effort'])
        return FakeResponse(every=0.05) if len(sent) == 1 else FakeResponse(_reply('{"up": "-z"}'))
    monkeypatch.setattr(ai.requests, 'post', fake)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high', 'medium'), deadline_s=0.2))
    assert answer == {'up': '-z'} and info['source'] == 'ai' and info['error'] is None
    assert sent == ['high', 'medium'] and info['effort'] == 'medium'
    assert [a['effort'] for a in info['attempts']] == ['high'] and 'deadline' in info['attempts'][0]['error']
    assert info['seconds'] >= 0.2 and info['cost'] == pytest.approx(0.001)


def test_a_second_deadline_fails(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []
    monkeypatch.setattr(ai.requests, 'post', lambda *a, **kw: sent.append(kw['json']['reasoning']['effort'])
                        or FakeResponse(every=0.05))
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high', 'medium', 'low'),
                                                                     deadline_s=0.15))
    assert answer is DEFAULT and 'deadline' in info['error'] and sent == ['high', 'medium']   # never low
    assert info['effort'] == 'medium' and [a['effort'] for a in info['attempts']] == ['high']
    # an ask that set its own effort goes one lower on the ladder
    sent.clear()
    ai.ask('q', [], SCHEMA, DEFAULT, effort='medium', route=library.Ask(efforts=('high',), deadline_s=0.15))
    assert sent == ['medium', 'low']
    # a round asked at low is asked at low once more
    sent.clear()
    ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('low', 'low'), deadline_s=0.15))
    assert sent == ['low', 'low']
    assert library.ASKS['describe held'].efforts == ('low', 'low')


def test_an_answer_given_lower_after_a_deadline_is_cached_as_that_effort_and_each_attempt_recorded(monkeypatch,
                                                                                                    tmp_path):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []

    def fake(*a, **kw):                          # the high ask never finishes; the medium one answers, then any
        sent.append(kw['json']['reasoning']['effort'])
        if len(sent) == 1:
            return FakeResponse(every=0.05)
        body = dict(_reply('{"up": "-z"}'), provider='Together',
                    usage={'cost': 0.001, 'prompt_tokens': 10, 'completion_tokens': 2})
        return FakeResponse(body)
    monkeypatch.setattr(ai.requests, 'post', fake)
    how = library.Ask(efforts=('high', 'medium'), deadline_s=0.2)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, cache_dir=tmp_path, route=how)
    assert sent == ['high', 'medium'] and info['effort'] == 'medium' and info['provider'] == 'Together'
    medium_settings = info['settings']
    assert medium_settings == ai.settings_id('medium', how)
    assert info['requested_settings'] == ai.settings_id('high', how)
    assert info['tokens'] == {'prompt': 10, 'completion': 2, 'reasoning': None}
    first = info['attempts'][0]
    assert first['effort'] == 'high' and first['provider'] is None and first['tokens'] is None
    # a high ask again is no cache hit: the cached answer is medium's
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, cache_dir=tmp_path, route=how)
    assert info['source'] == 'ai' and sent[2] == 'high'
    assert info['settings'] == ai.settings_id('high', how) and info['settings'] != medium_settings
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, cache_dir=tmp_path, effort='medium', route=how)
    assert info['source'] == 'cache' and info['effort'] == 'medium' and len(sent) == 3
    assert info['settings'] == medium_settings
    assert info['requested_settings'] == medium_settings


def test_other_failures_are_not_asked_lower(monkeypatch):
    for text in ('{"up": "sideways"}', 'no braces'):
        sent = _sequence(monkeypatch, text, text)
        answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
        assert answer is DEFAULT and all(b['reasoning'] == {'effort': library.entry(R).effort()} for b in sent)


def _sequence(monkeypatch, *texts):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []
    replies = iter(texts)

    def fake(*a, **kw):
        sent.append(kw['json'])
        return FakeResponse(_reply(next(replies)))
    monkeypatch.setattr(ai.requests, 'post', fake)
    return sent


def test_a_reply_cut_off_before_any_json_is_asked_once_more(monkeypatch):
    sent = _sequence(monkeypatch, '{"hips', '{"up": "-z"}')
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer == {'up': '-z'} and info['source'] == 'ai' and info['error'] is None
    assert len(sent) == 2 and sent[0] == sent[1]
    assert len(info['attempts']) == 1
    first = info['attempts'][0]
    assert first['reply'] == '{"hips' and 'no JSON object' in first['error'] and first['cost'] == 0.001
    assert info['cost'] == pytest.approx(0.002)


def test_two_unusable_replies_fail_as_before(monkeypatch):
    sent = _sequence(monkeypatch, '{"hips', 'nothing')
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer is DEFAULT and info['source'] == 'default' and len(sent) == 2
    assert 'no JSON object in the reply' in info['error'] and info['reply'] == 'nothing'
    assert len(info['attempts']) == 1


def test_an_answer_that_fits_the_schema_is_never_asked_again(monkeypatch):
    sent = _sequence(monkeypatch, '{"up": "+z"}', '{"up": "-z"}')
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer == {'up': '+z'} and len(sent) == 1 and 'attempts' not in info


def test_an_answer_code_cant_read_is_sent_once_more(monkeypatch):
    """No silent defaults: an answer outside the schema, or one whose JSON doesn't parse, is no answer the round can
    read, so the same ask goes once more, as a reply with no JSON object does."""
    for text, error in (('{"up": "sideways"}', 'answer outside schema: answer.up='),
                        ('{"nope": 1}', 'answer outside schema: answer is missing up'),
                        ('{"up": "-z", "x": }', 'JSONDecodeError')):
        sent = _sequence(monkeypatch, text, '{"up": "-z"}')
        answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
        assert answer == {'up': '-z'} and info['source'] == 'ai' and info['error'] is None, text
        assert len(sent) == 2 and sent[0] == sent[1], text
        assert len(info['attempts']) == 1 and info['attempts'][0]['error'].startswith(error), text
        assert info['cost'] == pytest.approx(0.002), text


def test_an_answer_code_cant_read_twice_takes_the_default_and_says_why(monkeypatch):
    sent = _sequence(monkeypatch, '{"up": "sideways"}', '{"up": "upside"}')
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer is DEFAULT and info['source'] == 'default' and len(sent) == 2
    assert info['error'].startswith('answer outside schema') and info['reply'] == {'up': 'upside'}
    assert info['attempts'][0]['reply'] == {'up': 'sideways'}


def test_one_ask_is_sent_again_once_whatever_failed_first(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise ai.requests.ConnectionError('aborted')
        return FakeResponse(_reply('{"up": "sideways"}'))
    monkeypatch.setattr(ai.requests, 'post', fake)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer is DEFAULT and len(calls) == 2 and info['error'].startswith('answer outside schema')


def test_the_second_send_has_a_deadline_of_its_own(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []

    def fake(*a, **kw):
        sent.append(1)
        if len(sent) == 1:
            raise ai.requests.ConnectionError('aborted')
        time.sleep(0.25)
        return FakeResponse(_reply('{"hips'))
    monkeypatch.setattr(ai.requests, 'post', fake)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high',), deadline_s=0.2))
    assert answer is DEFAULT and len(sent) == 2       # sent again; the second send's own deadline then passed
    assert 'deadline' in info['error'] and len(info['attempts']) == 1


def test_a_reply_that_stops_before_it_is_whole_is_asked_once_more_lower(monkeypatch):
    # opt6 and opt7, 2026-10-12: tikal's trunk-bones ask at high reasoned for about 5 minutes until Together dropped
    # the stream ({'error': {'message': 'Upstream error ... stream ended unexpectedly', 'code': 502} in a 200 body),
    # and a second send at high died the same way, leaving her trunk without bones; at medium it answers
    for first in ('error body', 'broken stream'):
        monkeypatch.setattr(ai, 'key', lambda: 'not a key')
        calls = []

        def fake(*a, **kw):
            calls.append(kw['json'])
            if len(calls) > 1:
                return FakeResponse(_reply('{"up": "-z"}'))
            if first == 'error body':
                return FakeResponse({'error': {'message': 'Upstream error from Together: the stream ended',
                                               'code': 502}, 'usage': {'cost': 0.0}})
            broken = FakeResponse(_reply('{"up": "-z"}'))

            def cut(chunk_size):
                yield b'{"choices'
                raise ai.requests.ConnectionError('connection reset mid-reply')
            broken.iter_content = cut
            return broken
        monkeypatch.setattr(ai.requests, 'post', fake)
        answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high', 'medium')))
        assert answer == {'up': '-z'} and info['source'] == 'ai' and info['error'] is None, first
        assert len(calls) == 2 and [c['reasoning']['effort'] for c in calls] == ['high', 'medium'], first
        assert info['effort'] == 'medium' and len(info['attempts']) == 1, first
        assert info['attempts'][0]['effort'] == 'high', first


def test_a_reply_cut_off_with_no_lower_effort_is_sent_once_more_as_it_was(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    calls = []

    def fake(*a, **kw):
        calls.append(kw['json'])
        if len(calls) == 1:
            return FakeResponse({'error': {'message': 'the stream ended', 'code': 502}, 'usage': {'cost': 0.0}})
        return FakeResponse(_reply('{"up": "-z"}'))
    monkeypatch.setattr(ai.requests, 'post', fake)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high',)))
    assert answer == {'up': '-z'} and len(calls) == 2 and calls[0] == calls[1]
    assert len(info['attempts']) == 1 and 'no choices' in info['attempts'][0]['error']


def test_a_transport_failure_is_asked_once_more(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    calls = []

    def fake(*a, **kw):
        calls.append(kw['json'])
        if len(calls) == 1:
            raise ai.requests.ConnectionError("('Connection aborted.', TimeoutError('The write operation timed out'))")
        return FakeResponse(_reply('{"up": "-z"}'))
    monkeypatch.setattr(ai.requests, 'post', fake)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
    assert answer == {'up': '-z'} and info['source'] == 'ai' and info['error'] is None
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len(info['attempts']) == 1 and 'ConnectionError' in info['attempts'][0]['error']
    assert info['attempts'][0]['reply'] is None


def test_two_transport_failures_fail_and_a_4xx_is_not_asked_again(monkeypatch):
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    for code, expected in ((503, 2), (429, 2), (400, 1)):
        calls = []

        class Bad(FakeResponse):
            def raise_for_status(self):
                raise ai.requests.HTTPError(f'{code}', response=type('R', (), {'status_code': code})())

        monkeypatch.setattr(ai.requests, 'post', lambda *a, **kw: calls.append(1) or Bad(_reply('x')))
        answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=R)
        assert answer is DEFAULT and 'HTTPError' in info['error'] and len(calls) == expected


class _QueuedTurns:
    """Advance only the sender's clock while each turn waits, then admit the send."""

    def __init__(self, monkeypatch, waits):
        from types import SimpleNamespace
        self.now, self.held, self.order = 0.0, False, []
        self.waits = iter(waits)
        monkeypatch.setattr(ai, 'time', SimpleNamespace(monotonic=lambda: self.now))
        monkeypatch.setattr(ai, '_sending', self)

    def __enter__(self):
        assert not self.held
        self.order.append('wait')
        self.now += next(self.waits)
        self.held = True
        self.order.append('acquired')
        return self

    def __exit__(self, *exc):
        assert self.held
        self.held = False
        self.order.append('released')

    def send(self, text):
        assert self.held, 'a request must acquire its turn before sending'
        self.order.append('sent')
        self.now += 0.1
        return FakeResponse(_reply(text))


def test_the_wait_for_a_turn_is_not_on_the_deadline(monkeypatch):
    # the line held by another ask past the deadline: the ask still goes out when the line frees, its clock starting
    # then, and the wait is kept as queued_s
    turns = _QueuedTurns(monkeypatch, [0.5])
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []
    monkeypatch.setattr(ai.requests, 'post', lambda *a, **kw: sent.append(1) or turns.send('{"up": "-z"}'))
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high', 'medium'), deadline_s=0.3))
    assert answer == {'up': '-z'} and len(sent) == 1 and info['effort'] == 'high' and 'attempts' not in info
    assert info['queued_s'] == pytest.approx(0.5) and info['seconds'] == pytest.approx(0.6)
    assert turns.order == ['wait', 'acquired', 'sent', 'released'] and not turns.held


def test_a_second_send_waits_for_the_line_and_then_goes(monkeypatch):
    # the first attempt gets no answer at once; the line is taken as it lets go and comes back after the deadline:
    # the second send still goes, with a clock of its own, and both waits are in queued_s
    turns = _QueuedTurns(monkeypatch, [0.5, 0.7])
    monkeypatch.setattr(ai, 'key', lambda: 'not a key')
    sent = []

    def fake(*a, **kw):
        sent.append(1)
        return turns.send('{"hips' if len(sent) == 1 else '{"up": "-z"}')
    monkeypatch.setattr(ai.requests, 'post', fake)
    answer, info = ai.ask('q', [], SCHEMA, DEFAULT, route=library.Ask(efforts=('high',), deadline_s=0.3))
    assert answer == {'up': '-z'} and len(sent) == 2 and len(info['attempts']) == 1
    assert info['source'] == 'ai' and info['effort'] == 'high' and info['error'] is None
    assert 'no JSON object' in info['attempts'][0]['error']
    assert info['queued_s'] == pytest.approx(1.2) and info['seconds'] == pytest.approx(1.4)
    assert turns.order == ['wait', 'acquired', 'sent', 'released'] * 2 and not turns.held
