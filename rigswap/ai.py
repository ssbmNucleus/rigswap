"""The one way the AI is asked: a question, some pictures, a JSON schema, a default.

How each round is asked (the model on OpenRouter, the one provider it is pinned to, its reasoning effort and its
deadline) is that round's entry in rigswap.library.asks, by the name its asks are recorded under: ``route``, which
watch.recorded passes; every ask names its round, and a route with no entry is an error. Any failure that remains
(no key, an error, a timeout, an answer outside the schema) returns the default, and the record says so; the round
that asked marks each fact the default stands in for as defaulted (vocabulary.Description.defaulted), which is no
reading. Answers are cached by the exact model, question, pictures, schema and the reasoning effort that answered, so
a repeated run asks only what changed (an answer the lower effort gave after a deadline is no answer to the higher one).

Two exceptions to asking once, each at most once, the earlier attempts kept in ``info['attempts']``:
- no usable answer is sent once more, the identical pictures and question. That is no answer at all (a reply with no
  JSON object: the model stopped mid-reply, or sent nothing; or a transport failure before any reply began: connection
  aborted, read or write timeout, HTTP 5xx or 429), or an answer code can't read (its JSON doesn't parse, it doesn't
  fit the schema, the reply's body is malformed). An HTTP 4xx other than 429 refuses the request itself, which the
  same request sent again can't change, so it is not sent again;
- a reply that does not come whole: the deadline passes, or the provider's stream ends before the model finished (an
  error body with no choices, or the connection breaking after the reply began). The same ask is sent once more at the
  next lower effort the round's entry lists (high, then medium). A second such failure fails the ask, except that a
  reply cut off with no lower effort to go to is sent once more as it was. The lowering is the remedy for a model
  that cannot finish: tikal's trunk-bones ask at high reasoned for about 5 minutes until Together dropped the stream,
  three runs out of three (opt6, opt7), and a second send at high died the same way; at medium it answers.
Each send's deadline starts when it gets its turn to send, not when the ask was numbered: waiting in a run's pool or
for the line is not the model's time, and an ask lowered in effort for having waited would be a judgement given away
for nothing (pilot 8: 10 of spiderman's 46 round-3 asks were lowered without ever being sent).
Neither is a retry of a judgement (an answer that fits the schema is never re-asked, right or wrong; one outside it
is no answer the round can read) nor a provider fallback (the provider stays pinned). Independent questions are
numbered and dispatched by ``rigswap.asking.ask_all``. An ask may set its own reasoning effort. ``ask`` is the
default way rigswap.swap asks; any callable with its signature will do."""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import requests

from . import asking
from .library import asks as library
from . import pictures as sent_pictures

URL = 'https://openrouter.ai/api/v1/chat/completions'
# No fallbacks: when Together is slow, OpenRouter's fallback provider received no pictures and still answered with
# plausible-looking cells (round 4, 2026-10-10). No answer is better than a blind one; the default takes over. Each
# round's entry (rigswap.library.asks) names its one provider.
# Each wait while connecting and while sending the pictures (requests uses the one number for both; when several
# asks send at once, a send can wait 10 s for the line), and each wait for more bytes of the reply. While the model
# works, OpenRouter sends a few bytes of keep-alive every 3 s or so.
CONNECT_S, READ_S = 30, 30
# How many asks send their pictures at the same time, over every rigswap process on this machine (several runs may go
# at once). The line out is slow: eighteen round-3 asks of about 1.3 MB each sent at once all but one failed to finish
# sending within CONNECT_S (Zelda, 2026-10-10). Only the sending waits its turn; the waits for the replies still
# overlap, so asks made together are still asked at once. The deadline includes the pool wait and the wait for a
# turn to send. The lower-effort attempt gets its own clock when made, before it waits for a turn.
SENDING_AT_ONCE = 2
# Where the turns to send are kept: one small file per turn, locked while an ask sends.
TURNS_DIR = Path(tempfile.gettempdir()) / 'rigswap-sending'
TURN_POLL_S = 0.02         # how often a waiting ask looks for a free turn


if os.name == 'nt':
    import msvcrt

    def _lock(fd):
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


class Turns:
    """``at_once`` turns to send, shared by every process that uses the same ``folder``: a turn is a lock on one small
    file there, held while an ask sends, so the asks of several runs going at once wait for each other too (a
    semaphore holds only within one process). ``with turns:`` waits for a free turn and holds it; the system lets go of
    a lock whose process ends."""

    def __init__(self, at_once, folder):
        self.at_once, self.folder = at_once, Path(folder)
        self._held = threading.local()           # the file this thread holds its turn by

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(self.at_once):
                fd = os.open(self.folder / f'turn-{i}', os.O_RDWR | os.O_CREAT)
                if _lock(fd):
                    self._held.fd = fd
                    return self
                os.close(fd)
            time.sleep(TURN_POLL_S)

    def __exit__(self, *exc):
        fd = self._held.fd
        try:
            _unlock(fd)
        finally:
            os.close(fd)


_sending = Turns(SENDING_AT_ONCE, TURNS_DIR)

# Where the key comes from, in order: the environment, the Nucleus app's key file, then 1Password when
# RIGSWAP_OPENROUTER_OP names an op:// reference. Never a file in
# this repo. The file is tried before `op read` because `op` hangs for its whole timeout when it can't prompt.
OP_REFERENCE = os.environ.get('RIGSWAP_OPENROUTER_OP', '')
_KEY = None   # looked up once per process

SYSTEM = ('You answer one narrow question about pictures of a 3D model. Answer only from what the pictures show. '
          'If the pictures do not show it, say so with the allowed "can\'t tell" answer. Reply with one JSON object '
          'that matches this JSON schema, and nothing else:\n{schema}')


KEY_FILE = Path.home() / '.nucleus-secrets' / 'openrouter.key'   # the Nucleus app's cached key, outside any repo

# The relay. With RIGSWAP_ASK_RELAY naming a folder, no ask goes to OpenRouter: each is written there as
# ask-<id>.json (the instructions with its schema, the question, the paths of the pictures it would send) and is
# answered by whoever reads the folder. The Nucleus app's MCP server hands it to the user's own AI and writes that
# AI's reply as answer-<id>.json ({"answer": {...}}). An answer outside the schema is handed back in problem-<id>.json
# and the ask waits for another, RELAY_TRIES answers in all; done-<id>.json says the ask is over, answered or not.
# No answer within RELAY_WAIT_S is the default, as any failed ask is. Relayed answers are cached under RELAY_MODEL and
# the round's name, never mixed with a model's: a round asked again on purpose ('core fill again', the tripwire's second
# try) is a new ask to the client, as it is a second model's elsewhere; keyed by 'client' alone it came back from the
# cache and the same wreck was built twice (Frieza, 2026-11-18).
RELAY_ENV = 'RIGSWAP_ASK_RELAY'
RELAY_MODEL = 'client'
RELAY_WAIT_S = float(os.environ.get('RIGSWAP_ASK_RELAY_WAIT_S') or 20 * 60)
RELAY_POLL_S = 0.25
RELAY_TRIES = 3
_relay_ids = itertools.count(1)
_relay_lock = threading.Lock()


def relay():
    """The relay folder (RIGSWAP_ASK_RELAY), or None when asks go to OpenRouter."""
    value = (os.environ.get(RELAY_ENV) or '').strip()
    return Path(value) if value else None


def answers(shared, own):
    """The folder a run's answers are cached in: ``shared``, which every run reads (an ask a model answered is the
    same ask, and the same model's answer, for any download that draws the same pictures), or, through the relay,
    ``own``: the import's own folder, which a change to that import (costume --fix) reads again and no other import
    does. A relayed answer is whoever answered it then: its key can't say which assistant that was, and in a shared
    cache a second assistant importing the same file onto the same fighter was handed the first one's answers and
    never asked (Sonnet's and Opus's imports of Taz, 2026-11-19)."""
    return Path(own) if relay() else Path(shared)


def key():
    global _KEY
    if _KEY:
        return _KEY
    k = (os.environ.get('OPENROUTER_API_KEY') or '').strip()
    if not k and KEY_FILE.is_file():
        k = KEY_FILE.read_text(encoding='utf-8').strip()
    if not k and OP_REFERENCE:
        try:
            k = subprocess.run(['op', 'read', OP_REFERENCE], capture_output=True, text=True, timeout=60,
                               check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            k = ''
    _KEY = k or None
    return _KEY


def _image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    kind = 'png' if str(path).lower().endswith('.png') else 'jpeg'
    return {'type': 'image_url', 'image_url': {'url': f'data:image/{kind};base64,{data}'}}


def provider(route=None):
    """The provider ``route``'s entry pins its asks to, with no fallbacks; None where the entry names none (a model
    its own maker alone serves: OpenRouter has no other provider to fall back to)."""
    pinned = library.entry(route).provider
    return None if pinned is None else {'order': [pinned], 'allow_fallbacks': False}


def settings(effort=None, route=None):
    """Everything an ask sends besides its question, its pictures and its schema: the system prompt (before its schema
    goes in), the model, the pinned provider, the reasoning settings and the reply's format. Resolve ``route`` as the
    sender does; None uses the library's baseline for request metadata without a round (not for sending an ask)."""
    how = library.entry(route if route is not None else library.DEFAULT)
    return {'system': SYSTEM, 'model': how.model, 'provider': provider(how),
            'reasoning': {'effort': effort or how.effort()},
            'response_format': {'type': 'json_object'}}


def settings_id(effort=None, route=None):
    """A short name for ``settings(effort, route)``, which each ask's record keeps (``info['settings']``): the same settings,
    the same name."""
    return hashlib.sha256(json.dumps(settings(effort, route), sort_keys=True).encode('utf-8')).hexdigest()[:16]


def request_body(question, pictures, schema, effort=None, route=None):
    """The body sent by ask; its JSON UTF-8 length includes base64 pictures and all request overhead."""
    sent = settings(effort, library.entry(route))
    return {
        'model': sent['model'],
        **({} if sent['provider'] is None else {'provider': sent['provider']}),
        'reasoning': sent['reasoning'],
        'messages': [
            {'role': 'system', 'content': sent['system'].format(schema=json.dumps(schema))},
            {'role': 'user', 'content': [{'type': 'text', 'text': question}] + [_image_part(p) for p in pictures]},
        ],
        'response_format': sent['response_format'],
        'usage': {'include': True},
    }


def _cache_key(question, pictures, schema, model, effort):
    h = hashlib.sha256()
    h.update(model.encode()); h.update(question.encode()); h.update(json.dumps(schema, sort_keys=True).encode())
    h.update(json.dumps({'effort': effort}, sort_keys=True).encode())
    for p in pictures:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


class NoJSONObject(ValueError):
    """The reply has no JSON object in it at all: cut off, empty, or not text."""


class NoWholeReply(requests.Timeout):
    """The reply did not come whole: asked once more at the next lower effort."""


class DeadlinePassed(NoWholeReply):
    """No whole reply came within the attempt's deadline."""


class ReplyCutOff(NoWholeReply):
    """The reply began and stopped before it was whole: the provider's stream ended (an error body with no choices),
    or the connection broke after the reply began."""


class OutsideSchema(ValueError):
    """The reply's JSON object does not fit the schema (_check says how)."""


def _no_answer(e):
    """True when ``e`` means the provider gave no answer at all: a reply with no JSON object, a connection that broke or
    timed out (not the deadline: that is checked apart), or an HTTP 5xx or 429."""
    if isinstance(e, NoJSONObject):
        return True
    if isinstance(e, NoWholeReply):         # asked once more lower, not sent again as it was
        return False
    if isinstance(e, requests.HTTPError):
        code = getattr(e.response, 'status_code', None)
        return code == 429 or (code is not None and code >= 500)
    return isinstance(e, (requests.ConnectionError, requests.Timeout))


def _unusable(e):
    """True when ``e`` means an answer came that code can't read: its JSON doesn't parse, it doesn't fit the schema
    (OutsideSchema), or the reply's body lacks what every reply has."""
    return (isinstance(e, (KeyError, IndexError, TypeError, ValueError))
            and not isinstance(e, (NoJSONObject, requests.RequestException)))


def ask(question, pictures, schema, default, cache_dir=None, effort=None, route=None):
    """Returns ``(answer, info)``. ``answer`` is the AI's when it came back and fits the schema, else ``default``.
    ``route`` is how it is asked: the name of its entry in rigswap.library.asks (the name its asks are recorded
    under), or an Ask; a name with no entry, or none, is a ValueError. ``effort`` ('low', 'medium', 'high') is this
    ask's reasoning effort in place of the entry's first; None keeps the entry's.
    ``info`` has ``source`` ('ai' | 'cache' | 'default'), ``error``, ``seconds``, ``cost``, ``model``, ``effort`` (the
    last attempt's: the one that answered), ``provider`` (who answered it) and ``tokens`` (its prompt, completion and
    reasoning tokens), ``settings`` (settings_id of the final attempt), ``requested_settings`` (settings_id of the
    initial request, before retries), and ``reply``: what the AI sent back,
    parsed when it parsed, even when it was not used;
    ``attempts`` (only when the ask was sent again: no usable answer, or a deadline passed): [{reply, error, seconds,
    cost, effort, provider, tokens}] of the earlier attempts, with ``seconds`` and ``cost`` then the whole ask's."""
    pictures = sent_pictures.prepare(pictures, route)
    how = library.entry(route)
    requested_settings = settings_id(effort, how)
    started, queued = asking.clock.get() or (time.monotonic(), 0.0)
    info = {'source': 'default', 'error': None, 'seconds': 0.0, 'cost': 0.0,
            'queued_s': round(queued, 1), 'effort': effort or how.effort()}
    try:
        answer, info, reply = _ask(question, pictures, schema, default, cache_dir, how,
                                  effort or how.effort(), info, started,
                                  route_name=route if isinstance(route, str) else None)
    except Exception as e:
        e.info = dict(info, source='raised', error=f'{type(e).__name__}: {e}',
                      seconds=time.monotonic() - started, settings=settings_id(info.get('effort'), how),
                      requested_settings=requested_settings)
        raise
    info['model'] = RELAY_MODEL if relay() else how.model
    info['settings'] = settings_id(info.get('effort'), how)
    info['requested_settings'] = requested_settings
    info['reply'] = reply
    return answer, info


def _ask(question, pictures, schema, default, cache_dir, how, effort, info, t0, route_name=None):
    """The ask itself, as ``how`` (an Ask) asks it, or through the relay. Returns ``(answer, info, reply)``;
    ``reply`` is what came back (parsed when it parsed)."""
    body = request_body(question, pictures, schema, effort, how)
    info['request_bytes'] = len(json.dumps(body).encode('utf-8'))
    relayed = relay()
    model = f'{RELAY_MODEL}:{route_name}' if relayed else how.model
    def cached(effort):
        return Path(cache_dir) / f'{_cache_key(question, pictures, schema, model, effort)}.json'
    cache = cached(effort) if cache_dir else None
    if cache and cache.is_file():
        hit = json.loads(cache.read_text(encoding='utf-8'))
        return (hit['answer'], {'source': 'cache', 'error': None, 'seconds': 0.0, 'cost': 0.0, 'effort': effort},
                hit['answer'])
    if relayed:
        return _relayed(relayed, question, pictures, schema, default, info, cache, route_name)

    k = key()
    if not k:
        info['error'] = 'no key'
        return default, info, None

    spent = []                      # the earlier attempts
    sent_again = lowered = False    # no usable answer, sent once more; a deadline passed, asked once more lower
    while True:
        reply, cost = None, 0.0
        for fact in ('tokens', 'provider'):                    # set by this attempt's reply, when one comes
            info.pop(fact, None)
        try:
            got = _post(body, k, info, how.deadline_s)
            usage = got.get('usage') or {}
            cost = float(usage.get('cost') or 0.0)
            info['cost'] = cost + sum(a['cost'] for a in spent)
            info['tokens'] = {'prompt': usage.get('prompt_tokens'), 'completion': usage.get('completion_tokens'),
                              'reasoning': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')}
            info['provider'] = got.get('provider')
            if 'choices' not in got:        # the provider's stream ended before the model finished: an error body
                raise ReplyCutOff(f'no choices in the reply: {got.get("error")}')
            reply = got['choices'][0]['message']['content']
            answer = reply = _json_object(reply)
            problem = _check(answer, schema)
            if problem:
                raise OutsideSchema(problem)
        except (requests.RequestException, KeyError, ValueError, IndexError, TypeError) as e:
            info['error'] = (f'answer outside schema: {e}' if isinstance(e, OutsideSchema) else
                             f'{"ValueError" if isinstance(e, NoJSONObject) else type(e).__name__}: {e}')
            lower = how.lower(effort) if isinstance(e, NoWholeReply) and not lowered else None
            again = (not lower and not sent_again
                     and (_no_answer(e) or isinstance(e, ReplyCutOff) or _unusable(e)))
            if lower or again:
                spent.append({'reply': reply, 'error': info['error'], 'seconds': round(time.monotonic() - t0, 1),
                              'cost': cost, 'effort': effort, 'provider': info.get('provider'),
                              'tokens': info.get('tokens')})
                info['attempts'] = spent
                info['error'] = None
                if lower:            # no whole reply: once more at the next lower effort
                    lowered, effort = True, lower
                    body = request_body(question, pictures, schema, effort, how)
                    info['effort'] = effort
                else:                # no usable answer: once more, the same ask
                    sent_again = True
                continue
            _attempts(info, spent, cost, t0)
            return default, info, reply
        break
    _attempts(info, spent, cost, t0)
    info['source'] = 'ai'
    if cache:                    # under the effort that answered: a lower one's answer is not the higher one's
        cache = cached(effort)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({'answer': answer, 'question': question, 'pictures': pictures}), encoding='utf-8')
    return answer, info, reply


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=1), encoding='utf-8')
    os.replace(temporary, path)


def _relayed(folder, question, pictures, schema, default, info, cache, route_name=None):
    """One ask answered through the relay folder (see RELAY_ENV). -> ``(answer, info, reply)`` as _ask returns it:
    source 'ai' with ``answered_by`` 'client' for an answer that fits the schema, else the default."""
    with _relay_lock:
        ask_id = f'{os.getpid()}-{next(_relay_ids)}'
    answered, refused = folder / f'answer-{ask_id}.json', folder / f'problem-{ask_id}.json'
    _write_json(folder / f'ask-{ask_id}.json', {
        'id': ask_id, 'round': route_name, 'instructions': SYSTEM.format(schema=json.dumps(schema)),
        'question': question, 'schema': schema, 'pictures': [str(Path(p).resolve()) for p in pictures]})
    started = time.monotonic()
    deadline = started + RELAY_WAIT_S
    reply, answer = None, None
    try:
        for tries_left in range(RELAY_TRIES - 1, -1, -1):
            while not answered.is_file():
                if time.monotonic() > deadline:
                    info['error'] = f'no answer through the relay within {RELAY_WAIT_S:.0f} s'
                    return default, info, reply
                time.sleep(RELAY_POLL_S)
            try:
                got = json.loads(answered.read_text(encoding='utf-8'))
                reply = got.get('answer') if isinstance(got, dict) else None
                problem = _check(reply, schema)
            except (OSError, ValueError) as e:
                problem = f'unreadable answer: {e}'
            answered.unlink(missing_ok=True)
            if not problem:
                answer = reply
                break
            info['error'] = f'answer outside schema: {problem}'
            _write_json(refused, {'problem': problem, 'answer': reply, 'tries_left': tries_left})
        else:
            return default, info, reply
    finally:
        info['seconds'] = round(time.monotonic() - started, 1)
        _write_json(folder / f'done-{ask_id}.json', {'answered': answer is not None, 'error': info.get('error')})
    refused.unlink(missing_ok=True)
    info.update(source='ai', error=None, answered_by=RELAY_MODEL)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({'answer': answer, 'question': question, 'pictures': pictures}), encoding='utf-8')
    return answer, info, reply


def _attempts(info, spent, cost, t0):
    """The whole ask's seconds into ``info``; with earlier attempts, those too and the whole ask's cost."""
    info['seconds'] = time.monotonic() - t0
    if spent:
        info['attempts'] = spent
        info['cost'] = cost + sum(a['cost'] for a in spent)


def _post(body, k, timing=None, deadline_s=library.DEFAULT.deadline_s):
    """Send the ask and read the whole reply within ``deadline_s`` of getting a turn to send (SENDING_AT_ONCE): the
    send's one clock covers sending the pictures, the wait and reading the reply, and a reply still coming when it
    passes is closed. The wait for a turn is not on it. Returns the reply's JSON; past the deadline, requests.Timeout.
    ``timing``, a dict when given, adds how long the ask waited for its turn ('queued_s') and how long sending took,
    until the reply's first bytes ('sending_s'), over every attempt."""
    t0 = time.monotonic()
    with _sending:
        t1 = time.monotonic()
        _add(timing, 'queued_s', t1 - t0)
        deadline = t1 + deadline_s
        try:
            try:
                r = requests.post(URL, json=body, stream=True,
                                  timeout=(min(CONNECT_S, deadline_s), min(READ_S, deadline_s)),
                                  headers={'Authorization': f'Bearer {k}', 'Content-Type': 'application/json'})
            finally:
                _add(timing, 'sending_s', time.monotonic() - t1)
        except requests.Timeout as e:
            if time.monotonic() >= deadline:
                raise DeadlinePassed(f'no whole reply within the {deadline_s:g} s deadline') from e
            raise
    late = DeadlinePassed(f'no whole reply within the {deadline_s:g} s deadline')
    chunks = []
    with r:
        r.raise_for_status()
        try:
            for chunk in r.iter_content(chunk_size=65536):
                if time.monotonic() > deadline:
                    raise late
                chunks.append(chunk)
        except (requests.ConnectionError, requests.Timeout) as e:
            if isinstance(e, NoWholeReply):
                raise
            raise ReplyCutOff(f'the connection broke after the reply began: {e}') from e
    if time.monotonic() > deadline:
        raise late
    return json.loads(b''.join(chunks))


def _add(timing, key, seconds):
    """Add ``seconds`` to ``timing[key]`` (rounded to a tenth), when there is a ``timing``."""
    if timing is not None:
        timing[key] = round(timing.get(key, 0.0) + seconds, 1)


def _json_object(text):
    """The JSON object in the reply's text: from its first { to its last }. ValueError when there is none."""
    if not isinstance(text, str):
        raise NoJSONObject(f'the reply has no text: {text!r}')
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise NoJSONObject(f'no JSON object in the reply: {text[:200]!r}')
    return json.loads(text[start:end + 1])


_KINDS = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'integer': int, 'number': (int, float)}


def _check(value, schema, where='answer'):
    """A small check that the answer has the schema's shape: required keys, kinds and enum values, at every depth
    (properties, additionalProperties, items), and anyOf (fits one of them). Enough to catch a wrong-shaped reply; not
    a full JSON-schema validator."""
    if 'anyOf' in schema:
        problems = [_check(value, s, where) for s in schema['anyOf']]
        if all(problems):
            return ' or '.join(problems)
    if 'enum' in schema and not any(value == e and type(value) is type(e) for e in schema['enum']):
        return f'{where}={value!r} not in {schema["enum"]}'
    t = schema.get('type')
    if t in _KINDS and (not isinstance(value, _KINDS[t]) or (t in ('integer', 'number') and isinstance(value, bool))):
        return f'{where} is not {"an" if t[0] in "aeiou" else "a"} {t}'
    if isinstance(value, dict):
        for k in schema.get('required', []):
            if k not in value:
                return f'{where} is missing {k}'
        props, extra = schema.get('properties', {}), schema.get('additionalProperties')
        for k, v in value.items():
            spec = props.get(k, extra if isinstance(extra, dict) else None)
            if spec:
                problem = _check(v, spec, f'{where}.{k}')
                if problem:
                    return problem
    if isinstance(value, list) and isinstance(schema.get('items'), dict):
        for i, v in enumerate(value):
            problem = _check(v, schema['items'], f'{where}[{i}]')
            if problem:
                return problem
    return None
