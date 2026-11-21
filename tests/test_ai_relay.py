"""Asks answered through the relay folder (RIGSWAP_ASK_RELAY): what the Nucleus app's MCP server reads and writes."""
import json
import threading
import time

import pytest

from rigswap import ai

SCHEMA = {'type': 'object', 'properties': {'up': {'enum': ['+z', '-z']}}, 'required': ['up'],
          'additionalProperties': False}
R = 'describe round 1'
DEFAULT = {'up': '+z'}


@pytest.fixture
def relay(tmp_path, monkeypatch):
    folder = tmp_path / 'asks'
    monkeypatch.setenv(ai.RELAY_ENV, str(folder))
    monkeypatch.setattr(ai, 'RELAY_POLL_S', 0.01)
    monkeypatch.setattr(ai, 'key', lambda: pytest.fail('a relayed ask never reaches OpenRouter'))
    return folder


def _picture(tmp_path):
    from PIL import Image
    path = tmp_path / 'view.png'
    Image.new('RGB', (64, 64), 'red').save(path)
    return path


def _answer_when_asked(folder, answers, seen):
    """Play the app: wait for each ask, record it, write the next answer (and read any problem it got back)."""
    def run():
        for answer in answers:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                asks = [a for a in sorted(folder.glob('ask-*.json')) if not (folder / a.name.replace('ask-', 'done-'))
                        .exists()] if folder.is_dir() else []      # an ask still open
                if asks and not (folder / asks[0].name.replace('ask-', 'answer-')).exists():
                    break
                time.sleep(0.01)
            record = json.loads(asks[0].read_text(encoding='utf-8'))
            seen.append(record)
            target = folder / f"answer-{record['id']}.json"
            target.write_text(json.dumps({'answer': answer}), encoding='utf-8')
            while target.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_relayed_ask_is_written_out_and_answered_by_the_client(relay, tmp_path):
    seen = []
    thread = _answer_when_asked(relay, [{'up': '-z'}], seen)
    answer, info = ai.ask('Which way is up?', [_picture(tmp_path)], SCHEMA, DEFAULT, cache_dir=tmp_path / 'cache',
                          route=R)
    thread.join(5)
    assert answer == {'up': '-z'} and info['source'] == 'ai' and info['answered_by'] == 'client'
    assert info['model'] == 'client'
    record = seen[0]
    assert record['round'] == R and record['question'] == 'Which way is up?' and record['schema'] == SCHEMA
    assert '"up"' in record['instructions'] and all(p.endswith(('.png', '.jpg')) for p in record['pictures'])
    assert json.loads((relay / f"done-{record['id']}.json").read_text())['answered'] is True
    again, info = ai.ask('Which way is up?', [_picture(tmp_path)], SCHEMA, DEFAULT, cache_dir=tmp_path / 'cache',
                         route=R)
    assert again == {'up': '-z'} and info['source'] == 'cache'        # cached under the client, asked once


def test_a_round_asked_again_on_purpose_is_a_new_ask_not_the_cache(relay, tmp_path, monkeypatch):
    # Frieza, 2026-11-18: the tripwire's second try ('core fill again') came back from the first's cache over the
    # relay, keyed by the client alone, and the same wreck was built twice
    monkeypatch.setattr(ai, 'RELAY_WAIT_S', 10)
    seen, asked = [], []
    for r, answer in (('core fill', {'up': '-z'}), ('core fill again', {'up': '+z'})):
        thread = _answer_when_asked(relay, [answer], seen)
        asked.append(ai.ask('Which way is up?', [_picture(tmp_path)], SCHEMA, DEFAULT, cache_dir=tmp_path / 'cache',
                            route=r))
        thread.join(5)
    assert [a for a, _ in asked] == [{'up': '-z'}, {'up': '+z'}] and [i['source'] for _, i in asked] == ['ai', 'ai']
    assert [r['round'] for r in seen] == ['core fill', 'core fill again']


def test_an_answer_outside_the_schema_is_handed_back_and_asked_again(relay, tmp_path):
    seen = []
    thread = _answer_when_asked(relay, [{'up': 'sideways'}, {}, {'up': '-z'}], seen)
    answer, info = ai.ask('Which way is up?', [_picture(tmp_path)], SCHEMA, DEFAULT, route=R)
    thread.join(5)
    assert answer == {'up': '-z'} and info['source'] == 'ai' and len(seen) == 3
    assert not (relay / f"problem-{seen[0]['id']}.json").exists()      # cleared once an answer fit


def test_no_answer_in_time_is_the_default(relay, tmp_path, monkeypatch):
    monkeypatch.setattr(ai, 'RELAY_WAIT_S', 0.1)
    answer, info = ai.ask('Which way is up?', [_picture(tmp_path)], SCHEMA, DEFAULT, route=R)
    assert answer == DEFAULT and info['source'] == 'default' and 'no answer' in info['error']
    done = next(relay.glob('done-*.json'))
    assert json.loads(done.read_text())['answered'] is False


def test_without_the_relay_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(ai.RELAY_ENV, raising=False)
    assert ai.relay() is None


def test_relayed_answers_are_kept_with_the_import_and_others_in_the_shared_cache(tmp_path, monkeypatch):
    """A relayed answer is whichever assistant gave it, and its key can't say which: it is kept in the import's own
    folder, never the shared cache a second assistant's import of the same file would read."""
    monkeypatch.delenv(ai.RELAY_ENV, raising=False)
    assert ai.answers(tmp_path / 'shared', tmp_path / 'own') == tmp_path / 'shared'
    monkeypatch.setenv(ai.RELAY_ENV, str(tmp_path / 'asks'))
    assert ai.answers(tmp_path / 'shared', tmp_path / 'own') == tmp_path / 'own'
