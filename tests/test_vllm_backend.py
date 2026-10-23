"""Local HTTP contract tests: exact IDs, teacher-forcing alignment and concurrent rollouts."""

import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from prompt_policy.backends import VLLMBackend, generate_batch
from prompt_policy.config import validate_config


@pytest.fixture
def api_backend(config, recorder):
    state = SimpleNamespace(requests=[], active=0, maximum=0, mutate=lambda choice: None)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state.requests.append(body)
                state.active += 1
                state.maximum = max(state.maximum, state.active)
            time.sleep(0.015)
            prefix = body["prompt"]
            choice = {
                "index": 0,
                "text": "unused server rendering",
                "finish_reason": "stop",
                "prompt_token_ids": prefix,
                "token_ids": [prefix[-1], 99],
                "prompt_logprobs": [None]
                + [{str(token): {"logprob": -position / 10}} for position, token in enumerate(prefix[1:], 1)],
            }
            state.mutate(choice)
            raw = json.dumps({"choices": [choice]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            with lock:
                state.active -= 1

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = VLLMBackend.__new__(VLLMBackend)
    backend.config, backend.recorder = copy.deepcopy(config), recorder
    backend.config["generation"].update(temperature=0.6, top_p=0.95, top_k=20)
    backend.serving = {
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "model": "pinned",
        "timeout_seconds": 5,
        "max_concurrency": 3,
    }
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **_: ",".join(map(str, ids)))
    backend.eos_ids = {99}
    backend.prompt_ids = lambda prompt, question: (10, int(question))
    try:
        yield backend, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_generation_preserves_original_ids_and_sampling(api_backend):
    backend, state = api_backend
    generated = backend.generate("P", "42")
    assert generated.response_ids == (42, 99)
    assert generated.prompt_ids == (10, 42)
    assert generated.text == "42"  # decode actual IDs, never retokenize server text
    assert generated.finish_reason == "eos"
    body = state.requests[0]
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.6, 0.95, 20)
    assert body["return_token_ids"] and not body["skip_special_tokens"]
    assert "seed" not in body and "stop" not in body


def test_prompt_logprobs_score_exact_fixed_tokens_at_causal_positions(api_backend):
    backend, state = api_backend
    assert backend.logprobs((10, 11), (50, 99)) == (-0.2, -0.3)
    request = state.requests[0]
    assert request["prompt"] == [10, 11, 50, 99]
    assert request["prompt_logprobs"] == 0
    assert (request["temperature"], request["top_p"], request["top_k"]) == (1.0, 1.0, -1)
    assert request["max_tokens"] == 1  # discarded dummy, never part of scored response


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda c: c.update(prompt_token_ids=[12]), "changed"),
        (lambda c: c.update(token_ids=None), "actual generated token IDs"),
        (lambda c: c.update(token_ids=[42]), "EOS"),
    ],
)
def test_http_rejects_changed_context_or_missing_exact_ids(api_backend, mutation, error):
    backend, state = api_backend
    state.mutate = mutation
    with pytest.raises(ValueError, match=error):
        backend.generate("P", "42")


def test_http_rejects_unaligned_prompt_logprobs(api_backend):
    backend, state = api_backend
    state.mutate = lambda c: c.update(prompt_logprobs=[None])
    with pytest.raises(ValueError, match="every supplied token position"):
        backend.logprobs((10,), (42, 99))


def test_concurrent_requests_preserve_order_and_accounting(api_backend):
    backend, state = api_backend
    outputs = generate_batch(backend, "P", ["42", "43", "44", "45", "46", "47"])
    assert [g.response_ids[0] for g in outputs] == list(range(42, 48))
    assert state.maximum >= 2
    assert backend.recorder.cost["task_calls"] == 6
    assert backend.recorder.cost["task_response_tokens"] == 12


def test_context_overflow_is_rejected_without_truncation(api_backend):
    backend, state = api_backend
    backend.config["model"]["max_context_tokens"] = 2
    with pytest.raises(ValueError, match="truncation"):
        backend.generate("P", "42")
    assert not state.requests


def test_serving_config_is_explicit_and_local(config):
    config["serving"] = {
        "base_url": "http://127.0.0.1:8017/v1",
        "model": "pinned",
        "max_concurrency": 3,
        "timeout_seconds": 1800,
        "server_manifest": "/tmp/server.json",
    }
    validate_config(config)
    config["serving"]["base_url"] = "https://external.example/v1"
    with pytest.raises(ValueError, match="local http"):
        validate_config(config)


def test_top1_probability_uses_vocabulary_alternative_at_exact_response_position(api_backend):
    backend, state = api_backend

    def alternatives(choice):
        for row in choice["prompt_logprobs"][1:]:
            row["777"] = {"logprob": -0.01, "rank": 1}

    state.mutate = alternatives
    actual, maximum = backend.logprob_statistics((10, 11), (50, 99))
    assert actual == (-0.2, -0.3) and maximum == (-0.01, -0.01)
    assert state.requests[0]["prompt_logprobs"] == 1


def test_replica_distribution_and_streaming_preserve_job_identity(api_backend):
    backend, state = api_backend
    assigned, streamed = [], []

    def peer_generate(prompt, question):
        assigned.append(question)
        return backend.generate(prompt, question)

    backend.replicas = [SimpleNamespace(generate=peer_generate)]
    outputs = generate_batch(
        backend, "P", ["42", "43", "44", "45"], on_result=lambda i, g: streamed.append((i, g.response_ids[0]))
    )
    assert sorted(assigned) == ["43", "45"]
    assert sorted(streamed) == [(0, 42), (1, 43), (2, 44), (3, 45)]
    assert [g.response_ids[0] for g in outputs] == [42, 43, 44, 45]


def test_http_reflection_renders_single_user_without_an_empty_system(api_backend):
    backend, state = api_backend
    rendered = []

    def apply_template(messages, **kwargs):
        rendered.append(messages)
        assert kwargs["add_generation_prompt"]
        return [10, 42]

    backend.tokenizer.apply_chat_template = apply_template
    output = backend.generate("", "editor instructions\n\nfixed trajectory evidence", reflection=True)
    assert rendered == [[{"role": "user", "content": "editor instructions\n\nfixed trajectory evidence"}]]
    assert output.prompt_ids == (10, 42)
    assert state.requests[0]["prompt"] == [10, 42]
    assert state.requests[0]["max_tokens"] == backend.config["reflection"]["max_new_tokens"]
    assert backend.recorder.cost["reflection_calls"] == 1
    assert backend.recorder.cost["task_calls"] == 0


def test_rollout_logprobs_preserve_sampled_ids_eos_and_policy(api_backend):
    backend, state = api_backend
    backend.serving['rollout_logprobs'] = 'processed_logprobs'

    def sampled_logprobs(choice):
        choice['logprobs'] = {
            'tokens': [f'token_id:{i}' for i in choice['token_ids']],
            'token_logprobs': [-0.125, 0.0],
        }

    state.mutate = sampled_logprobs
    result = backend.generate('P', '42')
    assert state.requests[0]['logprobs'] == 0
    assert state.requests[0]['return_tokens_as_token_ids']
    assert state.requests[0]['min_p'] == 0.0
    assert result.response_ids == (42, 99)
    assert result.rollout_token_logprobs == (-0.125, 0.0)
    assert result.rollout_logprobs_mode == 'processed_logprobs'
    assert result.rollout_sampling['temperature'] == 0.6
    assert result.rollout_sampling['top_k'] == 20
    assert result.rollout_sampling['top_p'] == 0.95


@pytest.mark.parametrize('values,tokens', [
    ([None, 0.0], ['token_id:42', 'token_id:99']),
    ([-0.1], ['token_id:42']),
    ([-0.1, 0.0], ['token_id:43', 'token_id:99']),
    ([float('nan'), 0.0], ['token_id:42', 'token_id:99']),
    ([0.1, 0.0], ['token_id:42', 'token_id:99']),
    ([-9999.0, 0.0], ['token_id:42', 'token_id:99']),
])
def test_rollout_logprobs_reject_missing_misaligned_and_clipped_values(api_backend, values, tokens):
    backend, state = api_backend
    backend.serving['rollout_logprobs'] = 'processed_logprobs'
    state.mutate = lambda choice: choice.update(logprobs={'token_logprobs': values, 'tokens': tokens})
    with pytest.raises(ValueError, match='logprob'):
        backend.generate('P', '42')


def test_logprob_mode_must_match_recorded_server_startup():
    from prompt_policy.backends import validate_rollout_logprobs_mode

    assert validate_rollout_logprobs_mode({'command': ['vllm', 'serve', 'model']}, 'raw_logprobs') == 'raw_logprobs'
    for flag in [['--logprobs-mode', 'processed_logprobs'], ['--logprobs-mode=processed_logprobs']]:
        manifest = {'command': ['vllm', 'serve', 'model', *flag]}
        assert validate_rollout_logprobs_mode(manifest, 'processed_logprobs') == 'processed_logprobs'
        with pytest.raises(ValueError, match='differs'):
            validate_rollout_logprobs_mode(manifest, 'raw_logprobs')
    with pytest.raises(ValueError, match='differs'):
        validate_rollout_logprobs_mode({'command': ['vllm', 'serve', 'model']}, 'processed_logprobs')
    with pytest.raises(ValueError, match='startup command'):
        validate_rollout_logprobs_mode({}, 'processed_logprobs')


def test_rollout_logprob_capture_does_not_change_teacher_forcing(api_backend):
    backend, state = api_backend
    backend.serving['rollout_logprobs'] = 'processed_logprobs'
    assert backend.logprobs((10, 11), (50, 99)) == (-0.2, -0.3)
    assert state.requests[0]['prompt_logprobs'] == 0
    assert 'logprobs' not in state.requests[0]


def test_rollout_logprobs_config_keeps_meaning_explicit(config):
    config['serving'] = {
        'base_url': 'http://127.0.0.1:8017/v1', 'model': 'pinned',
        'max_concurrency': 3, 'timeout_seconds': 1800, 'server_manifest': '/tmp/server.json',
        'rollout_logprobs': 'processed_logprobs',
    }
    config['generation']['temperature'] = 0.6
    validate_config(config)
    config['serving']['rollout_logprobs'] = 'processed_logits'
    with pytest.raises(ValueError, match='rollout_logprobs'):
        validate_config(config)
    config['serving']['rollout_logprobs'] = 'processed_logprobs'
    config['generation']['temperature'] = 0
    with pytest.raises(ValueError, match='stochastic'):
        validate_config(config)
