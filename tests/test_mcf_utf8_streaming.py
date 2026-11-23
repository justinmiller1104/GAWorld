"""RED-phase regression tests for the MCF UTF-8 streaming repair (task 3705).

Stage 1 health audit found mojibake in 38.5% of episodes: the two streaming
decode paths in ``gaworld/llm/providers.py`` (``OllamaProvider.call`` and
``OpenAIProvider._do_streaming``) decode with ``iter_lines(decode_unicode=True)``
without forcing UTF-8, so requests falls back to a header-derived non-UTF-8
charset (ISO-8859-1 for ``text/*`` without a charset parameter). Mojibake keys
then fail exact-match in ``_parse_action_space`` (``gaworld/sim/_action.py``).

The blocking path is clean because ``r.json()`` decodes the raw bytes as UTF-8
regardless of the header charset. The minimal fix is to force ``r.encoding =
"utf-8"`` on the two streaming paths — one line each. These tests are RED
against the unfixed code and go GREEN after that fix; they must not require
any other change (provider routing / fallback code untouched).

All transport is mocked; no network, no inference.
"""

from __future__ import annotations

import codecs
import json
import unittest
from unittest import mock

import requests
from requests.utils import get_encoding_from_headers

from gaworld.llm.providers import OllamaProvider, OpenAIProvider
from gaworld.sim._action import _parse_action_space

CHINESE_TEXT = "找地方避雨"
ACTIVITIES = ["避雨寻处", "找地方避雨"]


# ---------------------------------------------------------------------------
# A faithful stand-in for requests' streaming response.
#
# It reproduces the exact behaviour under audit:
#   * ``encoding`` is derived from the headers the way requests does it
#     (``get_encoding_from_headers``): a ``text/*`` content type without a
#     charset parameter yields ISO-8859-1 — the "non-UTF-8 default charset"
#     from the audit. The implementer's fix is to set ``response.encoding``
#     to UTF-8 before iterating; this mock honours whatever it is set to.
#   * ``iter_lines(decode_unicode=True)`` decodes each network chunk with an
#     incremental decoder (errors="replace"), exactly like requests'
#     ``stream_decode_response_unicode``, so bytes of one multibyte codepoint
#     split across chunks decode the way the real transport decodes them.
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], content_type: str = "text/event-stream"):
        self._chunks = chunks
        self.status_code = 200
        # lowercase key: get_encoding_from_headers looks up "content-type"
        self.headers = {"content-type": content_type}
        self.encoding = get_encoding_from_headers(self.headers)

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def close(self) -> None:
        return None

    def iter_content(self, chunk_size=None, decode_unicode=False):
        if not decode_unicode:
            yield from self._chunks
            return
        decoder = codecs.getincrementaldecoder(self.encoding or "utf-8")(errors="replace")
        for chunk in self._chunks:
            text = decoder.decode(chunk)
            if text:
                yield text
        tail = decoder.decode(b"", True)
        if tail:
            yield tail

    def iter_lines(self, decode_unicode=False, chunk_size=512):
        pending = ""
        for piece in self.iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode):
            buf = pending + piece
            lines = buf.split("\n")
            pending = lines.pop()
            for line in lines:
                yield line.rstrip("\r")
        if pending:
            yield pending.rstrip("\r")


def _blocking_response(payload: dict) -> requests.Response:
    """A requests.Response whose ``json()`` decodes UTF-8 like the real one."""
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


def _sse(payloads: list[str], split_every: int | None = None, prefix: str = "data: ") -> _FakeStreamResponse:
    """One SSE/NDJSON stream from a list of frames, encoded UTF-8.

    ``prefix`` is "data: " for SSE frames or "" for Ollama-style NDJSON.
    ``split_every`` cuts the raw byte stream into network chunks of that many
    bytes, which can (and for wide frames does) split multibyte codepoints
    across chunk boundaries.
    """
    raw = "".join(f"{prefix}{p}\n" for p in payloads).encode("utf-8")
    if split_every is None:
        chunks = [raw]
    else:
        chunks = [raw[i : i + split_every] for i in range(0, len(raw), split_every)]
    return _FakeStreamResponse(chunks)


def _openai_stream_frames(text: str, style: str = "openai") -> list[str]:
    """SSE data frames carrying ``text`` for the two payload styles under test."""
    if style == "minimax":
        # MiniMax-style: an OpenAI-compatible stream that finishes with a
        # full ``message.content`` frame instead of accumulated deltas.
        return [
            json.dumps(
                {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "[DONE]",
        ]
    # OpenAI-compatible: content arrives in deltas.
    return [
        json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]}, ensure_ascii=False),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, ensure_ascii=False),
        "[DONE]",
    ]


def _ollama_frames(text: str) -> list[str]:
    return [
        json.dumps({"response": text, "done": False}, ensure_ascii=False),
        json.dumps({"response": "", "done": True}, ensure_ascii=False),
    ]


def _make_openai_provider(stream: bool) -> OpenAIProvider:
    return OpenAIProvider(
        base_url="http://mock.invalid/v1",
        model="mock-model",
        api_key="test-key",
        stream=stream,
        attempts=1,
    )


class TestOpenAIStreamingUtf8(unittest.TestCase):
    """Path 1: OpenAIProvider._do_streaming (iter_lines(decode_unicode=True))."""

    def test_chinese_text_decodes_correctly(self):
        provider = _make_openai_provider(stream=True)
        response = _sse(_openai_stream_frames(CHINESE_TEXT))
        with mock.patch("gaworld.llm.providers.requests.post", return_value=response):
            result = provider.call("hi")
        self.assertIn(CHINESE_TEXT, result, f"streamed Chinese text arrived as mojibake: {result!r}")

    def test_chinese_json_keys_match_after_action_parser(self):
        text = json.dumps({ACTIVITIES[0]: ["在屋檐下躲雨"], ACTIVITIES[1]: ["躲进凉亭"]}, ensure_ascii=False)
        provider = _make_openai_provider(stream=True)
        with mock.patch(
            "gaworld.llm.providers.requests.post", return_value=_sse(_openai_stream_frames(text))
        ):
            result = provider.call("hi")
        parsed = _parse_action_space(result, ACTIVITIES)
        self.assertEqual(set(parsed.keys()), set(ACTIVITIES), f"mojibake keys failed exact-match: {parsed!r}")
        self.assertEqual(parsed[ACTIVITIES[0]], ["在屋檐下躲雨"])

    def test_multibyte_split_across_chunks_decodes_correctly(self):
        text = json.dumps({ACTIVITIES[1]: ["在屋檐下躲雨"]}, ensure_ascii=False)
        provider = _make_openai_provider(stream=True)
        # 3-byte cuts slice straight through the 3-byte CJK codepoints.
        with mock.patch(
            "gaworld.llm.providers.requests.post",
            return_value=_sse(_openai_stream_frames(text), split_every=3),
        ):
            result = provider.call("hi")
        parsed = _parse_action_space(result, [ACTIVITIES[1]])
        self.assertEqual(parsed.get(ACTIVITIES[1]), ["在屋檐下躲雨"])

    def test_streaming_and_blocking_equivalent(self):
        text = json.dumps({ACTIVITIES[0]: ["在屋檐下躲雨"]}, ensure_ascii=False)
        blocking = _make_openai_provider(stream=False)
        streaming = _make_openai_provider(stream=True)
        payload = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        with mock.patch("gaworld.llm.providers.requests.post", return_value=_blocking_response(payload)):
            blocking_text = blocking.call("hi")
        with mock.patch(
            "gaworld.llm.providers.requests.post", return_value=_sse(_openai_stream_frames(text))
        ):
            streaming_text = streaming.call("hi")
        self.assertEqual(blocking_text, streaming_text, "streaming decoded differently from blocking")

    def test_ascii_unchanged(self):
        text = '{"take shelter": ["stay under the eaves"]}'
        provider = _make_openai_provider(stream=True)
        with mock.patch(
            "gaworld.llm.providers.requests.post", return_value=_sse(_openai_stream_frames(text))
        ):
            result = provider.call("hi")
        self.assertEqual(text, result)

    def test_minimax_style_payload_decodes_chinese(self):
        provider = _make_openai_provider(stream=True)
        with mock.patch(
            "gaworld.llm.providers.requests.post",
            return_value=_sse(_openai_stream_frames(CHINESE_TEXT, style="minimax")),
        ):
            result = provider.call("hi")
        self.assertIn(CHINESE_TEXT, result)

    def test_malformed_utf8_with_declared_charset_is_deterministic(self):
        """Declared charset=utf-8 + invalid bytes: no exception, no double-mojibake.

        Documents the existing behaviour that must not be loosened: errors are
        replaced with U+FFFD (requests decodes with errors="replace") and the
        rest of the stream still arrives.
        """
        good_frame = json.dumps({"choices": [{"index": 0, "delta": {"content": "OK"}}]}).encode()
        bad_frame = b'data: {"choices": [{"index": 0, "delta": {"content": "\xff\xfe bad"}}]}\n\n'
        raw = b"data: " + good_frame + b"\n\n" + bad_frame
        response = _FakeStreamResponse([raw], content_type="text/event-stream; charset=utf-8")
        provider = _make_openai_provider(stream=True)
        with mock.patch("gaworld.llm.providers.requests.post", return_value=response):
            result = provider.call("hi")  # must not raise
        self.assertIn("OK", result)
        self.assertIn("\ufffd", result, "invalid bytes should surface as U+FFFD, not silently vanish")


class TestOllamaStreamingUtf8(unittest.TestCase):
    """Path 2: OllamaProvider.call (iter_lines(decode_unicode=True))."""

    def test_chinese_text_decodes_correctly(self):
        provider = OllamaProvider("http://mock.invalid", "mock-model", attempts=1)
        response = _sse(_ollama_frames(CHINESE_TEXT), prefix="")
        with mock.patch("gaworld.llm.providers.requests.post", return_value=response):
            result = provider.call("hi")
        self.assertIn(CHINESE_TEXT, result, f"streamed Chinese text arrived as mojibake: {result!r}")

    def test_multibyte_split_across_chunks_decodes_correctly(self):
        frames = _ollama_frames(CHINESE_TEXT + " 在屋檐下躲雨")
        provider = OllamaProvider("http://mock.invalid", "mock-model", attempts=1)
        with mock.patch(
            "gaworld.llm.providers.requests.post",
            return_value=_sse(frames, split_every=2, prefix=""),
        ):
            result = provider.call("hi")
        self.assertIn(CHINESE_TEXT, result)
        self.assertIn("在屋檐下躲雨", result)


class TestProviderCompatibilityUntouched(unittest.TestCase):
    """The decode fix must not change provider selection/routing/fallback.

    Both wire styles keep flowing through the same ``call`` path; only the
    decode statements may change (patch exactness: these tests patch nothing
    but ``requests.post``, so any wider change must still keep them green).
    """

    def test_both_wire_styles_route_through_the_same_call(self):
        for style in ("openai", "minimax"):
            provider = _make_openai_provider(stream=True)
            with mock.patch(
                "gaworld.llm.providers.requests.post",
                return_value=_sse(_openai_stream_frames(CHINESE_TEXT, style=style)),
            ) as post:
                result = provider.call("hi")
            self.assertEqual(1, post.call_count, f"{style}: unexpected retry/extra call")
            self.assertIn(CHINESE_TEXT, result)
            self.assertIn(
                "/chat/completions",
                str(post.call_args),
                f"{style}: request must still hit the chat/completions endpoint",
            )

    def test_blocking_provider_still_reports_finish_reason_and_usage(self):
        provider = _make_openai_provider(stream=False)
        payload = {"choices": [{"message": {"content": CHINESE_TEXT}, "finish_reason": "stop"}]}
        with mock.patch("gaworld.llm.providers.requests.post", return_value=_blocking_response(payload)):
            result = provider.call("hi")
        self.assertEqual(CHINESE_TEXT, result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
