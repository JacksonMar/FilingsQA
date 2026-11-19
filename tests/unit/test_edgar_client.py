import httpx
import pytest

from filingsqa.edgar.client import EdgarClient, EdgarError
from filingsqa.edgar.rate_limit import RateLimiter

UA = "FilingsQA test@example.com"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responses, sleeps=None):
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = EdgarClient(
        UA,
        max_retries=2,
        backoff_seconds=1.0,
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )
    return client, seen


def test_sends_identifying_user_agent():
    client, seen = make_client([httpx.Response(200, json={"ok": True})])

    assert client.get_json("https://data.sec.gov/x.json") == {"ok": True}
    assert seen[0].headers["User-Agent"] == UA


def test_user_agent_without_email_is_rejected():
    with pytest.raises(ValueError, match="email"):
        EdgarClient("FilingsQA")


def test_retries_on_429_with_exponential_backoff():
    sleeps: list[float] = []
    client, seen = make_client(
        [httpx.Response(429), httpx.Response(503), httpx.Response(200, text="ok")], sleeps
    )

    assert client.get_text("https://www.sec.gov/a.htm") == "ok"
    assert len(seen) == 3
    backoffs = [s for s in sleeps if s >= 1.0]
    assert backoffs == [1.0, 2.0]


def test_retries_on_transport_error():
    client, _ = make_client([httpx.ConnectError("boom"), httpx.Response(200, text="ok")])

    assert client.get_text("https://www.sec.gov/a.htm") == "ok"


def test_gives_up_after_max_retries():
    client, seen = make_client([httpx.Response(503)] * 3)

    with pytest.raises(EdgarError, match="after 3 attempts"):
        client.get("https://www.sec.gov/a.htm")
    assert len(seen) == 3


def test_404_is_not_retried():
    client, seen = make_client([httpx.Response(404)])

    with pytest.raises(EdgarError, match="404"):
        client.get("https://www.sec.gov/missing.htm")
    assert len(seen) == 1


def test_rate_limiter_spaces_calls_at_max_rate():
    clock = FakeClock()
    limiter = RateLimiter(10, clock=clock, sleep=clock.sleep)

    for _ in range(5):
        limiter.wait()

    assert clock.sleeps == pytest.approx([0.1] * 4)


def test_rate_limiter_does_not_sleep_when_calls_are_slow():
    clock = FakeClock()
    limiter = RateLimiter(10, clock=clock, sleep=clock.sleep)

    limiter.wait()
    clock.now += 1.0
    limiter.wait()

    assert clock.sleeps == []
