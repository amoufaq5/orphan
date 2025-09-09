from __future__ import annotations
import time, threading

class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = float(rate_per_sec)
        self.capacity = int(burst)
        self.tokens = float(burst)
        self.lock = threading.Lock()
        self.last = time.perf_counter()

    def take(self, tokens: float = 1.0):
        while True:
            with self.lock:
                now = time.perf_counter()
                elapsed = now - self.last
                self.last = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            time.sleep(0.001)
