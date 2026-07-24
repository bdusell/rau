import time

class Ticker:

    progress: float
    total: float

    def __init__(self, total: float) -> None:
        super().__init__()
        self.progress = 0
        self.total = total

    def tick(self) -> bool:
        raise NotImplementedError

    @property
    def fraction(self) -> float:
        return self.progress / self.total

    @property
    def percent(self) -> float:
        return 100.0 * self.fraction

    @property
    def int_percent(self) -> int:
        return int(100 * self.progress / self.total)

class OnChangeTicker(Ticker):

    _prev_tick: float

    def __init__(self, total: float) -> None:
        super().__init__(total)
        self._prev_tick = 0

    def tick(self) -> bool:
        new_tick = self.new_tick()
        result = new_tick > self._prev_tick
        self._prev_tick = new_tick
        return result

    def new_tick(self) -> float:
        raise NotImplementedError

class DividedTicker(OnChangeTicker):

    def __init__(self, total, ticks):
        super().__init__(total)
        self.ticks = ticks

    def new_tick(self):
        return self.progress * self.ticks // self.total

class TimedTicker(OnChangeTicker):

    _seconds: float
    _start_time: float

    def __init__(self, total: float, seconds: float) -> None:
        super().__init__(total)
        self._seconds = seconds
        self._start_time = time.time()

    def new_tick(self):
        return int((time.time() - self._start_time) / self._seconds)
