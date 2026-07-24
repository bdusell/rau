import datetime
import functools
import json
from collections.abc import Iterable
from typing import Any, IO

class Logger:

    def log(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

class NullLogger(Logger):

    def log(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        pass

class FileLogger(Logger):

    _file: IO
    _flush: bool
    _reopen: bool

    def __init__(self, file: IO, flush: bool = False, reopen: bool = False) -> None:
        super().__init__()
        self._file = file
        self._flush = flush
        self._reopen = reopen

    def log(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        self.log_event(LogEvent(event_type, get_current_time(), data, None))

    def log_event(self, event: LogEvent) -> None:
        self._file.write(event.type)
        self._file.write(' ')
        self._file.write(str(event.timestamp.timestamp()))
        if event.has_data():
            self._file.write(' ')
            event.write_data(self._file)
        self._file.write('\n')
        if self._reopen:
            self._file.close()
            self._file = open(self._file.name, 'a')
        elif self._flush:
            self._file.flush()

def read_log_file(file: Iterable[str]) -> Iterable[LogEvent]:
    return map(parse_log_line, file)

class LogParseError(ValueError):
    pass

def parse_log_line(line: str) -> LogEvent:
    try:
        fields = line.split(' ', 2)
        data_str: str | None
        try:
            event_type, timestamp_str = fields
        except ValueError:
            event_type, timestamp_str, data_str = fields
        else:
            data_str = None
        timestamp = parse_timestamp(float(timestamp_str))
    except ValueError:
        raise LogParseError(f'cannot parse log line {line!r}')
    return LogEvent(event_type, timestamp, None, data_str)

class LogEvent:

    type: str
    timestamp: datetime.datetime
    _data: dict[str, Any] | None
    _data_str: str | None

    def __init__(self,
        type: str,
        timestamp: datetime.datetime,
        data: dict[str, Any] | None,
        data_str: str | None
    ) -> None:
        self.type = type
        self.timestamp = timestamp
        self._data = data
        self._data_str = data_str

    @functools.cached_property
    def data(self) -> dict[str, Any] | None:
        if self._data is not None:
            return self._data
        elif self._data_str is not None:
            return json.loads(self._data_str)
        else:
            return None

    def has_data(self) -> bool:
        return self._data is not None or self._data_str is not None

    def write_data(self, fout: IO) -> None:
        if self._data_str is not None:
            fout.write(self._data_str)
        elif self._data is not None:
            json.dump(self._data, fout, separators=(',', ':'), sort_keys=True)

def get_current_time() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

def parse_timestamp(timestamp: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
