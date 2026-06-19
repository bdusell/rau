class MicroAveragedScoreAccumulator:

    _numerator: float
    _denominator: float

    def __init__(self) -> None:
        super().__init__()
        self._numerator = 0
        self._denominator = 0

    def update(self, numerator: float, denominator: float) -> None:
        self._numerator += numerator
        self._denominator += denominator

    def get_value(self) -> float:
        return self._numerator / self._denominator

class DictScoreAccumulator:

    _loss: None | dict[str, MicroAveragedScoreAccumulator]

    def __init__(self) -> None:
        super().__init__()
        self._loss = None

    def update(self, scores: dict[str, tuple[float, float]]) -> None:
        if self._loss is None:
            self._loss = { k : MicroAveragedScoreAccumulator() for k in scores.keys() }
        elif scores.keys() != self._loss.keys():
            raise ValueError(f'keys do not match')
        for key, (numerator, denominator) in scores.items():
            self._loss[key].update(numerator, denominator)

    def get_value(self) -> dict[str, float]:
        if self._loss is not None:
            return { k : v.get_value() for k, v in self._loss.items() }
        else:
            raise ValueError(f'have not seen any scores yet')
