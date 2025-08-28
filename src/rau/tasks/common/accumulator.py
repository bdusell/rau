class MicroAveragedScoreAccumulator:

    def __init__(self) -> None:
        super().__init__()
        self.numerator = 0
        self.denominator = 0

    def update(self, numerator: float, denominator: float) -> None:
        self.numerator += numerator
        self.denominator += denominator

    def get_value(self) -> float:
        return self.numerator / self.denominator

class DictScoreAccumulator:

    def __init__(self) -> None:
        super().__init__()
        self.loss = None

    def update(self, scores: dict[str, tuple[float, float]]) -> None:
        if self.loss is None:
            self.loss = { k : MicroAveragedScoreAccumulator() for k in scores.keys() }
        elif scores.keys() != self.loss.keys():
            raise ValueError
        for key, (numerator, denominator) in scores.items():
            self.loss[key].update(numerator, denominator)

    def get_value(self) -> dict[str, float]:
        return { k : v.get_value() for k, v in self.loss.items() }
