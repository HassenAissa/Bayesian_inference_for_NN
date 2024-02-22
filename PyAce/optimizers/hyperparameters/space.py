class Parameter:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class Number(Parameter):
    def __init__(self, lower_bound: float, upper_bound: float, name: str):
        super().__init__(name)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound


class Real(Number):
    def __init__(self, lower_bound: float, upper_bound: float, name: str):
        super().__init__(lower_bound, upper_bound, name)


class Integer(Number):
    def __init__(self, lower_bound: int, upper_bound: int, name: str):
        super().__init__(lower_bound, upper_bound, name)


class Constant(Parameter):
    def __init__(self, value, name: str):
        super().__init__(name)
        self._value = value
