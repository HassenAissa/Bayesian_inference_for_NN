class Parameter:
    """
    Represent a hyperparameter.
    Used to specify a hyperparameter in the search space in Hyperparameter Optimizer.
    """
    def __init__(self, name: str):
        self._name = name




    """
    Name of the hyperparameter to optimize (example : 'lr') 
    """
    @property
    def name(self) -> str:
        return self._name


class Number(Parameter):
    """
    Represent a hyperparameter that is a number.
    Used to specify a hyperparameter in the search space in Hyperparameter Optimizer.
    """
    def __init__(self, lower_bound: float, upper_bound: float, name: str):
        super().__init__(name)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    """
    Minimum value of the hyperparameter in the search space. 
    """
    @property
    def lower_bound(self):
        return self._lower_bound

    """
    Maximum value of the hyperparameter in the search space 
    """
    @property
    def upper_bound(self):
        return self._upper_bound


class Real(Number):
    """
    Represent a hyperparameter that is a real.
    Used to specify a hyperparameter in the search space in Hyperparameter Optimizer.
    """
    def __init__(self, lower_bound: float, upper_bound: float, name: str):
        super().__init__(lower_bound, upper_bound, name)


class Integer(Number):
    """
    Represent a hyperparameter that is an integer.
    Used to specify a hyperparameter in the search space in Hyperparameter Optimizer.
    """
    def __init__(self, lower_bound: int, upper_bound: int, name: str):
        super().__init__(lower_bound, upper_bound, name)


class Constant(Parameter):
    """
    Represent a constant hyperparameter that won't be changed by any optimizer.
    """
    def __init__(self, value, name: str):
        super().__init__(name)
        self._value = value
