import json

from Pyesian.optimizers.hyperparameters.HyperparameterOptimizer import HyperparameterOptimizer
from itertools import product

from Pyesian.optimizers.hyperparameters.space import Number, Real, Integer


class GridOptimizer(HyperparameterOptimizer):
    """A grid optimizer inheriting from HyperparametersOptimizer. It performs a grid search over the given hyperparameter \
    possibilities.

    compile should be called with additional argument n.
    Optimizer will test n combinations along each axis.

    Example :
    grid_optimizer.compile(Real(0,1,"lr"), Integer(0,100, "k"), n = 10)

    To specify the values to search on one axis use the specify argument. key is the name of the parameters and the
    value should be a list on all the possible values along the axis.
    Example :
    grid_optimizer.compile(Real(0,1,"lr"), Integer(0,100, "k"),specify={"lr":[1e-3,1e-2,1e-1]}, n = 10)

    """
    def __init__(self):
        super().__init__()
        self._axes = []
        self._names = []
        self._results = {}

    def _compile_extra_components(self, *args, **kwargs):
        self._n = kwargs["n"]
        specify = {}
        if "specify" in kwargs:
            specify = kwargs["specify"]
        for arg in args:
            if isinstance(arg, Number):
                n = self._n
                self._names.append(arg.name)
                if arg.name in specify:
                    n = specify[arg.name]
                if isinstance(n, list):
                    self._axes.append(n)
                else:
                    if n < 2:
                        raise ValueError("n can't be less than 2 for a grid search, do a constant parameter instead")
                    if isinstance(arg, Real):
                        epsilon = (arg.upper_bound - arg.lower_bound) / (n - 1)
                        self._axes.append([i * epsilon + arg.lower_bound for i in range(n)])
                    elif isinstance(arg, Integer):
                        size = arg.upper_bound - arg.lower_bound + 1
                        if n >= size:
                            self._axes.append([i + arg.lower_bound for i in range(size)])
                        else:
                            epsilon = (arg.upper_bound - arg.lower_bound) / (n - 1)
                            points = [int(round(i * epsilon + arg.lower_bound)) for i in range(n)]
                            self._axes.append(sorted(list(set(points))))

    def optimize(self):
        """Performs the grid search on the specified hyperparameters in compile.
        """
        self._results = {}

        omega = list(product(*self._axes))
        n_omega = len(omega)
        # print("Starting GridOptimizer ", n_omega, " possibilities will be evaluated on ", nb_processes, " cores!")

        # def display_progress():
        #    cont = True
        #    while cont:
        #        lock.acquire()
        #        n_res = len(self._results)
        #        lock.release()
        #        self._print_progress(n_res / n_omega, bar_length=20, suffix="Grid Optimizer",
        #                            completed=str(n_res) + "/" + str(n_omega))
        #       if n_res >= n_omega:
        #            cont = False
        #        time.sleep(1)
        #    print()

        # with multiprocessing.Pool(processes=nb_processes) as pool:
        #    pool.map(self.__task, omega)
        for w in omega:
            result = self._f(*w)
            # lock.acquire()
            self._results[w] = result
            # lock.release()
            n_res = len(self._results)
            self._print_progress(n_res / n_omega, bar_length=20, suffix="Grid Optimizer",
                                 completed=str(n_res) + "/" + str(n_omega))

    def save(self, path: str):
        """Saves the results of the hyperparameter optimization.

        Args:
            path (str): The path to the file in ehich the results will be saved
        """
        final_res = {"args_name": self._names, "results": self._results}
        with open(path, "w") as file:
            file.write(",".join(self._names)+"\n")
            for params, result in self._results.items():
                file.write(",".join([str(p) for p in params])+"\n")
                file.write(str(result)+"\n")
