class Policy:
    def __init__(self, func):
        self.params = dict()
        self._func = func   # policy's functional form taking two arguments: params, state; returns action

    def act(self, state):
        return self._func(self.params, state)

# example of simple policy y=kx+b
func = lambda params, x: params["k"]*x + params["b"]
p = Policy(func)
p.params["k"] = 5
p.params["b"] = 2
print(p.act(state=3)) # 5*3+2=17