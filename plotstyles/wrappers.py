import types

class RunChecker:
    """
    Perform a check before running a function.
    """
    def __init__(self, condition=None):
        self.condition = condition

    def __checkcondition(self, condition=None):
        ## condition is a func
        if condition:
            return condition(*self.args, **self.kwargs)
        else:
            return True

    def __call__(self):
        if self.__checkcondition(*self.args, **self.kwargs):
            return self.func(*self.args, **self.kwargs)
        else:
            print("The condition is not met and the function is not executed.")

def add(x, y):
    return x + y

a = RunChecker(add, 3, 4)
print(a)
    