import z3

def example():
    x = z3.Real('x')
    y = z3.Real('y')
    s = z3.Solver()
    s.set(unsat_core=True)
    s.assert_and_track(x + y > 5, "first")
    s.assert_and_track(x > 5, "second")
    s.assert_and_track(x < 5, "third")
    s.assert_and_track(y > 1, "fourth")
    result = s.check()
    if result==z3.sat:
        print(s.model())
    else:
        print(z3.unsat)
        print(s.unsat_core())

example()