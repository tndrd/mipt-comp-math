import numpy as np
import collections
from nonlinear import newton, _jacob

CALC_TYPE = np.float64

class ButcherTable:
    def __init__(self, A, b, c):
        self.validate(A, b, c)
        
        self.A = A.astype(CALC_TYPE)
        self.b = b.astype(CALC_TYPE)
        self.c = c.astype(CALC_TYPE)

    def validate(self, A, b, c):
        assert len(b.shape) == 1
        assert len(c.shape) == 1
        assert len(A.shape) == 2
        assert (A.shape[0] == b.shape[0])
        assert (c.shape[0] == b.shape[0])

    def get_s(self):
        return len(self.b)

class RKSolver:
    def __init__(self, btable, h):
        assert RKSolver.method_is_supported(btable)

        self.btable = btable
        self.h = float(h)
        self.nsteps = 0

        self.t = None
        self.y = None
        self.k = None

    # Check if method is supported.
    # To be supported, method has to be
    # explicit or diagonally implicit
    @staticmethod
    def method_is_supported(btable):
        s = btable.get_s()
        for i in range(s):
            for j in range(i + 1, s):
                if btable.A[i][j] != 0:
                    return False
        return True


    def init_problem(self, func, t0, y0):
        self.func = func
        self.t = float(t0)
        self.y = y0.astype(CALC_TYPE)

        self.k = np.zeros((self.btable.get_s(), len(y0))).astype(CALC_TYPE)

    def step(self):
        ret = self.t, self.y.copy()

        t = self.t
        y = self.y
        f = self.func
        h = self.h

        c = self.btable.c
        a = self.btable.A
        b = self.btable.b

        self.k[:] = 0
        for i in range(self.btable.get_s()):
            conv = np.zeros((len(y)))
            for j in range(i):
                conv += a[i][j] * self.k[j]
            
            y1 = y + h * conv
            t1 = t + c[i] * h

            f_impl = f(y1, t1)
            
            if a[i][i] == 0:
                self.k[i] = f_impl
            else:
                F = lambda k: f(y1 + h * a[i][i] * k, t1) - k
                self.k[i] = newton(F, None, f_impl, self.h)

        self.t += h
        self.y += h * np.dot(self.k.T, b)

        return ret
    
    def get_h(self): return self.h

# CROS Solver, modified to integrate
# non-autonomous systems
class CROSSolver:
    def __init__(self, h):
        self.h = float(h)
        self.func = None
        self.t = None
        self.y = None

    def init_problem(self, func, t0, y0):
        self.func = func
        self.t = float(t0)
        self.y = y0.astype(CALC_TYPE)

    def step(self):
        ret = self.t, self.y.copy()

        J = _jacob(lambda x: self.func(x, self.t), self.y, self.h)
        A = np.identity(len(self.y), dtype=CALC_TYPE) - self.h * J * (1 + 1j) / 2
        
        k = np.linalg.solve(A, self.func(self.y, self.t))
        self.y += self.h * np.real(k)
        self.t += self.h

        return ret

    def get_h(self): return self.h

# Rosenbrock-type 3rd order ODE solver
class Rbrk3Solver:
    class Params:
        a = 0.435866521508459
        
        p0 = a
        p1 = 0.4782408332745185
        p2 = 0.0858926452170225
        
        b10 = a
        b20 = a
        b21 = -2.116053335949811

    def __init__(self, h):
        self.h = float(h)
        self.func = None
        self.t = None
        self.y = None

    def init_problem(self, func, t0, y0):
        self.func = func
        self.t = float(t0)
        self.y = y0.astype(CALC_TYPE)

    def step(self):
        ret = self.t, self.y.copy()

        f = lambda x: self.func(x, self.t)

        J = _jacob(f, self.y, self.h)
        D = np.identity(len(self.y), dtype=CALC_TYPE) + self.Params.a * self.h * J
        Di = np.linalg.inv(D)

        k0 = self.h * Di @ f(self.y)
        k1 = self.h * Di @ f(self.y + self.Params.b10 * k0)
        k2 = self.h * Di @ f(self.y + self.Params.b20 * k0 + self.Params.b21 * k1)

        self.y += self.Params.p0 * k0 + self.Params.p1 * k1 + self.Params.p2 * k2
        self.t += self.h

        return ret
    
    def get_h(self): return self.h

# Base class for one-staged multi-step
# solvers. Encapsulates evaluating the start site
# and caching y, t, f data of previous steps
#
# TLDR: it may be optimized via implementing
# custom numpy-based queue instead of
# collections-based one. This would allow
# using numpy vectorized instructions and
# amortized O(1) queue pushing/popping.
#
# Optional params:
#   aux_solver_producer: functor that produces auxilary
#       one-step solver to evaluate first N points
#       default: Explicit RK4
#   start_subdivide: defines how the start site is subdivided into finer grid
#       for example, if N = 5, h = 0.2, start_subdivide = 2, then
#       auxilary solver will evalute 10 points with h = 0.1
class MultiStepSolverBase:
    def __init__(self, steps, h, aux_solver_producer=None, start_subdivide=1):
        self.h = h
        self.steps = steps
        self.func = None
        self._ytfcache = collections.deque()
        self.start_subdivide = int(start_subdivide)
        self.aux_solver = None

        if aux_solver_producer is None:
            self.aux_solver = MultiStepSolverBase.__create_aux_solver
        else:
            self.aux_solver = aux_solver_producer

    def init_problem(self, func, t0, y0):
        self.func = func
        self.__start_site(t0, y0)

    # Using Runge-Kutta Method to
    # evaluate first N points
    def __start_site(self, t0, y0):
        solver = self.aux_solver(self.h / self.start_subdivide)
        solver.init_problem(self.func, t0, y0)

        for i in range(self.steps * self.start_subdivide):
            t, y = solver.step()
            if (i % self.start_subdivide == 0):
                self._ytfcache.append((y, t, self.func(y, t)))
    

    def __ytf_cache_at(self, i):
        return self._ytfcache[i][0].copy(), self._ytfcache[i][1], self._ytfcache[i][2].copy()

    def _ytf_cache_tail(self):
        return self.__ytf_cache_at(0)
    
    def _ytf_cache_head(self):
        return self.__ytf_cache_at(-1)

    def _ytf_cache_update(self, y, t, f):
        self._ytfcache.append((y, t, f))
        self._ytfcache.popleft()

    @staticmethod
    def __create_aux_solver(h):
        return RungeKuttaCollection.create_e4(h)

class AdamsSolver(MultiStepSolverBase):
    # coeffs - Adams method coefficients, left to right
    # y_{n + 1} = y_n + c_implicit*h*f_{n+1} + h(c[-1]*f_n + c[-2]*f_{n-1} + ...)
    def __init__(self, coeffs, h, c_implicit=None, aux=None, sub=1):
        super().__init__(len(coeffs), h, aux, sub)
        self.coeffs = np.array(coeffs).astype(CALC_TYPE)
        self.c_implicit = c_implicit
    
    def step(self):
        ret = self._ytf_cache_tail() # oldest y, t, f
        y, t, _  = self._ytf_cache_head() # yn, tn, fn
        yn = y.copy()    

        conv = np.zeros_like(y)
        for c, (_, __, f) in zip(self.coeffs, self._ytfcache):
            conv += c * f

        y += self.h * conv
        t += self.h

        # At this point
        #   y = y_n + h(c[-1]*f_n + c[-2]*f_{n-1} + ...)
        # which is the y_{n+1} for explicit method.
        # If method is implicit, we need to solve
        #   y_{n+1} = y + h*c_implicit*f(y_{n+1}, t_{n+1})
        # where
        #   f(y_{n+1}, t_{n+1}) ~ f(y_n, t_n + h) + o(1)

        if (self.c_implicit is not None):
            func = lambda yn1: y - yn1 + self.h*self.c_implicit*self.func(yn1, t)
            appr = y + self.h * self.c_implicit * self.func(yn, t)
            y = newton(func, None, appr, self.h)

        self._ytf_cache_update(y, t, self.func(y, t))
        
        return ret[1], ret[0]
    
    def get_h(self): return self.h

class BDFSolver(MultiStepSolverBase):
    # Params define method according to "implicit" flag:
    # implicit is False:
    #   y_{n+1} = A*h*f(y_n, t_n) + c_0*y_n + c_1*y_{n-1} + c_2*y_{n-2} + ...
    # implicit is True:
    #   y_{n+1} = A*h*f(y_{n+1}, t_{n+1}) + c_0*y_n + c_1*y_{n-1} + c_2*y_{n-2} + ...
    def __init__(self, A, coeffs, h, implicit=False, aux=None, sub=1):
        super().__init__(len(coeffs), h, aux, sub)
        self.A = A
        self.coeffs = np.array(coeffs[::-1]).astype(CALC_TYPE)
        self.implicit = implicit

    def step(self):
        ret = self._ytf_cache_tail()
        yn, t, f = self._ytf_cache_head()
        t += self.h

        conv = np.zeros_like(f)
        for c, (y, _, __) in zip(self.coeffs, self._ytfcache):
            conv += c * y

        y = conv
        if self.implicit == False:
            y += self.A * self.h * f
        else:
            # At this point,
            #   y = c_0*y_n + c_1*y_{n-1} + c_2*y_{n-2} + ...
            # To evaluate y_{n+1}, we need to solve
            #   y_{n+1} = A * h * f(y_{n+1}, t_{n+1}) + y
            # Which can be approximated with o(1) as
            #   y_{n+1} ~ A * h * f(y_n, t_{n+1}) + y + o(1)
            func = lambda yn1: y - yn1 + self.A * self.h * self.func(yn1, t)
            appr = y + self.A * self.h * self.func(yn, t)
            y = newton(func, None, appr, self.h)

        self._ytf_cache_update(y, t, self.func(y, t))
        
        return ret[1], ret[0]
    
    def get_h(self): return self.h

class RungeKuttaCollection:
    # Creates explicit 1-staged 1-order solver
    # Butcher table:
    #  0 |   |
    # --------
    #    | 1 | 
    @staticmethod
    def create_e1(h):
        c = np.array([0])
        b = np.array([1])
        A = np.array([[0]])

        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)
    
    # Creates explicit 2-staged 2-order solver
    # Butcher table:
    #   0  |     |     |
    #  2/3 | 2/3 |     |
    # ------------------
    #      | 1/4 | 3/4 |
    @staticmethod
    def create_e2(h):
        c = np.array([0, 2/3])
        b = np.array([.25, .75])
        A = np.array([[0, 0],
                      [2/3, 0]])
        
        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)
    
    # Creates explicit 3-staged 3-order solver
    # Butcher table:
    #   0  |     |     |     |
    #  1/2 | 1/2 |     |     |
    #   1  | -1  |  2  |     |
    # ------------------------
    #      | 1/6 | 2/3 | 1/6 |
    @staticmethod
    def create_e3(h):
        c = np.array([0, .5, 1])
        b = np.array([1/6, 2/3, 1/6])
        A = np.array([[0,  0, 0],
                      [.5, 0, 0],
                      [-1, 2, 0]])
        
        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)
    
    # Creates explicit 4-staged 4-order solver
    # Butcher table:
    #   0  |  0  |  0  |  0  |  0  |
    #  1/2 | 1/2 |  0  |  0  |  0  |
    #  1/2 |  0  | 1/2 |  0  |  0  |
    #   1  |  0  |  0  |  1  |  0  |
    # ------------------------------
    #      | 1/6 | 1/3 | 1/3 | 1/6 |
    @staticmethod
    def create_e4(h):
        c = np.array([0, 1/2, 1/2, 1])
        b = np.array([1/6, 1/3, 1/3, 1/6])
        A = np.array([[0,   0,   0, 0],
                      [1/2, 0,   0, 0],
                      [0,   1/2, 0, 0],
                      [0,   0,   1, 0]])
        
        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)
    
    # Creates implicit 1-staged 1-order solver
    # (Backward Euler method)
    # Butcher table:
    #  1 | 1 |
    # --------
    #    | 1 |
    @staticmethod
    def create_i1(h):
        c = np.array([1])
        b = np.array([1])
        A = np.array([[1]])
        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)
    
    # Creates implicit 1-staged 2-order solver
    # Butcher table:
    #  1/2 | 1/2 |
    # ------------
    #      |  1  |
    @staticmethod
    def create_i2(h):
        c = np.array([1/2])
        b = np.array([1])
        A = np.array([[1/2]])
        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)
    
    # Creates implicit 2-staged 3-order solver
    # Butcher table:
    #  1/2 + a | 1/2 + a |    0    |
    #  1/2 - a |   -2a   | 1/2 + a |
    # ------------------------------
    #          |   1/2   |   1/2   |
    # where a = sqrt(3)/6      
    @staticmethod
    def create_i3(h):
        a = np.sqrt(3) / 6

        c = np.array([1/2 + a, 1/2 - a])
        b = np.array([1/2, 1/2])
        A = np.array([[1/2 + a,     0  ],
                      [  -2*a,  1/2 + a]])
        btable = ButcherTable(A, b, c)
        return RKSolver(btable, h)

class AdamsCollection:
    # Creates explicit 1-order Adams method solver
    #   y_{n+1} = y_n + h*f_n
    @staticmethod
    def create_e1(h, aux=None, sub=1):
        coeffs = [1]
        return AdamsSolver(coeffs, h, None, aux, sub)
    
    # Creates explicit 2-order Adams method solver
    #   y_{n+1} = y_n + h * (-.5 * f_{n-1} + 1.5*f_n)
    @staticmethod
    def create_e2(h, aux=None, sub=1):
        coeffs = [-.5, 1.5]
        return AdamsSolver(coeffs, h, None, aux, sub)
    
    # Creates explicit 3-order Adams method solver
    #   y_{n+1} = y_n + h * ((5/12) * f_{n-2} + (-16/12)*f_{n-1} + (23/12)*f_n)
    @staticmethod
    def create_e3(h, aux=None, sub=1):
        coeffs = [5/12, -16/12, 23/12]
        return AdamsSolver(coeffs, h, None, aux, sub)
    
    # Creates explicit 4-order Adams method solver
    #   y_{n+1} = y_n + h * ((-9/24)*f_{n-3} + (37/24) * f_{n-2} + (-59/24)*f_{n-1} + (55/24)*f_n)
    @staticmethod
    def create_e4(h, aux=None, sub=1):
        coeffs = [-9/24, 37/24, -59/24, 55/24]
        return AdamsSolver(coeffs, h, None, aux, sub)
    
    # Creates implicit 1-order Adams method solver
    #   y_{n+1} = y_n + h * f_{n + 1}
    @staticmethod
    def create_i1(h, aux=None, sub=1):
        coeffs = [0]
        c_impl = 1
        return AdamsSolver(coeffs, h, c_impl, aux, sub)
    
    # Creates implicit 2-order Adams method solver
    #   y_{n+1} = y_n + h/2 * f_{n} + h/2 * f_{n+1}
    @staticmethod
    def create_i2(h, aux=None, sub=1):
        coeffs = [1/2]
        c_impl = 1/2
        return AdamsSolver(coeffs, h, c_impl, aux, sub)
    
    # Creates implicit 3-order Adams method solver
    #   y_{n+1} = y_n + h[(-1/12)*f_{n-1} + (8/12)*f_n] + h(5/12)f_{n+1}
    @staticmethod
    def create_i3(h, aux=None, sub=1):
        coeffs = [-1/12, 8/12]
        c_impl = 5/12
        return AdamsSolver(coeffs, h, c_impl, aux, sub)
    
    # Creates implicit 4-order Adams method solver
    #   y_{n+1} = y_n + h[(1/24)f_{n-2} + (-5/24)f_{n-1} + (19/24)f_n] + h(9/24)f_{n+1}
    @staticmethod
    def create_i4(h, aux=None, sub=1):
        coeffs = [1/24, -5/24, 19/24]
        c_impl = 9/24
        return AdamsSolver(coeffs, h, c_impl, aux, sub)


class BDFCollection:
    # Creates explicit 1-order BDF method
    #   y_{n+1} = 1*h*f(y_n, t_n) + 1 * y_n
    @staticmethod
    def create_e1(h, aux=None, sub=1):
        return BDFSolver(1, [1], h, False, aux, sub)
    
    # Creates explicit 2-order BDF method
    #   y_{n+1} = 2h*f*(y_n, t_n) + 0 * y_n + 1 * y_{n-1}
    @staticmethod
    def create_e2(h, aux=None, sub=1):
        return BDFSolver(2, [0, 1], h, False, aux, sub)
    
    # Creates explicit 3-order BDF method
    #   y_{n+1} = 3h*f*(y_n, t_n) + (-3/2) * y_n + 3 * y_{n-1} + (-1/2)*y_{n-2}
    @staticmethod
    def create_e3(h, aux=None, sub=1):
        return BDFSolver(3, [-3/2, 3, -1/2], h, False, aux, sub)
    
    # Creates explicit 4-order BDF method
    #   y_{n+1} = 4h*f*(y_n, t_n) + (-10/3) * y_n + 6 * y_{n-1} + (-2)*y_{n-2} + (1/3)*y_{n-3}
    @staticmethod
    def create_e4(h, aux=None, sub=1):
        return BDFSolver(4, [-10/3, 6, -2, 1/3], h, False, aux, sub)
    
    # Creates implicit 1-order BDF method
    #   y_{n+1} = h*f_{n+1} + y_n
    @staticmethod
    def create_i1(h, aux=None, sub=1):
        return BDFSolver(1, [1], h, True, aux, sub)
    
    # Creates implicit 2-order BDF method
    #   y_{n+1} = (2/3)h*f_{n+1} + (4/3)y_n + (-1/3)y_{n-1}
    @staticmethod
    def create_i2(h, aux=None, sub=1):
        return BDFSolver(2/3, [4/3, -1/3], h, True, aux, sub)
    
    # Creates implicit 3-order BDF method
    #   y_{n+1} = (6/11)*h*f_{n+1} + (18/11)y_n + (-9/11)y_{n-1} + (2/11)y_{n-2}
    @staticmethod
    def create_i3(h, aux=None, sub=1):
        return BDFSolver(6/11, [18/11, -9/11, 2/11], h, True, aux, sub)
    
    # Creates implicit 4-order BDF method
    #   y_{n+1} = (12/25)h*f_{n+1} + (48/25)y_n + (-36/25)y_{n-1} + (16/25)y_{n-2} + (-3/25)y_{n-3}
    @staticmethod
    def create_i4(h, aux=None, sub=1):
        return BDFSolver(12/25, [48/25, -36/25, 16/25, -3/25], h, True, aux, sub)