import numpy as np
import collections

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
        assert c[0] == 0

    def get_s(self):
        return len(self.b)

class RKSolver:    
    def __init__(self, btable, h):
        self.btable = btable
        self.h = float(h)
        self.nsteps = 0

        self.problem = None
        self.t = None
        self.y = None
        self.k = None

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
        self.k[0] = f(y, t)
        for i in range(1, self.btable.get_s()):
            conv = np.zeros((len(y)))
            for j in range(i):
                conv += a[i][j] * self.k[j]

            self.k[i] = f(y + h * conv, t + c[i]*h)

        self.t += h
        self.y += h * np.dot(self.k.T, b)

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
class MultiStepSolverBase:
    def __init__(self, steps, h):
        self.h = h
        self.steps = steps
        self.func = None
        self._ytfcache = collections.deque()

    def init_problem(self, func, t0, y0):
        self.func = func
        self.__start_site(t0, y0)

    # Using Runge-Kutta Method to
    # evaluate first N points
    def __start_site(self, t0, y0):
        solver = MultiStepSolverBase.__create_aux_solver(self.h)
        solver.init_problem(self.func, t0, y0)

        for _ in range(self.steps):
            t, y = solver.step()
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
    def __init__(self, coeffs, h):
        super().__init__(len(coeffs), h)
        self.coeffs = np.array(coeffs).astype(CALC_TYPE)
    
    def step(self):
        ret = self._ytf_cache_tail() # oldest y, t, f
        y, t, _  = self._ytf_cache_head() # yn, tn, fn

        conv = np.zeros_like(y)
        for c, (_, __, f) in zip(self.coeffs, self._ytfcache):
            conv += c * f

        y += self.h * conv
        t += self.h

        self._ytf_cache_update(y, t, self.func(y, t))
        
        return ret[1], ret[0]
    
    def get_h(self): return self.h

class BDFSolver(MultiStepSolverBase):
    # Params define method
    # y_{n+1} = A*h*f(y_n, t_n) + c_0*y_n + c_1*y_{n-1} + c_2*y_{n-2} + ...
    def __init__(self, A, coeffs, h):
        super().__init__(len(coeffs), h)
        self.A = A
        self.coeffs = np.array(coeffs[::-1]).astype(CALC_TYPE)
    
    def step(self):
        ret = self._ytf_cache_tail()
        _, t, f = self._ytf_cache_head()

        y_new = self.A * self.h * f
        for c, (y, _, __) in zip(self.coeffs, self._ytfcache):
            y_new += c * y

        self._ytf_cache_update(y_new, t + self.h, self.func(y_new, t + self.h))
        
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

class AdamsCollection:
    # Creates explicit 1-order Adams method solver
    # y_{n+1} = y_n + h*f_n
    @staticmethod
    def create_e1(h):
        coeffs = [1]
        return AdamsSolver(coeffs, h)
    
    # Creates explicit 2-order Adams method solver
    # y_{n+1} = y_n + h * (-.5 * f_{n-1} + 1.5*f_n)
    @staticmethod
    def create_e2(h):
        coeffs = [-.5, 1.5]
        return AdamsSolver(coeffs, h)
    
    # Creates explicit 3-order Adams method solver
    # y_{n+1} = y_n + h * ((5/12) * f_{n-2} + (-16/12)*f_{n-1} + (23/12)*f_n)
    @staticmethod
    def create_e3(h):
        coeffs = [5/12, -16/12, 23/12]
        return AdamsSolver(coeffs, h)
    
    # Creates explicit 4-order Adams method solver
    # y_{n+1} = y_n + h * ((-9/24)*f_{n-3} + (37/24) * f_{n-2} + (-59/24)*f_{n-1} + (55/24)*f_n)
    @staticmethod
    def create_e4(h):
        coeffs = [-9/24, 37/24, -59/24, 55/24]
        return AdamsSolver(coeffs, h)

class BDFCollection:
    # Creates explicit 1-order BDF method
    # y_{n+1} = 1*h*f(y_n, t_n) + 1 * y_n
    @staticmethod
    def create_e1(h):
        return BDFSolver(1, [1], h)
    
    # Creates explicit 2-order BDF method
    # y_{n+1} = 2h*f*(y_n, t_n) + 0 * y_n + 1 * y_{n-1}
    @staticmethod
    def create_e2(h):
        return BDFSolver(2, [0, 1], h)
    
    # Creates explicit 3-order BDF method
    # y_{n+1} = 3h*f*(y_n, t_n) + (-3/2) * y_n + 3 * y_{n-1} + (-1/2)*y_{n-2}
    @staticmethod
    def create_e3(h):
        return BDFSolver(3, [-3/2, 3, -1/2], h)
    
    # Creates explicit 4-order BDF method
    # y_{n+1} = 4h*f*(y_n, t_n) + (-10/3) * y_n + 6 * y_{n-1} + (-2)*y_{n-2} + (1/3)*y_{n-3}
    @staticmethod
    def create_e4(h):
        return BDFSolver(4, [-10/3, 6, -2, 1/3], h)