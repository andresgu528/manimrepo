from scipy.integrate import solve_ivp
import numpy as np
import sympy
import sympy.physics.mechanics as mechanics

mechanics.init_vprinting()

m1, m2, l1, l2, g, t = sympy.symbols(['m1', 'm2', 'l1', 'l2', 'g', 't'])
th1, th2 = mechanics.dynamicsymbols(['theta1', 'theta2'])
th1d, th2d = mechanics.dynamicsymbols(['theta1', 'theta2'], 1)

N = mechanics.ReferenceFrame('N')
O = mechanics.Point('O')
O.set_vel(N, 0*N.x)
A = N.orientnew('A', 'Axis', [th1, N.z])
P1 = O.locatenew('P1', -l1*A.y)
B = N.orientnew('B', 'Axis', [th2, N.z])
P2 = P1.locatenew('P2', -l2*B.y)

p1 = mechanics.Particle('p1', P1, m1)
p2 = mechanics.Particle('p2', P2, m2)
p1.potential_energy = -m1*g*l1*sympy.cos(th1)
p2.potential_energy = -m2*g*(l1*sympy.cos(th1)+l2*sympy.cos(th2))
LM = mechanics.LagrangesMethod(mechanics.Lagrangian(N, p1, p2), [th1, th2])
LM.form_lagranges_equations()
derivative_sym = LM.rhs()

param_dict = {m1: 1,
                  m2: 1,
                  l1: 1,
                  l2: 1,
                  g: 9.8}

var_dict = {th1: 1,
            th2: 1,
            th1d: 0,
            th2d: 0}

derivative = derivative_sym.evalf(subs=param_dict)

derivative_func = sympy.lambdify([t, (th1, th2, th1d, th2d)],
                                 derivative.T.tolist()[0],
                                 'numpy')

num_sol = solve_ivp(derivative_func, [0,5], [0,0,0,1])