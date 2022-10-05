import numpy as np
from scipy.integrate import solve_ivp
import sympy
import sympy.physics.mechanics as mechanics
import manim

mechanics.init_vprinting()

class NPendulum:
    def __init__(self, n, masses=None, lengths=None, gravity=None):
        
        self.n = n
        self.solved_equations = None
        self.position_funcs = None
        # parameters
        m = sympy.symbols(f'm:{n}', real=True)
        l = sympy.symbols(f'l:{n}', real=True)
        g = sympy.Symbol('g', real=True)

        # canonical coordinates
        theta = mechanics.dynamicsymbols(f'theta:{n}', real=True)
        dtheta = mechanics.dynamicsymbols(f'theta:{n}', 1, real=True)

        # inertial reference frame
        N = mechanics.ReferenceFrame('N')
        O = mechanics.Point('O')
        O.set_vel(N, 0)

        Rfs = [N] # Reference frames
        Ps = [O] # Points
        Pas = [] # Particles
        for i in range(n):
            Rf = N.orientnew(f'Rf{i}', 'Axis', (theta[i], N.z))
            P = Ps[i].locatenew(f'P{i}', -l[i]*Rf.y)
            Pa = mechanics.Particle(f'Pa{i}', P, m[i])
            Pa.potential_energy = m[i]*g*mechanics.dot(P.pos_from(O), N.y)
            Rfs.append(Rf)
            Ps.append(P)
            Pas.append(Pa)
        
        self.coordinates = theta
        self.dcoordinates = dtheta
        self.frames = Rfs
        self.points = Ps
        self.particles = Pas

        L = mechanics.Lagrangian(N, *Pas)
        self.lagrangian = L
        self.energy = mechanics.kinetic_energy(self.frames[0], *self.particles) \
            + mechanics.potential_energy(*self.particles)
        LM = mechanics.LagrangesMethod(L, theta)
        
        self.motionequations = LM.form_lagranges_equations()
        self.derivatives_vector = LM.rhs()

        self.m = m
        self.l = l
        self.g = g

        if (masses or lengths or gravity):
            self.set_params(masses, lengths, gravity)


    def set_params(self, m, l, g):
        self.params = dict(zip(self.m+self.l+(self.g,), m+l+(g,)))
    
    def solve_motion_equations(self, t_span, y0, m=None, l=None, g=None):

        if (m or l or g):
            self.set_params(m, l, g)
        
        t = sympy.Symbol('t', real=True)

        derivatives_vector = self.derivatives_vector.evalf(
            subs=self.params).T.tolist()[0]
        derivatives_func = sympy.lambdify(
            [t,self.coordinates+self.dcoordinates],
            derivatives_vector,
            modules='numpy')
        solution = solve_ivp(
            derivatives_func, t_span=t_span, y0=y0, dense_output=True,
            rtol=1e-6, atol=1e-6
            ).sol
        self.solved_equations = solution
        return solution
    
    def generate_position_funcs(self):
        ref_point = self.points[0]
        ref_frame = self.frames[0]
        self.position_funcs = []
        for i in range(self.n):
            pos_vect = self.points[i+1].pos_from(ref_point).to_matrix(ref_frame)
            pos_vect = pos_vect.evalf(subs=self.params).T.tolist()[0]
            pos_func = sympy.lambdify(
                [self.coordinates+self.dcoordinates],
                pos_vect,
                modules='numpy')
            self.position_funcs.append(pos_func)
    
    def get_point_coordinates(self,
                              t,
                              point: mechanics.Point,
                              ref_point: mechanics.Point = None,
                              ref_frame: mechanics.ReferenceFrame = None):
        if ref_point is None:
            ref_point = self.points[0]
        if ref_frame is None:
            ref_frame = self.frames[0]
        coordinates = point.pos_from(ref_point).to_matrix(ref_frame)
        return self.evaluate_time(t, coordinates)
    
    def evaluate_time(self, t, expression):
        evaluated_coords = self.solved_equations(t)
        eval_dict = dict(zip(self.coordinates + self.dcoordinates, list(evaluated_coords)))
        eval_dict.update(self.params)
        return np.array(expression.evalf(subs=eval_dict)).astype(float).squeeze()
    
    def get_total_energy(self, t):
        return self.evaluate_time(t, self.energy)

class NPendulumScene(manim.Scene):
    def construct(self):

        manim.config.frame_width = 11

        self.camera.frame_center = [0,-0.75,0]

        n = 4
        m = (1,1.5, 1.25, 1); l=(0.5,1,0.75,0.5); g=9.8
        y0 = (np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0)
        npen = NPendulum(n)
        npen.set_params(m, l, g)
        t0, tf = t_span = (0, 30)
        
        print('Solving ODE...', end=' ')
        npen.solve_motion_equations(t_span,  y0)
        print('Done!')

        t = manim.ValueTracker(t0)
        points = manim.VGroup(manim.Dot())
        lines = manim.VGroup()
        
        for i in range(n):
            p = manim.Dot(radius=0.1*m[i]**0.5,
                          color=manim.color.Color(hue=i/n, saturation=0.5, luminance=0.5))
            p.add_updater(lambda mob, i=i: mob.move_to(
                npen.get_point_coordinates(t.get_value(), npen.points[i+1])))
            points.add(p)

            line = manim.always_redraw(
                lambda i=i: manim.Line(
                    npen.get_point_coordinates(t.get_value(), npen.points[i]),
                    npen.get_point_coordinates(t.get_value(), npen.points[i+1]),
                    color=manim.YELLOW
                    ))
            lines.add(line)

        points.update()
        lines.update()

        plane = manim.NumberPlane(x_range=[-8,8], y_range=[-6,6])
        energy = manim.DecimalNumber(0, 2)
        energy.move_to([-2,2,0])
        energy.add_updater(lambda mob: mob.set_value(npen.get_total_energy(t.get_value())))
        energy.update()

        self.add(plane, lines, points, energy)
        self.play(t.animate.set_value(tf), run_time=tf-t0, rate_func=manim.linear)
