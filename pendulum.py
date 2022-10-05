import numpy as np
from scipy.integrate import solve_ivp
from manim import *
from manim.opengl import *

config.frame_width = 10

class Pendulum(Scene):
    def construct(self):

        mensaje = Text("Tiamu Karen <3", font_size=20, color=PURPLE)
        ti, tf = (0, 8.53)
        m1 = 1
        l1 = 1
        g = 9.8

        thi = PI/3
        wi = 0

        def pendulum_derivative(t, y):
            return [y[1], -g/l1*np.sin(y[0])]

        ode_solution = solve_ivp(pendulum_derivative, (ti,tf), (thi, wi), dense_output=True).sol

        t = ValueTracker(ti)
        plane = NumberPlane(x_range=(-3,5), y_range=(-3,1.5))
        p = Dot(color=RED)
        
        def dot_coordinates(t):
            th = ode_solution(t)[0]
            return l1*np.array([np.sin(th), -np.cos(th), 0])+plane.get_origin()
        
        def get_line():
            return Line(plane.get_origin(), dot_coordinates(t.get_value()), color=YELLOW)

        p.add_updater(lambda mob: mob.move_to(dot_coordinates(t.get_value())))
        mensaje.add_updater(lambda mob: mob.next_to(p, RIGHT, buff=0.5))
        line = always_redraw(get_line)

        p.update()
        line.update()
        self.add(plane, line, p, mensaje)
        self.play(t.animate.set_value(tf), run_time = tf-ti, rate_func=linear)

class DoublePendulum(Scene):
    def construct(self):

        mensaje = Text("Me volví adicto\nAyudaaaaaa", font_size=36, color=PURPLE, should_center=True)
        ti, tf = (0, 59)
        m1, m2 = (2, 1)
        l1, l2 = (1, 0.5)
        g = 9.8

        th1i, th2i = (PI, PI/2)
        w1i, w2i = (0, 0)

        def kinetic(v):
            th1, th2, w1, w2 = v
            return 0.5*(m1+m2)*(l1*w1)**2 + 0.5*m2*(l2*w2)**2 + m2*l1*l2*w1*w2*np.cos(th1-th2)
        def potential(v):
            th1, th2, w1, w2 = v
            return -m1*g*l1*np.cos(th1) - m2*g*(l1*np.cos(th1) + l2*np.cos(th2))
        def energy(v):
            return kinetic(v) + potential(v)
        
        Ei = energy([th1i, th2i, w1i, w2i])

        def pendulum_derivative(t, y):
            th1, th2, w1, w2 = (y[0], y[1], y[2], y[3])
            dw1 = (-m2*l2*w2**2*np.sin(th1-th2) - (m1+m2)*g*np.sin(th1) \
                       - m2*l1*w1**2*np.sin(th1-th2)*np.cos(th1-th2) + m2*g*np.sin(th2)*np.cos(th1-th2)) \
                       / (l1*(m1 + m2*np.sin(th1-th2)**2))
            dw2 = (m2*l2*w2**2*np.sin(th1-th2)*np.cos(th1-th2) + (m1+m2)*g*np.sin(th1)*np.cos(th1-th2) \
                       + (m1+m2)*l1*w1**2*np.sin(th1-th2) - (m1+m2)*g*np.sin(th2)) \
                       / (l2*(m1 + m2*np.sin(th1-th2)**2))
            return [w1, w2, dw1, dw2]

        ode_solution = solve_ivp(pendulum_derivative,
                                 (ti,tf),
                                 (th1i, th2i, w1i, w2i),
                                 method='DOP853',
                                 dense_output=True, rtol=1e-12, atol=1e-12).sol

        t = ValueTracker(ti)
        E = DecimalNumber(Ei)
        E.add_updater(lambda mob: mob.set_value(energy(ode_solution(t.get_value()))))
        plane = NumberPlane(x_range=(-3,7), y_range=(-3,2.5))
        p1 = Dot(color=RED, radius=DEFAULT_DOT_RADIUS*np.sqrt(m1))
        p2 = Dot(color=BLUE, radius=DEFAULT_DOT_RADIUS**np.sqrt(m2))
        E.move_to(plane.get_origin() + np.array([-2, 2, 0]))
        
        def dot1_coordinates(t):
            th1 = ode_solution(t)[0]
            return l1*np.array([np.sin(th1), -np.cos(th1), 0])+plane.get_origin()
        
        def dot2_coordinates(t):
            th2 = ode_solution(t)[1]
            dot1 = dot1_coordinates(t)
            return dot1 + l2*np.array([np.sin(th2), -np.cos(th2), 0])
        
        def get_line1():
            return Line(plane.get_origin(), dot1_coordinates(t.get_value()), color=YELLOW)
        
        def get_line2():
            return Line(dot1_coordinates(t.get_value()), dot2_coordinates(t.get_value()), color=YELLOW)

        p1.add_updater(lambda mob: mob.move_to(dot1_coordinates(t.get_value())))
        p2.add_updater(lambda mob: mob.move_to(dot2_coordinates(t.get_value())))
        mensaje.add_updater(lambda mob: mob.next_to(p2, RIGHT, buff=0.5))
        line1 = always_redraw(get_line1)
        line2 = always_redraw(get_line2)

        p1.update()
        p2.update()
        line1.update()
        line2.update()
        self.add(plane, E, line1, p1, line2, p2, mensaje)
        self.play(t.animate.set_value(tf), run_time = tf-ti, rate_func=linear)