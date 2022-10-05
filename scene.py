from manim import *
import numpy as np

class Disperse(Animation):
    def __init__(self, mobject, dot_radius=0.05, dot_number=100, **kwargs):
        super().__init__(mobject, **kwargs)
        self.dot_radius = dot_radius
        self.dot_number = dot_number
    
    def begin(self):
        dots = VGroup(
            *[Dot(radius=self.dot_radius).move_to(self.mobject.point_from_proportion(p))
                for p in np.linspace(0,1, self.dot_number)])
        for dot in dots:
            dot.initial_position = dot.get_center()
            dot.shift_vector = 2*(dot.get_center() - self.mobject.get_center())
        dots.set_opacity(0)
        self.mobject.add(dots)
        self.dots = dots
        super().begin()
    
    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        scene.remove(self.dots)
    
    def interpolate_mobject(self, alpha):
        alpha = self.rate_func(alpha)
        if alpha <= 0.5:
            self.mobject.set_opacity(1-2*alpha, family=False)
            self.dots.set_opacity(2*alpha)
        else:
            self.mobject.set_opacity(0)
            self.dots.set_opacity(2*(1-alpha))
            for dot in self.dots:
                dot.move_to(dot.initial_position + 2*(alpha-0.5)*dot.shift_vector)

class CustomAnimationExample(Scene):
    def construct(self):
        st = Square(color=YELLOW, fill_opacity=1).scale(3)
        self.add(st)
        self.wait()
        self.play(Disperse(st, dot_radius=0.01, dot_number=500, run_time=4))

class ComplexExp(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=(-0.1, 4.25), y_range=(-1.5, 1.5), z_range=(-1.5, 1.5), y_length=5, z_length=5)
        curve = ParametricFunction(
            lambda p: axes.coords_to_point(p, np.exp(complex(0, PI*p)).real, np.exp(complex(0, PI*p)).imag),
            t_range=(0, 2, 0.1)
        )
        curve_extension = ParametricFunction(
            lambda p: axes.coords_to_point(p, np.exp(complex(0, PI*p)).real, np.exp(complex(0, PI*p)).imag),
            t_range=(2, 4, 0.1)
        )
        t = MathTex("z = e^{t \pi i}, \quad t\in [0, 2]")
        t.rotate(axis=OUT, angle=90*DEGREES).rotate(axis=UP, angle=90*DEGREES)
        t.next_to(curve, UP + OUT)
        self.set_camera_orientation(phi=90*DEGREES, theta=0, focal_distance=10000)
        self.add(axes)
        self.play(Create(curve, run_time=2), Write(t))
        self.wait()
        self.move_camera(phi=75*DEGREES, theta=-30*DEGREES)
        self.wait()
        four = MathTex("4").rotate(axis=OUT, angle=90*DEGREES).rotate(axis=UP, angle=90*DEGREES)
        four.move_to(t[0][12])
        self.play(Create(curve_extension, run_time=2), t[0][12].animate.become(four))
        self.wait()
        self.move_camera(phi=90*DEGREES, theta=-90*DEGREES, focal_distance=10000)
        self.wait()
        self.move_camera(phi=75*DEGREES, theta=-30*DEGREES)
        self.wait()
        self.move_camera(phi=0, theta=-90*DEGREES, focal_distance=10000)
        self.wait()
        self.move_camera(phi=75*DEGREES, theta=-30*DEGREES)
        self.wait()
        self.play(FadeOut(axes, curve, curve_extension, t, shift=IN))
        self.wait()