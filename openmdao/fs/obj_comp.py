from openmdao.api import ExplicitComponent
import numpy as np


class ObjComp(ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('x', shape=(2, ))
        self.add_output('obj')
        self.declare_partials('obj', 'x')

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['obj'] = x[0] * x[1] - 0.5 * x[0]**2 - x[1]
        # print(outputs['obj'])

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        partials['obj', 'x'] = np.array([x[1] - x[0], x[0] - 1])
