from openmdao.api import ExplicitComponent
import numpy as np


class ConstraintComp(ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('x', shape=(2, ))
        self.add_output('C')
        self.declare_partials('C', 'x')

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['C'] = x[0]**2 - x[1]
        # print(outputs['obj'])

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        partials['C', 'x'] = np.array([2 * x[0], -1])
