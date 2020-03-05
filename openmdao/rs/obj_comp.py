from openmdao.api import ExplicitComponent
import numpy as np


class ObjComp(ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('x')
        self.add_input('y')
        self.add_output('obj')
        self.declare_partials('obj', 'x')
        self.declare_partials('obj', 'y')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['obj'] = x * y - 0.5 * x**2 - y
        # print(outputs['obj'])

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        partials['obj', 'x'] = y - x
        partials['obj', 'y'] = x - 1


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem

    prob = Problem()

    comp = ObjComp()
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
