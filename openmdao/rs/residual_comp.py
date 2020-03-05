from openmdao.api import ImplicitComponent
import numpy as np


class ResidualComp(ImplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']
        residuals['y'] = x**2 - y

    def solve_nonlinear(self, inputs, outputs):
        x = inputs['x']
        outputs['y'] = x**2

    def linearize(self, inputs, outputs, partials):
        x = inputs['x']
        # y = outputs['y']

        partials['y', 'x'] = 2. * x
        partials['y', 'y'] = -1.

        # self.inv_jac = -1


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem

    prob = Problem()

    comp = ResidualComp()
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)