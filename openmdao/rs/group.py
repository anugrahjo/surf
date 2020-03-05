from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem
import numpy as np

from obj_comp import ObjComp
from residual_comp import ResidualComp


class group(Group):
    def initialize(self):
        pass

    def setup(self):
        IVC = IndepVarComp()
        IVC.add_output('x', val=1.5)

        self.add_subsystem('IVC', IVC, promotes=['*'])

        comp = ObjComp()
        self.add_subsystem('ObjComp', comp, promotes=['*'])

        comp = ResidualComp()
        self.add_subsystem('ResidualComp', comp, promotes=['*'])
