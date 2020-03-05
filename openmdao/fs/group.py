from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem
import numpy as np

from obj_comp import ObjComp
from constraint_comp import ConstraintComp


class group(Group):
    def initialize(self):
        pass

    def setup(self):
        IVC = IndepVarComp()
        IVC.add_output('x', val=np.array([1.5, 1]))

        self.add_subsystem('IVC', IVC, promotes=['*'])

        comp = ObjComp()
        self.add_subsystem('ObjComp', comp, promotes=['*'])

        comp = ConstraintComp()
        self.add_subsystem('ConstraintComp', comp, promotes=['*'])
