import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openmdao.api import Problem, ScipyOptimizeDriver, view_model
from openmdao.api import pyOptSparseDriver
from group import group

prob = Problem()
prob.model = group()

prob.model.add_design_var('x')

prob.model.add_constraint('C', equals=0)

prob.model.add_objective('obj')

# prob.driver = ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'

# prob.driver.options['tol'] = 1e-9
# prob.driver.options['obj'] = True

prob.setup()
prob.run_model()
prob.run_driver()
print(prob['obj'])
print(prob['C'])
print(prob['x'])

view_model(prob)
