'''
MarabouCore Example
====================

Top contributors (to current version):
  - Christopher Lazarus
  - Kyle Julian
  - Andrew Wu
  
This file is part of the Marabou project.
Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.
'''

from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

large = 10.0
inputQuery = MarabouCore.InputQuery()
inputQuery.setNumberOfVariables(6)

inputQuery.setLowerBound(0, 0)
inputQuery.setUpperBound(0, 1)
inputQuery.setLowerBound(1, -large)
inputQuery.setUpperBound(1, large)
inputQuery.setLowerBound(2, 0)
inputQuery.setUpperBound(2, large)
#inputQuery.setLowerBound(3, -large)
inputQuery.setLowerBound(3, 0)  # want to keep active phase of this relu
inputQuery.setUpperBound(3, large)
inputQuery.setLowerBound(4, 0)
inputQuery.setUpperBound(4, large)
inputQuery.setLowerBound(5, 0.5)
inputQuery.setUpperBound(5, 1)

equation1 = MarabouCore.Equation()
equation1.addAddend(1, 0)
equation1.addAddend(-1, 1)
equation1.setScalar(0)
inputQuery.addEquation(equation1)

equation2 = MarabouCore.Equation()
equation2.addAddend(1, 0)
equation2.addAddend(1, 3)
equation2.setScalar(0)
inputQuery.addEquation(equation2)

equation3 = MarabouCore.Equation()
equation3.addAddend(1, 2)
equation3.addAddend(1, 4)
equation3.addAddend(-1, 5)
equation3.setScalar(0)
inputQuery.addEquation(equation3)


#equation for active relu output x2b - x2f = 0, x2b variable index 3, x2f variable index 4
#equation in the form ax + by = c

equation4 = MarabouCore.Equation()
equation4.addAddend(1, 3)
equation4.addAddend(-1, 4)
equation4.setScalar(0)
inputQuery.addEquation(equation4)


MarabouCore.addReluConstraint(inputQuery, 1, 2)
#MarabouCore.addReluConstraint(inputQuery, 3, 4) #removing this relu constraints as it is already appered in equation 4

options = createOptions()
exitCode, vars, stats = MarabouCore.solve(inputQuery, options, "")
print(exitCode, vars, stats)
if exitCode == "sat":
    print(vars)
