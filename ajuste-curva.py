import anfis.anfis as c_anfis
import anfis.membership as amf
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def curve(x, y):
    return (math.sin(x)*math.sin(y))/(x*y)

size = 2000
x = [np.random.uniform(-10, 10) for i in range(0, size)]
y = [np.random.uniform(-10, 10) for i in range(0, size)]
z = [curve(x[i], y[i]) for i in range(0, size)]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()

X = np.array([[x[i], y[i]] for i in range(0, size)])
Y = np.array(z)

mf = [[['gaussmf',{'mean':-8.,'sigma':1}],['gaussmf',{'mean':-3,'sigma':1}],['gaussmf',{'mean':3,'sigma':1}],['gaussmf',{'mean':8,'sigma':1}]],
            [['gaussmf',{'mean':-10,'sigma':1}],['gaussmf',{'mean':-5,'sigma':1}],['gaussmf',{'mean':5,'sigma':1}],['gaussmf',{'mean':10,'sigma':1}]]]

mfc = amf.membershipfunction.MemFuncs(mf)
anf = c_anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=30)
print round(anf.consequents[-1][0],6)
print round(anf.consequents[-2][0],6)
print round(anf.fittedValues[9][0],6)

anf.plotErrors()
anf.plotResults()