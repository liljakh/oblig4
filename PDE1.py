'''
Lilja Katharina Høiback
oblig
15.03.2024
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


# matematiske verdier
L = 2
T = 0.1

# antall punkter
N = 10
N_t = 20

# initialbetingelser(randen er 0)
func = lambda x: x #np.exp(-10*(x-L/2)**2)
#steglenden er N-1 pga 1 steg færre enn antall punkter
h_x = L/(N-1)
k = T/(N_t-1)

gamma = k/(h_x**2)
if gamma >= 0.5:
    print('gamma er for stor')
else:
    print('gamma:',gamma)

# punktgitter
x = np.linspace(0,L,N)
t = np.linspace(0, T, N_t)

# putter startverdiene inn i en matrise
u_global = np.zeros((N,N_t))
func_vals = np.array([func(x_val) for x_val in x])
u_global[1: -1,0] = func_vals[1:-1]

u_1 = u_global.copy()

#ekplesitt metode
def func2():
    B = (1-2*gamma)*np.eye(N-2) + gamma*np.eye(N-2, k= -1) + gamma*np.eye(N-2,k=1)

    for j in range(N_t-1):
        u_1[1:-1, j+1] = np.matmul(B, u_1[1:-1, j])

meshX, meshT = np.meshgrid(x, t)
u_1_rotated = np.flip(np.rot90(u_1))

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

ax.plot_surface(meshX,meshT, u_1_rotated , cmap = cm.coolwarm)
#Plottet blir desverre litt rart, litt sammentrøkt, vet ikke helt hvordan jeg fikser dette
plt.show()