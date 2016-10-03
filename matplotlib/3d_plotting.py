import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation


t = 0
myfps = 40
size = 40.0
step = 0.1
damping_ratio = 0.1
radius_ratio = 0.8

fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim(-1.0,1.0)
X = np.arange(-size, size, step)
Y = np.arange(-size, size, step)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)


Z = np.cos(R * radius_ratio + 2.0 * np.pi * t) * np.exp(-damping_ratio * R)
drop_wave = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap='jet')

def update_wave(num=0):
    ax.clear()
    ax.set_zlim(-1.0,1.0)
    t = float(num)/ float(myfps)
    print num,t
    Z = np.cos(R * radius_ratio - 2.0 * np.pi * t) * np.exp(-damping_ratio * R)
    return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap='jet')


wave_ani = animation.FuncAnimation(fig, update_wave, init_func=update_wave,frames=myfps, interval=5)

wave_ani.save('wave.mp4', fps=myfps, extra_args=['-vcodec', 'libx264'])

