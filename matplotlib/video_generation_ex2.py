import theano
from theano import tensor as T
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


trX = np.linspace(-1,1, 101)
trY = 2.0 * trX + np.random.randn(*trX.shape) * 0.33

X = T.scalar()
Y = T.scalar()

def model(X, w):
    return X * w

w = theano.shared(np.asarray(0.0, dtype=theano.config.floatX))
y = model(X, w)


cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
updates = [[w, w - gradient * 0.01]]

train = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))

data_line = plt.plot([],[],'o')[0]
reg_line = plt.plot([],[],"--",linewidth=2)[0]



# initialization function: plot the background of each frame
def init():
    data_line.set_data(trX,trY)
    reg_line.set_data([],[])
    return [data_line,reg_line]

# animation function.  This is called sequentially
def animate(i):
    reg_line.set_data(trX,trX * w.get_value())
    for x, y in zip(trX, trY):
            train(x, y)
    return [data_line,reg_line]

#plt.title('Current Slope = %s' % str(w.get_value()))
#plt.plot(trX,trY,'o') 
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('example2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

        

