import numpy as np
import matplotlib.pyplot as plt

def sphere(x, y):
    return x**2 + y**2

def rosenbrock(x, y):
    return (1-x)**2 + 100*(y-x**2)**2

def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)



# 2D plots
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].contour(X, Y, rastrigin(X, Y), levels=20)
axs[0].set_title("Rastrigin Function")
axs[1].contour(X, Y, rosenbrock(X, Y), levels=20)
axs[1].set_title("Rosenbrock Function")
axs[2].contour(X, Y, sphere(X, Y), levels=20)
axs[2].set_title("Sphere Function")

# 3D plots
fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(X, Y, rastrigin(X, Y), cmap='viridis')
ax.set_title("Rastrigin Function")

ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(X, Y, rosenbrock(X, Y), cmap='viridis')
ax.set_title("Rosenbrock Function")

ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(X, Y, sphere(X, Y), cmap='viridis')
ax.set_title("Sphere Function")

plt.show()