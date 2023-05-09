import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def sphere(x):
    return np.sum(np.square(x))


def rosenbrock(x):
    return np.sum(100 * np.square(x[1:] - np.square(x[:-1])) + np.square(1 - x[:-1]))


def rastrigin(x):
    n = len(x)
    A = 10
    return A * n + np.sum(np.square(x) - A * np.cos(2 * np.pi * x))


# Define the HBMO algorithm
def hbmo(cost_func, dim, pop_size, max_iter, crossover_prob, mutation_prob):
    lower_bound = -5.12
    upper_bound = 5.12
    pop = np.random.uniform(lower_bound, upper_bound, size=(pop_size, dim))

    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = cost_func(pop[i])

    populations = [pop.copy()]

    for _ in range(max_iter):
        prob = fitness / np.sum(fitness)

        # Scout bees
        scouts = np.zeros((pop_size, dim))
        for i in range(pop_size):
            idx = np.random.choice(pop_size, p=prob)
            scouts[i] = pop[idx]

        # Crossover
        children = np.zeros((pop_size, dim))
        for i in range(pop_size):
            idx1 = i
            idx2 = np.random.choice(pop_size)
            if np.random.rand() < crossover_prob:
                alpha = np.random.rand(dim)
                children[i] = alpha * pop[idx1] + (1 - alpha) * pop[idx2]
            else:
                children[i] = pop[idx1]

        # Mutation
        for i in range(pop_size):
            if np.random.rand() < mutation_prob:
                children[i] += np.random.normal(scale=0.1, size=dim)

        # Evaluate
        children_fitness = np.zeros(pop_size)
        for i in range(pop_size):
            children_fitness[i] = cost_func(children[i])

        new_pop = np.zeros((pop_size, dim))
        new_fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if children_fitness[i] < fitness[i]:
                new_pop[i] = children[i]
                new_fitness[i] = children_fitness[i]
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]

        # Update
        pop = new_pop
        fitness = new_fitness

        # Add population to list
        populations.append(pop.copy())

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = pop[best_idx]
    return best_solution, best_fitness, populations


def animate_bees(benchmark_name, best_solution, best_fitness, populations):
    print(benchmark_name + ":")
    print("Best solution found:", best_solution)
    print("Best fitness found:", best_fitness)

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim([-5.12, 5.12])
    ax.set_ylim([-5.12, 5.12])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{benchmark_name} Function")

    
    # Initialize scatter plot
    scat = ax.scatter([], [], marker=11, alpha=0.3)

    # Update function for animation
    def update(i):
        pop = populations[i]
        scat.set_offsets(pop)
        return (scat,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=max_iter + 1, interval=100, blit=True)

    # Show animation
    plt.scatter(x=best_solution[0], y=best_solution[1], c='red', s=15)
    plt.show()
    print("=" * 100)

dim = 2
pop_size = 50
max_iter = 1000
crossover_prob = 0.5
mutation_prob = 0.1


# Run HBMO algorithm on Sphere function
sphere_best_solution, sphere_best_fitness, sphere_populations = hbmo(
    sphere, dim, pop_size, max_iter, crossover_prob, mutation_prob
)

# Run HBMO algorithm on Rosenbrock function
rosenbrock_best_solution, rosenbrock_best_fitness, rosenbrock_populations = hbmo(
    rosenbrock, dim, pop_size, max_iter, crossover_prob, mutation_prob
)

# Run HBMO algorithm on Rastrigin function
rastrigin_best_solution, rastrigin_best_fitness, rastrigin_populations = hbmo(
    rastrigin, dim, pop_size, max_iter, crossover_prob, mutation_prob
)

animate_bees(benchmark_name="Sphere", best_solution=sphere_best_solution, best_fitness=sphere_best_fitness, populations=sphere_populations)
animate_bees(benchmark_name="Rosenbrock", best_solution=rosenbrock_best_solution, best_fitness=rosenbrock_best_fitness, populations=rosenbrock_populations)
animate_bees(benchmark_name="Rastrigin", best_solution=rastrigin_best_solution, best_fitness=rastrigin_best_fitness, populations=rastrigin_populations)

