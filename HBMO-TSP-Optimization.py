import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data from the CSV file
data = pd.read_csv("15-Points.csv")

# Compute the distance matrix
num_cities = len(data)
distances = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        distances[i, j] = np.linalg.norm(
            data.loc[i, ["x", "y"]] - data.loc[j, ["x", "y"]]
        )
        distances[j, i] = distances[i, j]


# Define the fitness function for the TSP
def tsp_fitness(solution, distances):
    distance = 0
    for i in range(len(solution) - 1):
        distance += distances[solution[i], solution[i + 1]]
    distance += distances[solution[-1], solution[0]]
    return distance


# Define the 2-opt local search operator for the TSP
def tsp_2opt(solution, distances):
    num_cities = len(solution)
    for i in range(num_cities - 1):
        for j in range(i + 1, num_cities):
            new_solution = np.copy(solution)
            new_solution[i : j + 1] = np.flip(solution[i : j + 1])
            new_fitness = tsp_fitness(new_solution, distances)
            if new_fitness < tsp_fitness(solution, distances):
                return new_solution, new_fitness
    return solution, tsp_fitness(solution, distances)


temp_ = []

# Initialize the honey bees for the TSP
num_bees = 50
solutions = np.zeros((num_bees, num_cities), dtype=np.int32)
for i in range(num_bees):
    solutions[i] = np.random.permutation(num_cities)

# Evaluate the fitness of the solutions for the TSP
fitness = np.zeros(num_bees)
for i in range(num_bees):
    solutions[i], fitness[i] = tsp_2opt(solutions[i], distances)

# Main loop for the TSP
num_iterations = 100
best_fitness_tsp = np.inf
stagnation_count = 0
counter = 0
for t in range(num_iterations):
    # Select the elite bees
    elite_bees = np.argsort(fitness)[: num_bees // 2]

    # Employ the scout bees
    for i in range(num_bees // 2, num_bees):
        solutions[i] = np.random.permutation(num_cities)
        solutions[i], fitness[i] = tsp_2opt(solutions[i], distances)

    # Apply the 2-opt local search operator to the elite bees and their neighbors
    for i in elite_bees:
        for j in range(i - 2, i + 3):
            if j >= 0 and j < num_bees and j != i:
                solutions[j], fitness[j] = tsp_2opt(solutions[j], distances)

    # Replace the worst solutions
    worst_bees = np.argsort(fitness)[::-1][: num_bees // 2]
    for i in range(num_bees // 2):
        solutions[worst_bees[i]] = solutions[elite_bees[i]]
        solutions[worst_bees[i]], fitness[worst_bees[i]] = tsp_2opt(
            solutions[worst_bees[i]], distances
        )

    # Update the best fitness value for the TSP
    best_fitness_tsp = min(best_fitness_tsp, np.min(fitness))

    # Check for stagnation
    if best_fitness_tsp <= np.min(fitness):
        stagnation_count += 1
    else:
        stagnation_count = 0

    # Terminate if the algorithm has stagnated
    if stagnation_count >= 10:
        break

    # Print the best fitness value for the TSP
    print(f"Iteration {t+1}: Best fitness = {best_fitness_tsp}")
    temp_.append(best_fitness_tsp)
    counter += 1


def visualize_path(best_solution, distance, X, Y):
    fig, ax = plt.subplots()

    (line,) = ax.plot([], [], "-o", linewidth=2)
    (dot,) = ax.plot([], [], "o", color="black")

    ax.set_xlim(min(X) - 1, max(X) + 1)
    ax.set_ylim(min(Y) - 1, max(Y) + 1)
    ax.scatter(X, Y, c="red")
    ax.set_title(f"Solution Order: {best_solution}\nBest Distance: {distance}")
    for i, (x, y) in enumerate(zip(X, Y)):
        if i == 0:
            ax.annotate(str(i), (x + 1, y - 2))
        elif i == len(X) - 1:
            ax.annotate(str(i), (x + 1, y + 2))
        else:
            ax.annotate(str(i), (x + 1, y + 1))

    def update(frame):
        x = X[: frame + 1]
        y = Y[: frame + 1]
        line.set_data(x, y)
        dot.set_data(x[-1], y[-1])

        return line, dot

    ani = animation.FuncAnimation(fig, update, frames=len(X), interval=250, blit=False)
    plt.show()


best_solution_index = np.argmin(fitness)
best_solution = solutions[best_solution_index]

print(best_solution)
print(best_fitness_tsp)


coords_X = list(data.loc[best_solution, "x"])
coords_X.append(coords_X[0])
coords_Y = list(data.loc[best_solution, "y"])
coords_Y.append(coords_Y[0])

visualize_path(best_solution, best_fitness_tsp, coords_X, coords_Y)

# # Plot the data
plt.plot(range(counter), temp_, label="Honeybee Mating Optimization")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Fitness Value")
plt.legend()
plt.show()
