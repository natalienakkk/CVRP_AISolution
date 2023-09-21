import numpy as np
import copy
import math
import random

# Ackley function definition
def ackley(x):
    n = len(x)
    sum1 = -0.2 * math.sqrt(sum([i**2 for i in x]) / n)
    sum2 = sum([math.cos(2 * math.pi * i) for i in x]) / n
    result = 20 + math.e - 20 * math.exp(sum1) - math.exp(sum2)
    return result


def get_neighbours(solution, num_neighbours=None):
    if num_neighbours is None:
        num_neighbours = 1000

    neighbours = []
    for _ in range(num_neighbours):
        neighbour = copy.deepcopy(solution)
        num_dimensions_to_perturb = np.random.randint(1, len(neighbour) + 1)
        dimensions_to_perturb = np.random.choice(len(neighbour), num_dimensions_to_perturb, replace=False)

        for index in dimensions_to_perturb:
            perturbation = np.random.uniform(-32.768, 32.768)
            neighbour[index] = perturbation

        neighbours.append(neighbour)

    return neighbours


def best_neighbour(neighbours, tabu_list):
    best_neighbour = None
    best_value = float("inf")
    for neighbour in neighbours:
        # convert the numpy array to a tuple for comparison
        neighbour_tuple = tuple(neighbour)

        # make sure that this move is not in the tabu list
        if neighbour_tuple not in tabu_list:
            neighbour_value = ackley(neighbour)
            if neighbour_value < best_value:
                best_value = neighbour_value
                best_neighbour = neighbour
    return best_neighbour

def add_tabu_list(tabu_list, solution):
    if len(tabu_list) > 20:
        tabu_list.pop(0)
    solution_tuple = tuple(solution)
    tabu_list.append(solution_tuple)

def tabu_search(solution, max_iteration=100):
    tabu_list = []
    best_solution = solution

    for _ in range(max_iteration):
        solution_value = ackley(best_solution)
        solution_neighbours = get_neighbours(best_solution)
        Best_neighbour = best_neighbour(solution_neighbours, tabu_list)
        neighbour_value = ackley(Best_neighbour) if Best_neighbour is not None else float("inf")
        if neighbour_value < solution_value:
            best_solution = Best_neighbour
        add_tabu_list(tabu_list, Best_neighbour)
    total_cost = ackley(best_solution)
    return best_solution, total_cost


def acceptance_probability(cost, new_cost, T):
    if new_cost < cost:
        return 1.0
    return math.exp((cost - new_cost) / T)

def best_neighbour_(neighbours, temperature):
    current_solution = neighbours[0]
    current_fitness = ackley(current_solution)

    for neighbour in neighbours:
        neighbour_fitness = ackley(neighbour)
        if neighbour_fitness < current_fitness:
            current_solution = neighbour
            current_fitness = neighbour_fitness
        else:
            # Calculate the acceptance probability
            acceptance_prob = np.exp((current_fitness - neighbour_fitness) / temperature)
            if np.random.rand() < acceptance_prob:
                current_solution = neighbour
                current_fitness = neighbour_fitness

    return current_solution

def simulated_annealing(solution, T_min=0.1, cooling_rate=0.9, T=1000):
    current_solution = solution
    current_cost = ackley(current_solution)
    best_solution = current_solution
    best_cost = current_cost

    while T > T_min:
        neighbours = get_neighbours(solution)
        #new_solution = best_neighbour_(neighbours, T)
        new_solution = random.choice(neighbours)
        new_cost = ackley(new_solution)

        if acceptance_probability(current_cost, new_cost, T) > random.random():
            current_solution = new_solution
            current_cost = new_cost

        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        T *= cooling_rate

    return best_solution, best_cost

class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = np.copy(position)
        self.best_fitness = ackley(position)

class PSO:
    def __init__(self, num_particles, dimensions, max_iterations):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.max_iterations = max_iterations
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = np.random.uniform(low=-32.768, high=32.768, size=self.dimensions)
            particle = Particle(position)
            particles.append(particle)
        return particles

    def update_particle_velocity(self, particle, inertia_weight, cognitive_weight, social_weight):
        r1 = np.random.rand(self.dimensions)
        r2 = np.random.rand(self.dimensions)

        cognitive_component = cognitive_weight * r1 * (particle.best_position - particle.position)
        social_component = social_weight * r2 * (self.global_best_position - particle.position)
        particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component

    def update_particle_position(self, particle):
        particle.position = particle.position + particle.velocity

    def update_global_best(self, particles):
        for particle in particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = np.copy(particle.best_position)

    def run(self):
        particles = self.initialize_particles()
        self.global_best_position = particles[0].position

        for _ in range(self.max_iterations):
            for particle in particles:
                self.update_particle_velocity(particle, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4)
                self.update_particle_position(particle)

                fitness = ackley(particle.position)
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

            self.update_global_best(particles)

        return self.global_best_position, self.global_best_fitness





if __name__ == '__main__':
    # Define the number of dimensions for the Ackley function
    n_dimensions = 10

    # Generate a random initial solution within the typical bounds
    initial_solution = np.random.uniform(-32.768, 32.768, n_dimensions)

    # Call the tabu_search function with the initial solution
    best_solution, best_fitness = tabu_search(initial_solution)
    print("for Tabu search: ")
    print("Best solution found:", best_solution)
    print("Best fitness value:", best_fitness)

    best_solution, best_fitness = tabu_search(initial_solution)
    #best_solution_, best_fitness_ = simulated_annealing(initial_solution)
    print("for Simulated anneling: ")
    print("Best solution found:", best_solution)
    print("Best fitness value:", best_fitness)

    pso = PSO(num_particles=50, dimensions=10, max_iterations=100)
    best_position, best_fitness = pso.run()
    print("for PSO: ")
    print("Best solution found:", best_position)
    print("Best fitness value:", best_fitness)

