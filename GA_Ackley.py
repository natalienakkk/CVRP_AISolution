import numpy as np

def ackley(x):
    n = len(x)
    sum1 = -0.2 * np.sqrt(np.sum([i**2 for i in x]) / n)
    sum2 = np.sum([np.cos(2 * np.pi * i) for i in x]) / n
    result = 20 + np.e - 20 * np.exp(sum1) - np.exp(sum2)
    return result

class GeneticAlgorithm:
    def __init__(self, population_size, num_dimensions, crossover_rate=0.8, mutation_rate=0.1, elitism_ratio=0.1):
        self.population_size = population_size
        self.num_dimensions = num_dimensions
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio

    def initialize_population(self):
        return np.random.uniform(low=-32.768, high=32.768, size=(self.population_size, self.num_dimensions))

    def selection(self, population, fitness):
        # Rank-based selection
        ranks = np.argsort(fitness)
        num_elites = int(self.elitism_ratio * self.population_size)
        selected_indices = ranks[:num_elites]

        return population[selected_indices]

    def crossover(self, parents):
        children = []

        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]

            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.num_dimensions)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1 = parent1
                child2 = parent2

            children.extend([child1, child2])

        return np.array(children)

    def mutation(self, population):
        for i in range(len(population)):
            for j in range(self.num_dimensions):
                if np.random.rand() < self.mutation_rate:
                    population[i, j] = np.random.uniform(low=-32.768, high=32.768)

        return population

    def run(self, max_iterations):
        population = self.initialize_population()

        for _ in range(max_iterations):
            fitness = np.array([ackley(individual) for individual in population])

            selected_parents = self.selection(population, fitness)
            children = self.crossover(selected_parents)
            mutated_children = self.mutation(children)

            population = np.concatenate((selected_parents, mutated_children))

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        return best_individual, best_fitness


# Example usage
ga = GeneticAlgorithm(population_size=300, num_dimensions=10)
best_individual, best_fitness = ga.run(max_iterations=5000)
print("Best solution found:", best_individual)
print("Best fitness value:", best_fitness)