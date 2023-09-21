from random import randint
import random
import numpy as np
import math
import re
import time


MUTATIONRATE = 0.5
MIGRATION_RATE = 10
MIGRATION_NUM = 10


class City:
    def __init__(self, city_id, coordinate, demand):
        self.coordinate = coordinate
        self.demand = demand
        self.city_id = city_id

class data:
    def __init__(self, cities, routes):
        self.cities = cities
        self.routes = routes
        self.fitness = -1

class islands:
    def __init__(self, POPSIZE, MUTATION, elitism_size, cities, capacity,storage):
        self.POPSIZE = POPSIZE
        self.ELITISM_SIZE = elitism_size*POPSIZE
        self.population = []
        self.new_population = []
        self.MUTATION_RATE = MUTATION
        self.cities = cities
        self.capacity = capacity
        self.distances = {}
        self.storage = storage

    def generate_clarke_wright_routes(self, perturb_factor=0.05):
        depot = self.storage
        unvisited_cities = self.cities.copy()
        savings = {}  # (i, j): saving value for all i, j

        for city1 in unvisited_cities:
            for city2 in unvisited_cities:
                if city1 != city2:
                    saving = self.distances[(depot.city_id, city1.city_id)] + self.distances[
                        (depot.city_id, city2.city_id)] - self.distances[(city1.city_id, city2.city_id)]
                    savings[(city1, city2)] = saving

        # Perturb savings
        perturbed_savings = {k: v * (1 + random.uniform(-perturb_factor, perturb_factor)) for k, v in savings.items()}

        # Sort savings in descending order
        sorted_savings = sorted(perturbed_savings.items(), key=lambda x: x[1], reverse=True)


        routes = []
        while unvisited_cities and sorted_savings:
            city1, city2 = sorted_savings.pop(0)[0]
            if city1 in unvisited_cities and city2 in unvisited_cities:
                current_route = [depot, city1, city2, depot]
                current_demand = city1.demand + city2.demand
                unvisited_cities.remove(city1)
                unvisited_cities.remove(city2)

                while True:
                    city_to_add = None
                    for city in unvisited_cities:
                        if current_demand + city.demand <= self.capacity:
                            saving1 = self.distances[(depot.city_id, city.city_id)] - self.distances[
                                (current_route[-2].city_id, city.city_id)]
                            saving2 = self.distances[(depot.city_id, city.city_id)] - self.distances[
                                (current_route[1].city_id, city.city_id)]

                            if saving1 > saving2 and saving1 > 0:
                                city_to_add = (city, -2)
                            elif saving2 > saving1 and saving2 > 0:
                                city_to_add = (city, 1)

                            if city_to_add:
                                break

                    if city_to_add:
                        city, position = city_to_add
                        current_route.insert(position, city)
                        current_demand += city.demand
                        unvisited_cities.remove(city)
                    else:
                        break

                routes.append(current_route)

        return routes

    def init_population(self):
        for i in range(self.POPSIZE):
            routes = self.generate_clarke_wright_routes()
            individual = data(self.cities, routes)
            self.population.append(individual)
        # for individual in self.population:
        #     print("individual: ")
        #     for route in individual.routes:
        #         print("route:")
        #         for city in route:
        #             print(city.city_id)



    # euclidian distances between all cities
    def calculate_distance(self, coordinates):
        distances = {}  # dictionary {(i,j): distance value} for all i,j
        for i in coordinates:
            for j in coordinates:
                if i != j and (j, i) not in distances:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    distances[(i, j)] = distance
                    distances[(j, i)] = distance

        self.distances = distances

    def total_route_distnace(self, route):
        distance = 0
        for i in range(len(route) - 1):
            city1_id = route[i].city_id
            city2_id = route[i + 1].city_id
            distance += self.distances[(city1_id, city2_id)]
        return distance

    def fitness(self):
        for individual in self.population:
            total_distance = 0
            for route in individual.routes:
                total_distance += self.total_route_distnace(route)
            individual.fitness = total_distance

    def fitness_sort(self, x):
        return x.fitness

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def elitism(self):
        for i in range(int(self.ELITISM_SIZE)):
            self.new_population.append(self.population[i])

    def swap(self):
        self.population = self.new_population
        self.new_population=[]

    def find_index(self, city, route):
        for i in range(len(route)):
            if route[i].city_id == city.city_id:
                return i
        return -1


    def PMX(self, parent1, parent2):
        # Select random routes from each parent
        parent1_route = random.choice(parent1.routes)
        parent2_route = random.choice(parent2.routes)

        # Remove the storage (depot) from the beginning and end of each route
        parent1_route = parent1_route[1:-1]
        parent2_route = parent2_route[1:-1]
        min_len=min(len(parent1_route),len(parent2_route))
        # Get a random index and values at that index for both parents
        random_index = random.randint(0,min_len - 1)
        value1 = parent1_route[random_index]
        value2 = parent2_route[random_index]

        # Swap the values in the parent routes
        parent1_route[random_index] = value2
        parent2_route[random_index] = value1

        # Find the positions of the swapped values and swap them as well
        index1 = self.find_index(value1, parent1_route)
        index2 = self.find_index(value2, parent2_route)

        if index1 != -1 and index1 != random_index:
            parent1_route[index1] = value2

        if index2 != -1 and index2 != random_index:
            parent2_route[index2] = value1

        # Reconstruct the child routes while maintaining the depot at the beginning and end of each route
        child1_routes = [route if route != parent1_route else [self.storage] + parent1_route + [self.storage] for route
                         in parent1.routes]
        child2_routes = [route if route != parent2_route else [self.storage] + parent2_route + [self.storage] for route
                         in parent2.routes]

        child1 = data(parent1.cities, child1_routes)
        child2 = data(parent2.cities, child2_routes)

        return child1, child2


    def scramble_mutation(self, individual):
        mutated_routes = []
        for route in individual.routes:
            # Remove the storage (depot) from the beginning and end of the route
            route_no_depot = route[1:-1]
            if len(route_no_depot) > 2 and random.random() < self.MUTATION_RATE :
                # Choose two random positions within the route
                pos1, pos2 = random.sample(range(len(route_no_depot)-1), 2)
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1

                # Scramble the sublist between the two positions
                sublist = route_no_depot[pos1:pos2]
                random.shuffle(sublist)
                route_no_depot[pos1:pos2] = sublist
                # Add the storage (depot) back to the beginning and end of the route
                mutated_route = [self.storage] + route_no_depot + [self.storage]
            else:
                mutated_route = route
            mutated_routes.append(mutated_route)
        mutated_individual = data(individual.cities, mutated_routes)
        return mutated_individual


    def genetic_algorithm(self):
        self.elitism()
        while len(self.new_population) < self.POPSIZE:

            parent1_index = randint(0, self.POPSIZE//2)
            parent2_index = randint(0, self.POPSIZE//2)
            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]


            child1,child2 = self.PMX(parent1,parent2)
            #child1, child2 = self.OX(parent1, parent2)
            mutated_child1 = self.scramble_mutation(child1)
            mutated_child2 = self.scramble_mutation(child2)

            self.new_population.append(mutated_child1)
            self.new_population.append(mutated_child2)

    def print_best_solution(self):
        self.sort_by_fitness()
        best_individual = self.population[0]
        print("Best Solution:")
        for route_idx, route in enumerate(best_individual.routes):
            print(f"Route {route_idx + 1}: ", end=" ")
            for city in route:
                print(city.city_id, end=" ")
            print()

        print("Fitness (Total Distance):", best_individual.fitness)


def migration(f, g, selection_type, immg_num):
    if selection_type == "Random":
        migrants = Random(g.population, immg_num)
    elif selection_type == "RWS":
        migrants = RWS(g.population, immg_num)
    updated_population = replacement(f.population, migrants)
    f.population = updated_population


def replacement(population, immigrants):
    population.sort(key=lambda x: x.fitness)
    j = 0
    for i in range(len(population)):
        if j >= len(immigrants):break
        population[i] = immigrants[j]
        j += 1
    return population

def Random(population, immg_num):
    return random.sample(population, immg_num)



def RWS(population, immg_num):
    fitnesses = sum(individual.fitness for individual in population)
    probabilities = [individual.fitness / fitnesses for individual in population]
    migrants = random.choices(population, weights=probabilities, k=immg_num)
    return migrants

if __name__ == '__main__':
    with open("example1.txt", "r") as f:
        lines = f.readlines()

    # get population size
    dimension = None
    for line in lines:
        if line.startswith("DIMENSION : "):
            dimension = int(re.search(r'\d+', line).group())
            break

    capacity = None
    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = int(re.search(r'\d+', line).group())
            break

    # Extract node coordinates from NODE_COORD_SECTION
    coordinates = {}
    for i in range(lines.index("NODE_COORD_SECTION\n") + 1, lines.index("DEMAND_SECTION\n")):
        node_id, x, y = map(int, lines[i].strip().split())
        coordinates[node_id] = (x, y)

    # Extract demands from DEMAND_SECTION
    demands = {}
    for i in range(lines.index("DEMAND_SECTION\n") + 1, lines.index("DEPOT_SECTION\n")):
        node_id, demand = map(int, lines[i].strip().split())
        demands[node_id] = demand

    # create an instance of city and add it to the population
    cities = []
    for i in range(1, len(demands) + 1):
        coordinate = np.zeros(2)
        coordinate[0] = coordinates[i][0]
        coordinate[1] = coordinates[i][1]
        city = City(i, coordinate, demands[i])
        cities.append(city)
    storage = cities[0]
    cities.remove(storage)

    cvrp1 = islands(300, 0.5, 0.2, cities, capacity, storage)
    cvrp2 = islands(300, 0.5, 0.2, cities, capacity, storage)
    max_generation = 100
    cvrp1.calculate_distance(coordinates)
    cvrp1.init_population()
    cvrp2.calculate_distance(coordinates)
    cvrp2.init_population()
    for i in range(max_generation):
        start_clockticks = time.perf_counter()
        start_absolute = time.time()
        cvrp1.fitness()
        cvrp1.genetic_algorithm()
        cvrp1.print_best_solution()
        cvrp1.swap()

        cvrp2.fitness()
        cvrp2.genetic_algorithm()
        cvrp2.print_best_solution()
        cvrp2.swap()

        if i % MIGRATION_RATE == 0:
            migration(cvrp1, cvrp2, "Random", MIGRATION_NUM)
            migration(cvrp2, cvrp1, "Random", MIGRATION_NUM)

end_absolute = time.time()
end_clockticks = time.perf_counter()
print("CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
    end_absolute - start_absolute))

