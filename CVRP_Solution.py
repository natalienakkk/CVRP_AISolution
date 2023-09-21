import random
import numpy as np
import math
import re
import copy
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
alpha = 1


class Cluster:
    def __init__(self, centroid, demand):
        self.centroid = centroid
        self.demand = demand
        self.cities = []

    def distance_to(self, city):
        x1, y1 = self.centroid.coordinate
        x2, y2 = city.coordinate
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance_to_2(self, city_coordinate):
        x1, y1 = self.centroid.coordinate
        x2, y2 = city_coordinate
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class City:
    def __init__(self, city_id, coordinate, demand):
        self.coordinate = coordinate
        self.demand = demand
        self.city_id = city_id


class CVRP:
    def __init__(self, POPSIZE, capacity):
        self.population = []
        self.POPSIZE = POPSIZE
        self.capacity = capacity
        self.distances = {}
        self.clusters = []
        self.routes = {}

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


    def k_means_clustering(self, n_clusters):
        population_array = np.array([city.coordinate for city in self.population])
        population_array = np.delete(population_array, 0, axis=0)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300)
        kmeans.fit(population_array)

        clusters = [Cluster(City(0, centroid, 0), 0) for centroid in kmeans.cluster_centers_]
        #create clusters
        for city in self.population[1:]:
            label = kmeans.predict([city.coordinate])[0]
            clusters[label].cities.append(city)
            clusters[label].demand += city.demand

        return clusters

    def adjust_cluster_demand(self, city, original_cluster):
        sorted_clusters = sorted(self.clusters,
                                 key=lambda cluster: (cluster.demand, cluster.distance_to(city)))

        for cluster in sorted_clusters:
            if cluster.demand + city.demand <= self.capacity:
                cluster.cities.append(city)
                cluster.demand += city.demand
                original_cluster.cities.remove(city)
                original_cluster.demand -= city.demand
                break

        else:
            sorted_clusters = sorted(self.clusters, key=lambda cluster: (
            self.capacity - cluster.demand, cluster.distance_to(city)))

            for cluster in sorted_clusters:
                if cluster.demand + city.demand <= self.capacity:
                    cluster.cities.append(city)
                    cluster.demand += city.demand
                    original_cluster.cities.remove(city)
                    original_cluster.demand -= city.demand
                    break


    def generate_solution(self):
        n_clusters = 4
        self.clusters = self.k_means_clustering(n_clusters)
        for cluster in self.clusters:
            if cluster.demand > self.capacity:
                for city in cluster.cities:
                    self.adjust_cluster_demand(city,cluster)
                    if cluster.demand <= self.capacity:break

        routes_dict = {}
        for i, cluster in enumerate(self.clusters):
            route_name = f"route{i + 1}"
            routes_dict[route_name] = [city for city in cluster.cities]

        self.routes = routes_dict


    # get total distnace of the route by iterating over the cities according to original order
    def total_route_distnace(self, route):
        distance = 0
        for i in range(len(route) - 1):
            city1_id = route[i].city_id
            city2_id = route[i + 1].city_id
            distance += self.distances[(city1_id, city2_id)]
        return distance

    # create neighbours by find all the possibale swaps between cities in the route
    def get_neighbours(self, route):
        neighbours = []
        if len(route) == 3:
            neighbour = copy.deepcopy(route)
            neighbours.append(neighbour)
            return neighbours
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route) - 1):
                neighbour = copy.deepcopy(route)
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                neighbours.append(neighbour)
        return neighbours

    # find best neighbour by iterating on all neighbours and find minimum route distance
    def best_neighbour(self, neighbours, tabu_list):
        best_neighbour = None
        best_distance = float("inf")
        for neighbour in neighbours:
            # make sure that this move is not in the tabu list
            if neighbour is not tabu_list:
                neighbour_distance = self.total_route_distnace(neighbour)
                if neighbour_distance < best_distance:
                    best_distance = neighbour_distance
                    best_neighbour = neighbour
        # if all moves are in tabu_list or all neighbour_distance > best_distance
        if best_neighbour is None and len(neighbours)!=0:
            return random.choice(neighbours)
        return best_neighbour

    def add_tabu_list(self, tabu_list, swap):
        if len(tabu_list) > 20:
            tabu_list.pop(0)
        tabu_list.append(swap)

    def tabu_search(self, route):
        tabu_list = []
        max_iteration = 10
        i = 0
        best_route = route
        iteration_costs = []
        while i < max_iteration:
            i += 1
            route_distance = self.total_route_distnace(best_route)
            route_neighbours = self.get_neighbours(route)
            best_neighbour = self.best_neighbour(route_neighbours, tabu_list)
            if best_neighbour is not None:
                neighbour_distance = self.total_route_distnace(best_neighbour)
            else: neighbour_distance=float("inf")
            if neighbour_distance < route_distance:
                best_route = best_neighbour


            self.add_tabu_list(tabu_list, best_neighbour)
            best_route = self.apply_3_opt(best_route)
            iteration_costs.append(route_distance)
        total_cost = self.total_route_distnace(best_route)
        return best_route, total_cost , iteration_costs

    def apply_3_opt(self, route):
        n = len(route)
        best_route = route
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 3):
                for j in range(i + 2, n - 1):
                    for k in range(j + 2, n):
                        new_route = (
                                best_route[:i] +
                                best_route[i:j][::-1] +
                                best_route[j:k][::-1] +
                                best_route[k:])
                        # Calculate distances
                        current_distance = self.total_route_distnace(best_route)
                        new_distance = self.total_route_distnace(new_route)
                        # Check if the new route improves the distance
                        if new_distance < current_distance:
                            best_route = new_route
                            improved = True
        return best_route

    def heuristic_ACO(self, route):
        n = len(route)
        heuristic_matrix = {}
        for i in range(n):
            city1_id = route[i].city_id
            for j in range(n):
                if i != j:
                    city2_id = route[j].city_id
                    heuristic_matrix[(i, j)] = 1 / self.distances[(city1_id, city2_id)]
        return heuristic_matrix

    def calculate_selection_probability(self, current_city, tabu_list, route, pheromone_matrix, heuristic_matrix,
                                        city_to_index):
        alpha = 1
        beta = 1
        probabilities = {}
        unvisited_cities = set(city for city in route if city not in tabu_list)
        Sum = sum((pheromone_matrix[city_to_index[current_city]][city_to_index[city]] ** alpha) * (
                    heuristic_matrix[city_to_index[current_city], city_to_index[city]] ** beta) for city in
                  unvisited_cities)
        for j in unvisited_cities:
            numer = (pheromone_matrix[city_to_index[current_city]][city_to_index[j]] ** alpha) * (
                        heuristic_matrix[city_to_index[current_city], city_to_index[j]] ** beta)
            probabilities[j] = numer / Sum
        return probabilities

    def update_pheromone(self, route, ants_route, pheromone_matrix):
        # Set evaporation rate and pheromone increase rate
        evaporation_rate = 0.5  # value between 0 and 1 that determines how much pheromone evaporates on each edge after each iteration
        pheromone_increase_rate = 1  # value that determines how much pheromone to add back onto each edge for each ant that traversed

        # Update pheromone levels on each edge
        for i in range(len(pheromone_matrix)):
            for j in range(len(pheromone_matrix)):
                if i != j:
                    # Evaporate pheromone on this edge
                    pheromone_matrix[i][j] *= (1 - evaporation_rate)
                    # Update pheromone on this edge based on ant's tour
                    if (i, j) in ants_route:
                        pheromone_matrix[i][j] += evaporation_rate * (pheromone_increase_rate / len(ants_route))
        return pheromone_matrix

    def ACO(self, route, storage):
        ants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_iteration = 100
        best_route = None
        best_route_distance = float("inf")
        city_to_index = {city_id: i for i, city_id in enumerate(route)}
        pheromone_matrix = np.full((len(route), len(route)), 0.1)
        heuristic_matrix = self.heuristic_ACO(route)
        iteration_costs = []

        for i in range(max_iteration):
            for ant in ants:
                tabu_list_ACO = []
                start = random.choice(route)
                self.add_tabu_list(tabu_list_ACO, start)
                unvisited_cities = set(city for city in route if city not in tabu_list_ACO)
                while unvisited_cities:
                    probabilities = self.calculate_selection_probability(start, tabu_list_ACO, route, pheromone_matrix,
                                                                         heuristic_matrix, city_to_index)
                    max_prob_city = max(probabilities, key=probabilities.get)
                    start = max_prob_city
                    self.add_tabu_list(tabu_list_ACO, start)
                    unvisited_cities = set(city for city in route if city not in tabu_list_ACO)
                ants_route = [city for city in tabu_list_ACO]
                ants_route_distance = self.total_route_distnace(ants_route)
                if ants_route_distance < best_route_distance:
                    best_route_distance = ants_route_distance
                    best_route = ants_route

            pheromone_matrix = self.update_pheromone(route, ants_route, pheromone_matrix)
            best_route_distance += self.distances[(storage.city_id, best_route[0].city_id)] + self.distances[
                (storage.city_id, best_route[-1].city_id)]
            iteration_costs.append(best_route_distance)
        best_route = self.apply_3_opt(best_route)
        return best_route, best_route_distance,iteration_costs

    def acceptance_probability(self, cost, new_cost, T):
        if new_cost < cost:
            return 1.0
        return math.exp((cost - new_cost) / T)

    def histogram(self, iteration_costs):
        iterations = np.arange(len(iteration_costs))
        plt.plot(iterations, iteration_costs, marker='o', linestyle='-', color='b')
        plt.xlabel('Iterations')
        plt.ylabel('Total Cost')
        plt.title('Change of Total Cost')
        plt.show()

    def simulated_annealing(self, route):
        T_min = 0.1
        cooling_rate = 0.9
        T = 1000

        current_route = route
        current_cost = self.total_route_distnace(current_route)
        best_route = current_route
        best_cost = current_cost
        iteration_costs = []

        while T > T_min:
            neighbours = self.get_neighbours(current_route)
            new_route = random.choice(neighbours)
            new_route = self.apply_3_opt(new_route)
            new_cost = self.total_route_distnace(new_route)
            iteration_costs.append(current_cost)

            if self.acceptance_probability(current_cost, new_cost, T) > random.random():
                current_route = new_route
                current_cost = new_cost

            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

            T *= cooling_rate


        return best_route, best_cost ,iteration_costs

    def greedy_tsp(self, cities):
        n = len(cities)
        path = [cities[0].city_id]
        remaining = []
        for i in range(1, n - 1):
            remaining.append(cities[i].city_id)

        total_cost = 0
        while remaining:
            current = path[-1]
            nearest = None
            nearest_dist = float('inf')

            for candidate in remaining:
                dist = self.distances[(current, candidate)]
                if dist < nearest_dist:
                    nearest = candidate
                    nearest_dist = dist

            path.append(nearest)
            remaining.remove(nearest)
            total_cost += nearest_dist

        path.append(cities[0].city_id)
        total_cost += self.distances[(path[1], path[0])] + self.distances[(path[-2], path[-1])]

        return path, total_cost


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

    # create CVRP instance
    cvrp = CVRP(dimension, capacity)

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
    for i in range(1, len(demands) + 1):
        coordinate = np.zeros(2)
        coordinate[0] = coordinates[i][0]
        coordinate[1] = coordinates[i][1]

        city = City(i, coordinate, demands[i])
        cvrp.population.append(city)

    # calculate distances between cities
    cvrp.calculate_distance(coordinates)
    cvrp.generate_solution()

    storage = cvrp.population[0]


    # for cluster in cvrp.clusters:
    #     if cluster:
    #         print("Cluster:")
    #         total_demand_cluster = 0
    #     for city in cluster.cities:
    #         total_demand_cluster += city.demand
    #         print(f"\tCity {city.city_id}: demand={city.demand}")
    #

    tabu_routes_result = []
    greedy_routes_result = []
    aco_routes_result = []
    simulated_annealing_routes_result = []
    Total_TABU = 0
    Total_GREEDY = 0
    Total_ACO = 0
    Total_SA = 0
    total_cost_iter = []
    count = 0
    for _, route in cvrp.routes.items():
        count += 1
        start_clockticks = time.perf_counter()
        start_absolute = time.time()
        best_route_aco, probabilities,iteration_costs = cvrp.ACO(route, storage)
        aco_routes_result.append(best_route_aco)
        Total_ACO += probabilities
        if count == 1:
            total_cost_iter = [0] * len(iteration_costs)
        for i in range(len(iteration_costs)):
            total_cost_iter[i] += iteration_costs[i]
    cvrp.histogram(total_cost_iter)
    end_absolute = time.time()
    end_clockticks = time.perf_counter()
    print("for ACO: "+"CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
        end_absolute - start_absolute))

    total_cost_iter = []
    count = 0
    for _, route in cvrp.routes.items():
        count += 1
        start_clockticks = time.perf_counter()
        start_absolute = time.time()
        route.insert(0, storage)
        route.append(storage)
        best_route, total_cost , iteration_costs = cvrp.tabu_search(route)
        if count == 1:
            total_cost_iter = [0] * len(iteration_costs)
        for i in range(len(iteration_costs)):
            total_cost_iter[i] += iteration_costs[i]
        tabu_routes_result.append(best_route)
        Total_TABU += total_cost
    cvrp.histogram(total_cost_iter)
    end_absolute = time.time()
    end_clockticks = time.perf_counter()
    print("for TABU SEARCH : "+"CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
            end_absolute - start_absolute))

    total_cost_iter = []
    count = 0
    for _, route in cvrp.routes.items():
        count += 1
        start_clockticks = time.perf_counter()
        start_absolute = time.time()
        best_route_SA, total_cost_SA, iteration_costs = cvrp.simulated_annealing(route)
        simulated_annealing_routes_result.append(best_route_SA)
        Total_SA += total_cost_SA
        if count == 1:
            total_cost_iter = [0] * len(iteration_costs)
        for i in range(len(iteration_costs)):
            total_cost_iter[i] += iteration_costs[i]
    cvrp.histogram(total_cost_iter)
    end_absolute = time.time()
    end_clockticks = time.perf_counter()
    print("for SA : "+"CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
            end_absolute - start_absolute))

    # print("Greedy Tsp results: ", end=' ')
    # print()
    # for best_route in greedy_routes_result:
    #     print(best_route, end=' ')
    #     print()
    # print("Greedy TSP total cost= ", Total_GREEDY)
    # print()

    print("Tabu search results: ", end=' ')
    print()
    for best_route in tabu_routes_result:
        print("[", end=' ')
        for city in best_route:
            print(city.city_id, end=' ')
        print("]")
    print("Tabu Search total cost= ", Total_TABU)
    print()

    print("ACO results: ", end=' ')
    print()
    for best_route in aco_routes_result:
        print("[ 1", end=' ')
        for city in best_route:
            print(city.city_id, end=' ')
        print("1 ]")
    print("ACO total cost= ", Total_ACO)
    print()

    print("simulated annealing results: ", end=' ')
    print()
    for best_route in simulated_annealing_routes_result:
        print("[", end=' ')
        for city in best_route:
            print(city.city_id, end=' ')
        print("]")
    print("Simulated Annealing total cost= ", Total_SA)
    print()

