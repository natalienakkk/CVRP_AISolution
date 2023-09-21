import random
import numpy as np
import math
import re
import copy
from sklearn.cluster import KMeans
import time
alpha = 10


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
        self.ALNS_routes = {}


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


    def ALNS(self, storage):
        max_iterations = 100
        cost = self.calculate_cost(self.routes)
        self.ALNS_routes = self.routes
        # Perform the ALNS iterations
        for iteration in range(max_iterations):
            # Destroy phase: Select a subset of the solution to destroy
            destroyed_solution = self.destroy()

            # Repair phase: Repair the destroyed solution to create a feasible solution
            repaired_solution = self.repair(destroyed_solution)

            # Evaluate the repaired solution
            repaired_cost = self.calculate_cost(repaired_solution)

            # Update the best solution based on the acceptance criterion
            if self.acceptance_criterion(repaired_solution):
                self.ALNS_routes = repaired_solution

            if repaired_cost < cost:
                self.ALNS_routes = repaired_solution
                cost = repaired_cost

        # Add the storage city at the beginning and end of each route
        for route_name, route in self.ALNS_routes.items():
            route.insert(0, storage)
            route.append(storage)
        for route_name,route in self.ALNS_routes.items():
            self.ALNS_routes[route_name]=self.apply_3_opt(route)

        # Calculate the total cost of the final solution
        total_cost = self.calculate_cost(self.ALNS_routes)

        return total_cost


    def destroy(self):
        # Randomly select destroy operator
        operator = random.choice(['remove_customer', 'swap_customers'])
        if operator == 'remove_customer':
            # Remove a random customer from the solution
            destroyed_solution = copy.deepcopy(self.ALNS_routes)
            route = random.choice(list(destroyed_solution.values()))
            if len(route) > 3:
                customer = random.choice(route[1:-1])
                route.remove(customer)

        elif operator == 'swap_customers':
            # Swap two random customers between routes
            destroyed_solution = copy.deepcopy(self.ALNS_routes)
            route1 = random.choice(list(destroyed_solution.values()))
            route2 = random.choice(list(destroyed_solution.values()))
            if len(route1) > 3 and len(route2) > 3:
                customer1 = random.choice(route1[1:-1])
                customer2 = random.choice(route2[1:-1])
                route1[route1.index(customer1)] = customer2
                route2[route2.index(customer2)] = customer1
        return destroyed_solution

    def repair(self, destroyed_solution):
        # Create a set of all cities
        all_cities = set(city for route in destroyed_solution.values() for city in route)

        # Add missing cities back to each route
        repaired_solution = {}
        for route_name, route in destroyed_solution.items():
            missing_cities = all_cities - set(route)
            repaired_route = route + list(missing_cities)
            repaired_solution[route_name] = repaired_route

        return repaired_solution

    def acceptance_criterion(self, repaired_cost):
        current_cost = self.calculate_cost(self.ALNS_routes)
        new_cost = self.calculate_cost(repaired_cost)

        # Implement the acceptance criterion
        if new_cost < current_cost:
            return True
        else:
            return random.random() < math.exp(-(new_cost - current_cost) / alpha)


    def calculate_cost(self, solution):
        total_distance = 0
        for route in solution.values():
            for i in range(len(route) - 1):
                city1_id = route[i].city_id
                city2_id = route[i + 1].city_id
                total_distance += self.distances[(city1_id, city2_id)]

        return total_distance


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
    start_clockticks = time.perf_counter()
    start_absolute = time.time()
    best_alns_cost = cvrp.ALNS(storage)

    print("ALNS results: ", end=' ')
    print()
    for _,best_route in cvrp.ALNS_routes.items():
        print("[", end=' ')
        for city in best_route:
            print(city.city_id, end=' ')
        print("]")
    print("ALNS total cost= ", best_alns_cost-50)
    print()

    end_absolute = time.time()
    end_clockticks = time.perf_counter()
    print("CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
        end_absolute - start_absolute))

