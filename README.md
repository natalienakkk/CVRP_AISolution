# CVRP_AISolution
AI-driven solution for the Capacitated Vehicle Routing Problem.

Background:

The Capacitated Vehicle Routing Problem (CVRP) is a classic combinatorial optimization problem in the field of operations research. In CVRP, a fleet of vehicles is tasked with delivering goods to a set of customers in such a way that:

1.Each customer's demand is satisfied.

2.The total demand a vehicle carries does not exceed its capacity.

3.The overall cost (often distance or time) is minimized.

Our approach to solving the CVRP is unique. We leverage clustering techniques to group cities based on their geographical locations. Once cities are grouped, we ensure that the total demand within each cluster does not exceed vehicle capacity. If a cluster's demand is too high, we reassign cities to neighboring clusters. After forming balanced clusters, we then employ various metaheuristic algorithms to determine the optimal route within each cluster.

Approach:


1.Clustering Cities: Cities are first separated into clusters based on their geographical locations using clustering techniques.

2.Balancing Demand: For clusters where the demand exceeds vehicle capacity, cities are moved to the closest cluster. This ensures that the total demand in any given cluster does not exceed the vehicle's capacity.

3.Route Optimization: Once clusters are formed and balanced, we use different metaheuristic algorithms to find the best routes within each cluster. The algorithms used include:

3.1.Tabu Search

3.2.Ant Colony Optimization (ACO)

3.3.Simulated Annealing

4. Another approach : there is another 2 suggested approaches ALNS and genetic algorithm .



Results:

After applying the above approach, routes are generated for each cluster, ensuring that the overall cost is minimized. Detailed results, including the routes taken and the associated costs, are provided.


Running the Project:

There are 5 input examples : example1,example2,example3,example4,example5 - each example contains input parameters like capacity , expected results , each city details etc.

Choose the input file from the 5 given and Run CVRP_Solution.py by writing the name of the input file you choose in the main function. 

For ALNS approach for solving CVRP run file ALNS_CVRP.py exactly according to how is explained earlier.
For Genetic Algorithm approach just  run file CVRP_GA2.py exactly according to how is explained earlier.
