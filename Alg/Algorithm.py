import numpy as np
import random

class AntColonyOptimizer:
    def __init__(self, num_ants, num_cities, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        """
        Args:
            num_ants (int): Number of ants in the colony.
            num_cities (int): Total number of cities.
            alpha (float): Importance of Pheromone (History).
            beta (float): Importance of Heuristic (Distance/Cost).
            evaporation_rate (float): How fast pheromones disappear (0.0 - 1.0).
        """
        self.num_ants = num_ants
        self.num_cities = num_cities
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        
        # Pheromone Matrix: Starts with 1.0 everywhere
        self.pheromone = np.ones((num_cities, num_cities))

    def solve():
        # TODO: do the solve algorithm
        return None

    def _calculate_heuristic(self, dist_mat, vel_mat, traffic_mat, consum_mat, time_weight):
        """
        Combines Time and Fuel into a single probability factor (Eta) to find the wanted path.
        High Heuristic = Very Desirable Road.
        """
        fuel_weight = 1.0 - time_weight

        # Fuel Price 
        fuel = 20

        # Calculate Formulas
        time_mat = (dist_mat / vel_mat) + traffic_mat
        cost_mat = consum_mat * dist_mat * fuel
                
        # Normalize matrices to 0-1 range so one doesn't dominate the other
        # Add epsilon (1e-9) to avoid division by zero
        norm_time = time_mat / (time_mat.max() + 1e-9)
        norm_fuel = cost_mat / (cost_mat.max() + 1e-9)

        # Weighted Cost
        weighted_cost = (norm_time * time_weight) + (norm_fuel * fuel_weight)
        
        # Heuristic is 1 / Cost (because ants like Low Cost)
        # We add a tiny number to avoid 1/0 errors
        return 1.0 / (weighted_cost + 1e-9)

    def _construct_path(self, heuristic_matrix):
        """
        A single ant walks through the graph.
        """
        # Start at a random city
        start_node = random.randint(0, self.num_cities - 1)
        path = [start_node]
        visited = set(path)

        for _ in range(self.num_cities - 1):
            current = path[-1]
            
            # Calculate Probabilities for all neighbors
            probs = self._calculate_probabilities(current, visited, heuristic_matrix) #heuristic_matrix is the matrix from the heuristic function
            
            # Roulette Wheel Selection (Pick next city based on probability)
            next_city = self._roulette_wheel_selection(probs)
            
            path.append(next_city)
            visited.add(next_city)

        # Return to start (Depot) to close the loop (byrg3 lnfs elmakan tany 3ashan myb2ash one-way trip)
        path.append(start_node)
        return path

    def _calculate_probabilities(self, current_city, visited, heuristic_matrix):
        """
        ACO Probability Formula: P = (Pheromone^alpha) * (Heuristic^beta)
                                      ______tau______     _____eta______
        """
        probabilities = np.zeros(self.num_cities)
        
        for city in range(self.num_cities):
            if city in visited:
                probabilities[city] = 0.0 # Can't visit twice
            else:
                tau = self.pheromone[current_city][city] ** self.alpha
                eta = heuristic_matrix[current_city][city] ** self.beta
                probabilities[city] = tau * eta
        
        # Normalize probabilities to sum to 1
        total = probabilities.sum()
        if total == 0:
            # If ant is stuck (shouldn't happen in fully connected graph), pick random unvisited
            unvisited = [c for c in range(self.num_cities) if c not in visited]
            probs = np.zeros(self.num_cities)
            for u in unvisited:
                probs[u] = 1.0 / len(unvisited)
            return probs
            
        return probabilities / total

    def _roulette_wheel_selection(self, probabilities):
        """
        Picks an index based on the probability array.
        """
        return np.random.choice(range(self.num_cities), p=probabilities)

    def _deposit_pheromone(self, path, cost):
        """
        Adds pheromone to the edges in the path.
        Amount = 1.0 / Cost
        """
        deposit_amount = 1.0 / (cost + 1e-9)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.pheromone[u][v] += deposit_amount
            self.pheromone[v][u] += deposit_amount # Symmetric graph

    def _evaluate_path_cost(self, path, time_mat, fuel_mat, time_w):
        # TODO: Calculates total cost of a specific path
        return cost

# --- TEST BLOCK (For Member A/B to verify it works) ---
if __name__ == "__main__":
    # Fake Data (provided by Member E)
    num_cities = 5
    
    # 1. Create Fake Time Matrix (0-10 hours)
    time_data = np.random.rand(num_cities, num_cities) * 10 
    np.fill_diagonal(time_data, 0) # Distance to self is 0

    # 2. Create Fake Fuel Matrix (0-100 liters)
    fuel_data = np.random.rand(num_cities, num_cities) * 100
    np.fill_diagonal(fuel_data, 0)

    # 3. Initialize ACO
    aco = AntColonyOptimizer(num_ants=10, num_cities=num_cities)
    
    # 4. Run with High Importance on TIME (0.9)
    print("Optimizing for TIME...")
    path, cost = aco.solve(time_data, fuel_data, time_importance=0.9)
    print(f"Best Path: {path}")
    print(f"Weighted Cost: {cost:.2f}")

    # 5. Run with High Importance on FUEL (0.1)
    print("\nOptimizing for FUEL...")
    path, cost = aco.solve(time_data, fuel_data, time_importance=0.1)
    print(f"Best Path: {path}")
    print(f"Weighted Cost: {cost:.2f}")