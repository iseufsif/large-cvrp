import random

def random_removal(solution, num_remove):
    # save all customers
    flat_customers = [cust for route in solution for cust in route]

    # randomly select customers to remove -- num_remove = number of customers you want to remove
    to_remove = set(random.sample(flat_customers, num_remove))
    
    partial_solution = []
    removed_customers = []

    for route in solution:
        # rebuild partial routes
        new_route = [cust for cust in route if cust not in to_remove]
        removed = [cust for cust in route if cust in to_remove]
        if new_route:
            partial_solution.append(new_route)
        removed_customers.extend(removed)

    return partial_solution, removed_customers
