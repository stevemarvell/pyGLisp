import random

OPERATIONS = ['+', '-', '*', '/']

MAX_DEPTH = 5  # > 0

POPULATION_SIZE = 1000
NUM_GENERATIONS = 500

INPUTS = [{'a': random.randint(0, 9), 'b': random.randint(0, 9), 'c': random.randint(0, 9)} for i in range(100)]

OUTPUTS = [3 * i['a'] + 2 * i['b'] + i['a'] * i['c'] for i in INPUTS]  # for instance

# allowed variable names
VARIABLES = list(set(var for input_set in INPUTS for var in input_set.keys()))


def generate_random_expression(depth):
    if depth == 1:
        return random.choice(VARIABLES)
    else:
        op = random.choice(OPERATIONS)
        if op == '/':
            # Ensure the denominator is not '0' or another variable
            return [op, generate_random_expression(depth - 1), random.randint(1, 9)]
        else:
            return [op, generate_random_expression(depth - 1), generate_random_expression(depth - 1)]


class DivisionByZeroError(Exception):
    pass

def evaluate_expression(expression, inputs, depth=0):
    if depth > MAX_DEPTH:
        print("Maximum depth exceeded")
        return 0  # Return 0 if maximum depth is exceeded @todo handle this better

    if isinstance(expression, str):
        return int(inputs[expression])
    elif isinstance(expression, int):
        return expression
    else:
        op = expression[0]
        left_operand = evaluate_expression(expression[1], inputs, depth + 1)
        right_operand = evaluate_expression(expression[2], inputs, depth + 1)
        if op == '+':
            return left_operand + right_operand
        elif op == '-':
            return left_operand - right_operand
        elif op == '*':
            return left_operand * right_operand
        elif op == '/':
            return left_operand / right_operand

def fitness(expression):
    # @todo reward shorter
    total_score = 0
    for input_set, output in zip(INPUTS, OUTPUTS):
        try:
            result = evaluate_expression(expression, input_set)
            total_score += abs(result - output)
        except ZeroDivisionError:
            total_score += 1000  # Penalize divisions by zero heavily
    return total_score


def crossover(parent1, parent2):
    if len(parent1) <= 1 or len(parent2) <= 1:
        # If either parent has a length less than or equal to 1, return the parents themselves
        return parent1, parent2
    else:
        split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2


def mutate(expression, at_depth=1, mutation_rate=0.2):

    # @todo add reduction
    if random.random() >= mutation_rate:
        return expression

    if at_depth == MAX_DEPTH:
        if isinstance(expression, int):
            return random.randint(1, 9)
        else:
            return random.choice(VARIABLES)

    op = random.choice(OPERATIONS)

    if random.random() < 0.1:
        return [op, generate_random_expression(random.randint(1, MAX_DEPTH - at_depth)),
                generate_random_expression(random.randint(1, MAX_DEPTH - at_depth))]

    if isinstance(expression, int):
        if random.random() < 0.5:
            return [op, random.choice(VARIABLES), expression]
        else:
            return [op, expression, random.choice(VARIABLES)]

    if isinstance(expression, str):
        if random.random() < 0.5:
            return [op, random.randint(1, 9), expression]
        else:
            return [op, expression, random.randint(1, 9)]

    # it's a complex one

    [op, left, right] = expression

    if random.random() < 0.1:
        op = random.choice(OPERATIONS)

        if op == '/':
            return [op, left, random.randint(1, 9)]

        return [op, left, right]

    return [op, mutate(left, at_depth + 1), mutate(right, at_depth + 1)]


def generate_initial_population(population_size):
    return [generate_random_expression(random.randint(2, MAX_DEPTH)) for _ in range(population_size)]


def evolve(population):
    new_population = []
    # Elitism
    sorted_population = sorted(population, key=lambda x: fitness(x))
    new_population.append(sorted_population[0])

    # reduce selection pressure
    fitness_values = [fitness(x) + 10 for x in population]

    while len(new_population) < len(population):
        parent1, parent2 = random.choices(population, k=2, weights=[1 / f for f in fitness_values])
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    # @todo go and prune things below max depth
    return new_population


def format_expression(expression):
    if isinstance(expression, str) or isinstance(expression, int):
        return str(expression)

    op = expression[0]
    left = format_expression(expression[1])
    right = format_expression(expression[2])
    return f"({left} {op} {right})"


def main():
    population = generate_initial_population(POPULATION_SIZE)
    for generation in range(NUM_GENERATIONS):
        population = evolve(population)
        best_individual = min(population, key=lambda x: fitness(x))
        print(
            f"Generation {generation + 1}, Best Individual: {format_expression(best_individual)}, Fitness: {fitness(best_individual)}")


if __name__ == "__main__":
    main()
