import numpy
import random
from functools import cmp_to_key
import time
import sys

random.seed()

Nd = 9  
global_fitness = None
geracao = None 

class Population(object):
    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []
        
        helper = Candidate()
        helper.values = [[[] for j in range(Nd)] for i in range(Nd)]
        
        for row in range(Nd):
            for column in range(Nd):
                for value in range(1, 10):
                    if given.values[row][column] == 0:
                        if not (given.is_column_duplicate(column, value) 
                               and not given.is_block_duplicate(row, column, value) 
                               and not given.is_row_duplicate(row, value)):
                            helper.values[row][column].append(value)
                    else:
                        helper.values[row][column].append(given.values[row][column])
                        break

        for _ in range(Nc):
            candidate = Candidate()
            for row in range(Nd):
                row_values = []
                for col in range(Nd):
                    if given.values[row][col] != 0:
                        row_values.append(given.values[row][col])
                    else:
                        row_values.append(random.choice(helper.values[row][col]))
                
                while len(set(row_values)) != Nd:
                    for col in range(Nd):
                        if given.values[row][col] == 0:
                            row_values[col] = random.choice(helper.values[row][col])
                candidate.values[row] = row_values
            self.candidates.append(candidate)
        
        self.update_fitness()
        return

    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return

    def sort(self):
        self.candidates.sort(key=cmp_to_key(Population.sort_fitness))
        return

    @staticmethod
    def sort_fitness(x, y):
        if x.fitness < y.fitness:
            return 1
        elif x.fitness == y.fitness:
            return 0
        else:
            return -1

class Candidate(object):
    def __init__(self):
        self.values = numpy.zeros((Nd, Nd), dtype=int)
        self.fitness = None
        return

    def update_fitness(self):
        row_count = numpy.zeros(Nd)
        column_count = numpy.zeros(Nd)
        block_count = numpy.zeros(Nd)
        row_sum = 0
        column_sum = 0
        block_sum = 0

        for i in range(0, Nd):  
            for j in range(0, Nd):  
                row_count[self.values[i][j]-1] += 1 

            row_sum += (1.0/len(set(row_count)))/Nd
            row_count = numpy.zeros(Nd)

        for i in range(0, Nd):  
            for j in range(0, Nd):  
                column_count[self.values[j][i]-1] += 1  

            column_sum += (1.0 / len(set(column_count)))/Nd
            column_count = numpy.zeros(Nd)

        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j]-1] += 1
                block_count[self.values[i][j+1]-1] += 1
                block_count[self.values[i][j+2]-1] += 1
                
                block_count[self.values[i+1][j]-1] += 1
                block_count[self.values[i+1][j+1]-1] += 1
                block_count[self.values[i+1][j+2]-1] += 1
                
                block_count[self.values[i+2][j]-1] += 1
                block_count[self.values[i+2][j+1]-1] += 1
                block_count[self.values[i+2][j+2]-1] += 1

                block_sum += (1.0/len(set(block_count)))/Nd
                block_count = numpy.zeros(Nd)

        if (int(row_sum) == 1 and int(column_sum) == 1 and int(block_sum) == 1):
            fitness = 1.0
        else:
            fitness = column_sum * block_sum
        
        self.fitness = fitness
        return
        
    def mutate(self, mutation_rate, given):
        r = random.uniform(0, 1.1)
        while(r > 1): 
            r = random.uniform(0, 1.1)
    
        success = False
        if (r < mutation_rate): 
            while(not success):
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1
                
                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while(from_column == to_column):
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)   

                if(given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0):
                    if(not given.is_column_duplicate(to_column, self.values[row1][from_column])
                       and not given.is_column_duplicate(from_column, self.values[row2][to_column])
                       and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column])
                       and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column])):

                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True
        return success

class Given(Candidate):
    def __init__(self, values):
        self.values = values
        return
        
    def is_row_duplicate(self, row, value):
        for column in range(0, Nd):
            if(self.values[row][column] == value):
               return True
        return False

    def is_column_duplicate(self, column, value):
        for row in range(0, Nd):
            if(self.values[row][column] == value):
               return True
        return False

    def is_block_duplicate(self, row, column, value):
        i = 3*(int(row/3))
        j = 3*(int(column/3))

        if((self.values[i][j] == value)
           or (self.values[i][j+1] == value)
           or (self.values[i][j+2] == value)
           or (self.values[i+1][j] == value)
           or (self.values[i+1][j+1] == value)
           or (self.values[i+1][j+2] == value)
           or (self.values[i+2][j] == value)
           or (self.values[i+2][j+1] == value)
           or (self.values[i+2][j+2] == value)):
            return True
        else:
            return False

class Tournament(object):
    def __init__(self):
        return
        
    def compete(self, candidates):
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.fitness
        f2 = c2.fitness

        if(f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while(r > 1):  
            r = random.uniform(0, 1.1)
        if(r < selection_rate):
            return fittest
        else:
            return weakest
    
class CycleCrossover(object):
    def __init__(self):
        return
    
    def crossover(self, parent1, parent2, crossover_rate):
        child1 = Candidate()
        child2 = Candidate()
        
        child1.values = numpy.copy(parent1.values)
        child2.values = numpy.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while(r > 1):  
            r = random.uniform(0, 1.1)
            
        if (r < crossover_rate):
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while(crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)
                
            if(crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
                
            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2): 
        child_row1 = numpy.zeros(Nd)
        child_row2 = numpy.zeros(Nd)

        remaining = list(range(1, Nd+1))  
        cycle = 0
        
        while((0 in child_row1) and (0 in child_row2)):  
            if(cycle % 2 == 0):  
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]
                
                while(next != start): 
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else: 
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]
                
                while(next != start):  
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]
                    
                cycle += 1
            
        return child_row1, child_row2  
           
    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if(parent_row[i] in remaining):
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if(parent_row[i] == value):
                return i


class Sudoku(object):
    def __init__(self):
        self.given = None
        return
    
    def load(self, path):
        with open(path, "r") as f:
            values = numpy.loadtxt(f).reshape((Nd, Nd)).astype(int)
            self.given = Given(values)
        return
    
    def solve(self):
        global global_fitness, geracao

        Nc = 1000  
        Ne = int(0.1*Nc)  
        Ng = 5000  
        Nm = 0  

        phi = 0
        sigma = 1
        mutation_rate = 0.2
    
        self.population = Population()
        self.population.seed(Nc, self.given)
    
        stale = 0
        for generation in range(0, Ng):
        
            print("Geração %d" % generation)
            
            best_fitness = 0.0
            for c in range(0, Nc):
                fitness = self.population.candidates[c].fitness
                if(fitness == 1):
                    print("Solução encontrada na geração %d!" % generation)
                    geracao = generation
                    print(self.population.candidates[c].values)
                    return self.population.candidates[c]

                if(fitness > best_fitness):
                    best_fitness = fitness

            print("Melhor fitness: %f" % best_fitness)
            global_fitness = best_fitness

            next_population = []

            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = numpy.copy(self.population.candidates[e].values)
                elites.append(elite)

            for count in range(Ne, Nc, 2):
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                child1.update_fitness()  
                child2.update_fitness() 

                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.given)
                child1.update_fitness()
                if(success):
                    Nm += 1
                    if(child1.fitness > old_fitness):
                        phi += 1

                old_fitness = child2.fitness  
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()
                if(success):
                    Nm += 1
                    if(child2.fitness > old_fitness):
                        phi += 1

                old_fitness = child2.fitness  
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()
                if(success):
                    Nm += 1
                    if(child2.fitness > old_fitness):
                        phi = phi + 1
                
                next_population.append(child1)
                next_population.append(child2)

            for e in range(0, Ne):
                next_population.append(elites[e])
                
            self.population.candidates = next_population
            self.population.update_fitness()
            
            if(Nm == 0):
                phi = 0  
            else:
                phi = phi / Nm
            
            if(phi > 0.2):
                sigma = sigma/0.998
            elif(phi < 0.2):
                sigma = sigma*0.998

            mutation_rate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
            Nm = 0
            phi = 0

            self.population.sort()
            if(self.population.candidates[0].fitness != self.population.candidates[1].fitness):
                stale = 0
            else:
                stale += 1

            if(stale >= 100):
                print("A população está estagnada. Reiniciando...")
                self.population.seed(Nc, self.given)
                stale = 0
                sigma = 1
                phi = 0
                Nm = 0
                mutation_rate = 0.06
        
        print("Nenhuma solução encontrada.")
        return None
        
def is_solution_correct(solution):
        
    for row in solution:
        if len(set(row)) != 9 or any(num < 1 or num > 9 for num in row):
            return False
    
    for col in range(9):
        column = [solution[row][col] for row in range(9)]
        if len(set(column)) != 9 or any(num < 1 or num > 9 for num in column):
            return False

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = [solution[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            if len(set(block)) != 9 or any(num < 1 or num > 9 for num in block):
                return False

    return True 

s = Sudoku()

if len(sys.argv) > 1:
    file_name = sys.argv[1] 
    print(f"Arquivo passado como argumento: {file_name}")
    s.load(file_name)  
else:
    print("Por favor, forneça o nome do arquivo como argumento.")

tic = time.time()
solution = s.solve()
toc = time.time()

print("Tempo gasto: ", toc - tic, "s")

def save_final(file_name, output_filename, time_taken):
    global global_fitness, geracao

    with open(file_name, 'r') as f:
        input_content = f.read()

    output_filename = file_name.replace("entrada", "solucao")
    
    numeros = list(map(int, input_content.split()))
    sudoku = [numeros[i:i+9] for i in range(0, len(numeros), 9)]

    with open(output_filename, 'w') as f:
        f.write(f"Entrada:\n")
        for linha in sudoku:
            f.write(" ".join(map(str, linha)) + "\n")
        f.write(f"\nSolucao:\n")
        for linha in solution.values:
            f.write(" ".join(map(str, linha)) + "\n")
        f.write(f"\nTempo gasto: {time_taken:.4f} segundos\n")
        f.write(f"Melhor fitness: {global_fitness}\n")
        f.write(f"Geracao: {geracao}")
    print(f"Arquivo salvo como {output_filename}")

if solution:
    if is_solution_correct(solution.values):
        print("A solução está correta!")
        output_filename = file_name.replace("entrada", "solucao")
        save_final(file_name, output_filename, toc - tic)
    else:
        print("A solução encontrada está incorreta.")
else:
    print("Nenhuma solução foi encontrada.")
