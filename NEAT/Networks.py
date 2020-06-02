import Network
import random
import math

Network = Network.Network

class Networks:
  def __init__(self, population, input_size, output_size):
    self.population_size = population

    self.weight_mutation = 0.8 # 0.8
    self.uniform_change = 0.9
    self.disable = 0.75
    self.raw_mutation = 0.25 # not being used
    self.new_node = 0.03 # 0.03
    self.new_link = 0.05 # 0.05

    self.input_index = 0
    self.hidden_index = 1
    self.output_index = 2
    
    self.input_size = input_size
    self.output_size = output_size
    self.networks = [Network(input_size, output_size) for i in range(self.population_size)]
    
    self.species = []
    self.species_reps = [] # list of representative genome for each species
    self.c_one = 1
    self.c_two = 1
    self.c_three = 0.4 # 0.4
    self.threshold = 3
    self.compat_mod = 0.3

  def max_innovation(self, conns): # WORKS
    """
    Return the highest innovation number in a genome.
    It does so by returning the innovation number of the last connection.

    *Requires an unmodified connection
    """
    return conns[-1][2] # return the innovation number of the LAST connection

  def prep_connections(self, conn_one, conn_two): # WORKS
    """
    Takes two connections and equalizes their lengths by filling in gaps with a blank list []
    """
    conn_one_prepped = [[att for att in conn] for conn in conn_one]
    conn_two_prepped = [[att for att in conn] for conn in conn_two]

    if self.max_innovation(conn_one) > self.max_innovation(conn_two):
      max_innov = self.max_innovation(conn_one)
    else:
      max_innov = self.max_innovation(conn_two)

    for innov in range(max_innov):
      try:
        if conn_one_prepped[innov][2] != innov + 1: # if there is a missing gene
          conn_one_prepped.insert(innov, [])
      except IndexError:
        conn_one_prepped.append([])

      try:
        if conn_two_prepped[innov][2] != innov + 1:
          conn_two_prepped.insert(innov, [])
      except IndexError:
        conn_two_prepped.append([])

    return conn_one_prepped, conn_two_prepped

  def avg_weight_diff(self, conn_one, conn_two, absolute_value=True): # WORKS
    """
    Takes the average weight differences of MATCHING genes

    *Requires two modified and equalized genomes
    """
    total_weight_diff = 0
    matching_genes = 0
    for conn_index, conn in enumerate(conn_one):
      if conn_one[conn_index] != [] and conn_two[conn_index] != []:
        matching_genes += 1
        if absolute_value == True:
          total_weight_diff += abs(conn_one[conn_index][0] - conn_two[conn_index][0])
        else:
          total_weight_diff += conn_one[conn_index][0] - conn_two[conn_index][0]

    return total_weight_diff / matching_genes

  def count_excess(self, conn_one, conn_two): # WORKS
    """
    Count and identify the excess genes between two genomes by:
      1. Finding the max innovation of each genome
      2. Any genes between the smaller max innovation and the larger max innovation is considered an excess gene.

    *Requires an unmodified connection
    """
    max_innov_one = self.max_innovation(conn_one)
    max_innov_two = self.max_innovation(conn_two)
    excess_genes = []

    if max_innov_one > max_innov_two:
      for conn in conn_one:
        if conn[2] > max_innov_two:
          excess_genes.append(conn)
    else:
      for conn in conn_two:
        if conn[2] > max_innov_one:
          excess_genes.append(conn)

    return excess_genes, len(excess_genes)

  def count_disjoint(self, conn_one, conn_two): # WORKS
    """
    Count the number of disjoint genes between two genomes.

    * Requires modified and equalized genomes
    """
    excess_genes, _ = self.count_excess(conn_one, conn_two)
    conn_one_prepped, conn_two_prepped = self.prep_connections(conn_one, conn_two)
    disjoint_genes = []

    for conn_index, _ in enumerate(conn_one_prepped):
      if conn_one_prepped[conn_index] != [] and conn_two_prepped[conn_index] == [] and not conn_one_prepped[conn_index] in excess_genes:
        disjoint_genes.append(conn_one_prepped[conn_index])
      elif conn_one_prepped[conn_index] == [] and conn_two_prepped[conn_index] != [] and not conn_two_prepped[conn_index] in excess_genes:
        disjoint_genes.append(conn_two_prepped[conn_index])

    return disjoint_genes, len(disjoint_genes)

  def speciate(self): # WORKS
    """
    - Separates the networks into species based on compatibility distance and compatibility threshold
    - Adjusts fitnesses of each network with respect to specie length
    - Repicks species representatives
    """
    for specie in self.species:
      specie.clear()

    for network in self.networks:
      if len(self.species) == 0: # if there are no species yet
        self.species.append([network])
        self.species_reps.append(network)
        continue

      for specie, rep in enumerate(self.species_reps):
        network_conns_prepped, rep_conns_prepped = self.prep_connections(network.connections, rep.connections)

        excess_genes, excess_count = self.count_excess(network.connections, rep.connections)
        disjoint_genes, disjoint_count = self.count_disjoint(network.connections, rep.connections)
        avg_weight_difference = self.avg_weight_diff(network_conns_prepped, rep_conns_prepped)

        if len(network.connections) < 20 and len(rep.connections) < 20:
          N = 1
        elif len(network.connections) > len(rep.connections):
          N = len(network.connections)
        elif len(rep.connections) >= len(network.connections):
          N = len(rep.connections)

        compatibility_distance = ((self.c_one * excess_count) / 1) + ((self.c_two * disjoint_count) / 1) + (self.c_three * avg_weight_difference)

        if compatibility_distance < self.threshold:
          self.species[specie].append(network)
          break # stop checking through each specie if this network has been successfully speciated
        elif specie == len(self.species) - 1: # if it is the last specie and the network still has not been speciated
          self.species.append([network])
          self.species_reps.append(network)
          self.threshold += self.compat_mod

    for specie_index, specie in enumerate(self.species):
      # REPICK SPECIES REPRESENTATIVES
      if len(specie) != 0:
        self.species_reps[specie_index] = random.choice(specie)

      # ADJUST FITNESSESS
      for network in specie:
        network.fitness /= len(specie)

  def mutate_node(self, network, conns, nodes): # WORKS
    """
    Initializes a new node splitting an existing enabled connection. 
    As a result, the split connection is disabled and two new connection genes are added to the genome.

    *Requires an unmodified genome
    """
    updated_conns = [[att for att in conn] for conn in conns]
    updated_nodes = [[node for node in node_type] for node_type in nodes]

    split_choices = [conn for conn in updated_conns if conn[1][0][0] < network.node_num + 1 and conn[1][1][-1] == 3 and conn[-1] == "ENABLED"] # choices include connections from input -> output or hidden -> output
    if len(split_choices) == 0:
      return None, None

    split_conn_index = random.randrange(0, len(split_choices))
    split_conn_start_node = split_choices[split_conn_index][1][0]
    split_conn_end_node = split_choices[split_conn_index][1][1]
    new_node = [network.update_node(), 0, self.hidden_index + 1]
    
    updated_conns[updated_conns.index(split_choices[split_conn_index])][-1] = "DISABLED" # disable split connection
    updated_nodes[self.hidden_index].append(new_node)
    updated_conns.append([1, [split_conn_start_node, new_node], network.update_inn(), "ENABLED"]) # connection going into new node
    updated_conns.append([updated_conns[split_conn_index][0], [new_node, split_conn_end_node], network.update_inn(), "ENABLED"]) # connection leading out of new node

    return updated_conns, updated_nodes

  def mutate_connections(self, network, conns, nodes):
    """
    Initializes a new connection with a random weight connecting two random un-connected nodes.
    Node connection goes from low -> high e.g. input -> hidden, hidden -> upper hidden, hidden -> output, input -> output

    *Requires an unmodified genome
    """
    updated_conns = [[att for att in conn] for conn in conns]

    if nodes[self.hidden_index] == []:
      return None
    else:
      start_node_index = random.randrange(self.input_index, self.hidden_index + 1)

    if start_node_index == self.input_index and nodes[self.hidden_index] == []:
      out_node_index = self.output_index
    elif start_node_index == self.input_index and nodes[self.hidden_index] != []:
      out_node_index = random.randrange(self.hidden_index, self.output_index + 1)
    elif start_node_index == self.hidden_index:
      out_node_index = random.randrange(self.hidden_index, self.output_index + 1)

    start_node = random.choice(nodes[start_node_index])
    end_node = random.choice(nodes[out_node_index])

    if start_node in nodes[self.hidden_index] and end_node in nodes[self.hidden_index]: # if both nodes are hidden nodes
      if start_node[0] >= end_node[0]: # if start node is higher than end node or both reference the same node
        return None

    new_connection = [random.uniform(-1, 1), [start_node, end_node], network.update_inn(), "ENABLED"]
    if not new_connection[1] in [conn[1] for conn in conns]: # if connection does not exist
      updated_conns.append(new_connection)
      return updated_conns

    network.innovation_num -= 1
    return None

  def mutate_weights(self, conns):
    """
    Can either nudge weights by some small value or change them completely

    *Requires an unmodified genome
    """
    updated_conns = [[att for att in conn] for conn in conns]
    for conn in updated_conns:
      if random.random() <= self.uniform_change:
        conn[0] += random.gauss(0, 1) / 50 # change weight by slight value
      else:
        conn[0] = random.uniform(-1, 1) # completely changes weight

    return updated_conns

  def crossover(self, fitness_one, fitness_two, conn_one, conn_two):
    """
    Breeds 2 genomes and returns a mutated/un-mutated "child" network object
    """
    child_conns = []
    child_nodes = [[], [], []]
    prepped_conn_one, prepped_conn_two = self.prep_connections(conn_one, conn_two)

    # CROSS CONNECTION GENOMES
    for conn_index, _ in enumerate(prepped_conn_one):
      if prepped_conn_one[conn_index] != [] and prepped_conn_two[conn_index] != []:
        child_conns.append(random.choice([prepped_conn_one[conn_index], prepped_conn_two[conn_index]]))

        # 75% chance of disabling connection if disabled in either parent
        if prepped_conn_one[conn_index][-1] == "DISABLED" or prepped_conn_two[conn_index][-1] == "DISABLED":
          if random.random() <= self.disable:
            child_conns[-1][-1] = "DISABLED"
          else:
            child_conns[-1][-1] = "ENABLED"

      elif prepped_conn_one[conn_index] != [] and prepped_conn_two[conn_index] == []:
        if fitness_one >= fitness_two:
          child_conns.append(prepped_conn_one[conn_index])

          # 75% chance of disabling connection if disabled in either parent
          if prepped_conn_one[conn_index][-1] == "DISABLED":
            if random.random() <= self.disable:
              child_conns[-1][-1] = "DISABLED"
            else:
              child_conns[-1][-1] = "ENABLED"

      elif prepped_conn_one[conn_index] == [] and prepped_conn_two[conn_index] != []:
        if fitness_two >= fitness_one:
          child_conns.append(prepped_conn_two[conn_index])
          
          # 75% chance of disabling connection if disabled in either parent
          if prepped_conn_two[conn_index][-1] == "DISABLED":
            if random.random() <= self.disable:
              child_conns[-1][-1] = "DISABLED"
            else:
              child_conns[-1][-1] = "ENABLED"

    # FIND CHILD NODES
    unsorted_nodes = []
    for conn in child_conns:
      if not conn[1][0] in unsorted_nodes:
        unsorted_nodes.append(conn[1][0])
      if not conn[1][1] in unsorted_nodes:
        unsorted_nodes.append(conn[1][1])

    for node in unsorted_nodes:
      child_nodes[node[-1] - 1].append(node)
    for node_type in child_nodes:
      node_type.sort()

    # CREATE NEW CHILD NETWORK
    child = Network(self.input_size, self.output_size)
    child.nodes = child_nodes
    child.connections = child_conns
    child.innovation_num = self.max_innovation(child_conns)
    if len(child_nodes[self.hidden_index]) > 0:
      child.node_num = max([node[0] for node in child_nodes[self.hidden_index]])
    else:
      child.node_num = max([node[0] for node in child_nodes[self.output_index]])

    # MUTATE NETWORK
    if random.random() <= self.weight_mutation:
      child.connections = self.mutate_weights(child.connections)
    if random.random() <= self.new_node:
      updated_connections, updated_nodes = self.mutate_node(child, child.connections, child.nodes)
      if updated_connections != None and updated_nodes != None:
        child.connections = updated_connections
        child.nodes = updated_nodes
    if random.random() <= self.new_link:
      updated_conns = self.mutate_connections(child, child.connections, child.nodes)
      if updated_conns != None:
        child.connections = updated_conns

    return child

  def eliminate(self, population, percentage):
    """
    Eliminates the bottom x% of the population

    *Given population should be a species
    """
    population_fitnesses = sorted([network.fitness for network in population])
    new_population = population.copy()

    for i in range(int(percentage * len(population))):
      killed = False
      for network in new_population:
        if not killed:
          if network.fitness == population_fitnesses[i]:
            new_population.remove(network)
            killed = True

    return new_population

  def evolve(self, generations, control_percent=0.8, insert_champions=True):
    """
    Evolves a population of NNs for n generations

    STEPS:
      1. Evaluate
      2. Speciate
      3. Adjust fitnesses
      4. Allocate/eliminate
      5. Reproduce

    *Assumes each network has already been evaluated and has a representative fitness score
    """
    for gen in range(generations):
      offspring = []
      self.speciate()

      species_adj_fitness_sums = [sum([network.fitness for network in specie]) for specie in self.species]
      species_allocations = [round(self.population_size * (specie_adj_fitness_sum / sum(species_adj_fitness_sums))) for specie_adj_fitness_sum in species_adj_fitness_sums]
    
      print("ALLOCATIONS: " + str(species_allocations))

      # Insert champions into next generation
      if insert_champions == True:
        for specie_index, specie in enumerate(self.species):
          if len(specie) >= 5:
            fitnesses = [network.fitness for network in specie]
            offspring.append(specie[fitnesses.index(max(fitnesses))])
            species_allocations[specie_index] -= 1
      
      # Kill worst performing members
      for specie_index, _ in enumerate(self.species):
        self.species[specie_index] = self.eliminate(self.species[specie_index], control_percent)

      # Crossover
      for specie_index, specie in enumerate(self.species):
        for i in range(int(species_allocations[specie_index])):
          network_one = random.choice(specie)
          network_two = random.choice(specie)

          child = self.crossover(network_one.fitness, network_two.fitness, network_one.connections, network_two.connections)
          offspring.append(child)

      self.networks = offspring.copy()
