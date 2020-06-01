import numpy as np

class Network:
  def __init__(self, input_size, output_size):
    self.nodes = []
    self.connections = []
    self.innovation_num = 0
    self.node_num = 0
    self.fitness = 0

    self.input_index = 0
    self.hidden_index = 1
    self.output_index = 2

    self.input_size = input_size
    self.output_size = output_size

    self.fitness = 0

    self.create_network()

  def update_node(self):
    """
    Incrementally updates the node number for a network when adding a new node
    """
    self.node_num += 1
    return self.node_num
  
  def update_inn(self):
    """
    Incrementally updates the global innovation number for a network
    """
    self.innovation_num += 1
    return self.innovation_num

  def create_network(self):
    """
    Initializes a feed-forward neural network with no hidden layers
    """
    self.nodes.append([[self.update_node(), 0, 1] for i in range(self.input_size)])
    self.nodes.append([])
    self.nodes.append([[self.update_node(), 0, 3] for i in range(self.output_size)])

    for out in range(self.output_size): # 1
      for inp in range(self.input_size): # 9
        self.connections.append([2 * np.random.random() - 1, [self.nodes[self.input_index][inp], self.nodes[self.output_index][out]], self.update_inn(), "ENABLED"])
