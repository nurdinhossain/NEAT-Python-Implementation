def action(data, network, h_activation=relu, o_activation=sigmoid):
  """
  Inputs data into a neural network and returns the raw output

  *Data must be a scalar (list)
  """
  connections = [[conn[0], [conn[1][0].copy(), conn[1][1].copy()], conn[2], conn[3]] for conn in network.connections]
  nodes = [[node.copy() for node in node_type] for node_type in network.nodes]

  for part in range(len(data)): # for each individual value in each training example
    nodes[network.input_index][part][1] = data[part] # applies 1 training example to all network nodes

  for hidden_node in nodes[network.hidden_index]:
    for conn in connections:
      if conn[1][1][0] == hidden_node[0] and conn[-1] == "ENABLED":
        start_node_type = nodes[conn[1][0][-1] - 1]
        for node in start_node_type:
          if node[0] == conn[1][0][0]:
            start_node = node

        hidden_node[1] += conn[0] * start_node[1]

    hidden_node[1] = h_activation(hidden_node[1])
    #print(hidden_node[1])
  #print()

  for output_node in nodes[network.output_index]:
    for conn in connections:
      if conn[1][1][0] == output_node[0] and conn[-1] == "ENABLED":
        start_node_type = nodes[conn[1][0][-1] - 1]
        for node in start_node_type:
          if node[0] == conn[1][0][0]:
            start_node = node

        output_node[1] += conn[0] * start_node[1]

    output_node[1] = o_activation(output_node[1])
    #print(output_node[1])
  #print()

  return nodes[network.output_index]
