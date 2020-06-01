# NEAT-Python-Implementation
A Python implementation of Dr. Stanley's and Dr. Miikkulainen's NEAT algorithm.

Original paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

To use:
  - Initialize a population by instantiating a Networks object with 3 parameters - population size, input size, and output size.
  - Input data into each organism using the action function (input data must be a list)
  - Evaluate each organism and give each a fitness score
  - Use the evolve() method for the Networks object to evolve the population
  - Reiterate
  
Current bugs:
  - This implementation gets a bit buggy with any network that takes in more than 2 inputs. Evolution may be slow or may not happen at all.

What has it solved:
  - This implementation has successfully solved the XOR function and maxed out the score for OpenAI's cartpole environment.
