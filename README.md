# NEAT-Python-Implementation
A Python implementation of Dr. Stanley's and Dr. Miikkulainen's NEAT algorithm.

Original paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Brief Explanation:

NEAT is a genetic algorithm that uses evolution to slowly complexify a simple population so that members of the population become sufficient in solving a certain problem. It does so by giving each organism in the population a fitness score, or a measure of how well the organism did. Organisms with high fitness scores are bred and pass offspring into the next generation. Organisms with low fitness scores are killed. This process of breeding the best and discarding the rest essentially encourages good genes to be passed on to the next generation and for bad genes to be eliminated. Additionally, organisms in a NEAT population are placed into species based on how similar they are to one another. This allows different architectures to optimize over time, allowing NEAT to explore a multitude of possible solutions. Lastly, NEAT organisms can go through mutations, which change the genome of an organism and introduce diversity into the population, allowing for even more solutions to be discovered.

Libraries/Dependencies:
  - PyGame - `pip install pygame`
  - NumPy - `pip install numpy`

To implement:
  - Initialize a population by instantiating a Networks object with 3 parameters - population size, input size, and output size.
  - Input data into each organism using the action function (input data must be a list)
  - Evaluate each organism and give each a fitness score
  - Use the evolve() method for the Networks object to evolve the population
  - Reiterate

What has it solved:
  - This implementation has successfully solved the XOR function and maxed out the score for OpenAI's cartpole environment.
  - This implementation is also able to evolve populations in the game Snake and Flappy Bird (in examples folder)
  
Examples
  - Snake: inspired by Sentdex/thenewboston - https://www.youtube.com/watch?v=K5F-aGDIYaM&list=PL6gx4Cwl9DGAjkwJocj7vlc_mFU-4wXJq
  - Flappy Bird: inspired by Code Bullet - https://www.youtube.com/watch?v=WSW-5m8lRMs
