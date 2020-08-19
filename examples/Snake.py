import pygame
import random
import sys
sys.path.append("../NEAT")
from Network import Network
from Networks import Networks
from activations import sigmoid, relu, tanh
from action import action
import matplotlib.pyplot as plt

pygame.init()

# VARIABLES
displayWidth = 500
displayHeight = 500
red = (255, 0, 0)
green = (0, 255, 0)
white = (255, 255, 255)

clock = pygame.time.Clock()
fps = 30

display = pygame.display.set_mode((displayWidth, displayHeight))
pygame.display.set_caption("Snake")

# FUNCTIONS
def drawBackground(display):
    display.fill(white)

# CLASSES
class Snake:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.xSpeed = 0
        self.ySpeed = -10
        self.bodyX = [self.x]
        self.bodyY = [self.y]
        self.size = size
        self.length = 3
        self.alive = True

    def draw(self):
        if self.alive:
            for body in range(len(self.bodyX)):
                pygame.draw.rect(display, green, [self.bodyX[body], self.bodyY[body], self.size, self.size])

    def changeDir(self, case):
        if self.alive:
            if case == "w":
                self.xSpeed = 0
                self.ySpeed = -self.size
            elif case == "a":
                self.xSpeed = -self.size
                self.ySpeed = 0
            elif case == "s":
                self.xSpeed = 0
                self.ySpeed = self.size
            elif case == "d":
                self.xSpeed = self.size
                self.ySpeed = 0      

    def move(self):
        if self.alive:
            self.x += self.xSpeed
            self.y += self.ySpeed
            self.bodyX.insert(0, self.x)
            self.bodyY.insert(0, self.y)

            while len(self.bodyX) > self.length:
                self.bodyX.pop()
                self.bodyY.pop()

class Apple:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    def draw(self):
        pygame.draw.rect(display, red, [self.x, self.y, self.size, self.size])

def gameLoop():
    gameOver = False
    gameExit = False
    populationSize = 250
    snakeBrains = Networks(populationSize, 7, 4)
    #snakeBrains.new_node = 0.07
    #snakeBrains.new_link = 0.1
    generations = 0
    generations_list = []
    avg_fitnesses = []
    while not gameExit:
        snakeSize = 10
        snakeX = round((displayWidth / 2) / snakeSize) * snakeSize
        snakeY = round((displayHeight / 2) / snakeSize) * snakeSize     
        snakes = [Snake(snakeX, snakeY, snakeSize) for i in range(len(snakeBrains.networks))]

        apples = [Apple(round(random.randrange(0, displayWidth - snakeSize) / snakeSize) * snakeSize, round(random.randrange(0, displayHeight - snakeSize) / snakeSize) * snakeSize, snakeSize) for i in range(len(snakeBrains.networks))]
        time = 0
        max_time = 200
        num_apples = 0
        show_replay = False

        generations += 1
        if generations % 10 == 0:
          max_time += 50
        while not gameExit and not gameOver:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gameExit = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        show_replay = True

            drawBackground(display)

            for snake in snakes:
              snake.draw()
            for apple in apples:
              apple.draw()

            # INSERT DATA INTO NEURAL NETWORKS
            outputs = []
            for network in range(len(snakeBrains.networks)):
                current_snake = snakes[network]
                inputs = [snakes[network].x - apples[network].x, snakes[network].y - apples[network].y, snakes[network].x - snakes[network].bodyX[-1], snakes[network].y - snakes[network].bodyY[-1]]
                
                # MOVING RIGHT - done
                if current_snake.xSpeed > 0:
                    for body_index, _ in enumerate(current_snake.bodyX):
                        if body_index == 0:
                          continue
                        # front
                        if current_snake.x + current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # above
                        if current_snake.x == current_snake.bodyX[body_index] and current_snake.y - current_snake.size == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # below
                        if current_snake.x == current_snake.bodyX[body_index] and current_snake.y + current_snake.size == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                # MOVING LEFT - done
                elif snakes[network].xSpeed < 0:
                    for body_index, _ in enumerate(current_snake.bodyX):
                        if body_index == 0:
                          continue
                        # front
                        if current_snake.x - current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # above
                        if current_snake.x == current_snake.bodyX[body_index] and current_snake.y - current_snake.size == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # below
                        if current_snake.x == current_snake.bodyX[body_index] and current_snake.y + current_snake.size == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                # MOVING DOWN                      
                elif snakes[network].ySpeed > 0:
                    for body_index, _ in enumerate(current_snake.bodyX):
                        if body_index == 0:
                          continue
                        # right
                        if current_snake.x + current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # left
                        if current_snake.x - current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # front
                        if current_snake.x == current_snake.bodyX[body_index] and current_snake.y + current_snake.size == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                # MOVING UP     
                elif snakes[network].ySpeed < 0:
                    for body_index, _ in enumerate(current_snake.bodyX):
                        if body_index == 0:
                          continue
                        # right
                        if current_snake.x + current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # left
                        if current_snake.x - current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)

                        # front
                        if current_snake.x == current_snake.bodyX[body_index] and current_snake.y - current_snake.size == current_snake.bodyY[body_index]:
                            inputs.append(100)
                            break
                        elif body_index == len(current_snake.bodyX) - 1:
                            inputs.append(0)
                            
                output = action(inputs, snakeBrains.networks[network], sigmoid, sigmoid)
                outputs.append(output)

            # MOVE SNAKE BASED ON NETWORK DECISION
            for snake in range(len(snakes)):
                if snakes[snake].alive:
                    decision = "w"
                    greatest_num = 0
                    output_nodes = outputs[snake]

                    if output_nodes[0][1] > greatest_num:
                        greatest_num = output_nodes[0][1]
                        decision = "w"
                    if output_nodes[1][1] > greatest_num:
                        greatest_num = output_nodes[1][1]
                        decision = "a"
                    if output_nodes[2][1] > greatest_num:
                        greatest_num = output_nodes[1][1]
                        decision = "s"
                    if output_nodes[3][1] > greatest_num:
                        greatest_num = output_nodes[1][1]
                        decision = "d"

                    snakes[snake].changeDir(decision)
                
            # ALLOWS SNAKE TO EAT APPLE
            for snake in range(len(snakes)):
                if snakes[snake].x == apples[snake].x and snakes[snake].y == apples[snake].y:
                    num_apples += 1
                    snakes[snake].length += 1
                    snakeBrains.networks[snake].fitness += 50
                    apples[snake].x = round(random.randrange(0, displayWidth - snakeSize) / snakeSize) * snakeSize
                    apples[snake].y = round(random.randrange(0, displayHeight - snakeSize) / snakeSize) * snakeSize

            # UPDATE FITNESS BASED ON DISTANCE TO APPLE
            for snake in range(len(snakes)):
                snakes[snake].move()
                if snakes[snake].alive:
                  if snakes[snake].x > apples[snake].x - (4 * snakeSize) and snakes[snake].x < apples[snake].x + (4 * snakeSize) and snakes[snake].y > apples[snake].y - (4 * snakeSize) and snakes[snake].y < apples[snake].y + (4 * snakeSize):
                    #snakeBrains.networks[snake].fitness += 100
                    pass

            # DETECT BODY/WALL COLLISION
            for snake in range(len(snakes)):
                if snakes[snake].length > 1:
                    for body in range(len(snakes[snake].bodyX)):
                        if body == 0:
                            continue
                        if snakes[snake].x == snakes[snake].bodyX[body] and snakes[snake].y == snakes[snake].bodyY[body]:
                            snakes[snake].alive = False
                            #snakeBrains.networks[snake].fitness /= 1.5 # /= 1.5

            # DISPLAY TELEPORT
            for snake in snakes:
                if snake.x > displayWidth:
                  snake.x = 0
                if snake.x < 0:
                  snake.x = displayWidth
                if snake.y > displayHeight:
                  snake.y = 0
                if snake.y < 0:
                  snake.y = displayHeight

            # END GAME IF NO SNAKES REMAIN
            life = [snake.alive for snake in snakes]
            if not True in life:
              gameOver = True

            time += 1
            if time > max_time:
              gameOver = True
            
            pygame.display.update()
            clock.tick(fps * 4)

        print(str(num_apples) + " apples eaten by population")
        """
        REPLAY
        """
        if show_replay:
            max_snake_fitness = max([network.fitness for network in snakeBrains.networks])
            for network in snakeBrains.networks:
              if network.fitness == max_snake_fitness:
                brain = network
                break
            snake = Snake(snakeX, snakeY, snakeSize)
            apple = Apple(round(random.randrange(0, displayWidth - snakeSize) / snakeSize) * snakeSize, round(random.randrange(0, displayHeight - snakeSize) / snakeSize) * snakeSize, snakeSize)

            replay_time = 0
            while replay_time < 500:
              for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                  if event.key == pygame.K_r:
                    replay_time = -9999999999999999
                    print("Infinite")
                  if event.key == pygame.K_e:
                    replay_time = 0
                    print("Reset")
              drawBackground(display)
              snake.draw()
              apple.draw()

              # INSERT DATA INTO NEURAL NETWORKS
              current_snake = snake
              inputs = [snake.x - apple.x, snake.y - apple.y]
              
              # MOVING RIGHT - done
              if current_snake.xSpeed > 0:
                  for body_index, _ in enumerate(current_snake.bodyX):
                      if body_index == 0:
                        continue
                      # front
                      if current_snake.x + current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # above
                      if current_snake.x == current_snake.bodyX[body_index] and current_snake.y - current_snake.size == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # below
                      if current_snake.x == current_snake.bodyX[body_index] and current_snake.y + current_snake.size == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

              # MOVING LEFT - done
              elif current_snake.xSpeed < 0:
                  for body_index, _ in enumerate(current_snake.bodyX):
                      if body_index == 0:
                        continue
                      # front
                      if current_snake.x - current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # above
                      if current_snake.x == current_snake.bodyX[body_index] and current_snake.y - current_snake.size == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # below
                      if current_snake.x == current_snake.bodyX[body_index] and current_snake.y + current_snake.size == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

              # MOVING DOWN                      
              elif current_snake.ySpeed > 0:
                  for body_index, _ in enumerate(current_snake.bodyX):
                      if body_index == 0:
                        continue
                      # right
                      if current_snake.x + current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # left
                      if current_snake.x - current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # front
                      if current_snake.x == current_snake.bodyX[body_index] and current_snake.y + current_snake.size == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

              # MOVING UP     
              elif current_snake.ySpeed < 0:
                  for body_index, _ in enumerate(current_snake.bodyX):
                      if body_index == 0:
                        continue
                      # right
                      if current_snake.x + current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # left
                      if current_snake.x - current_snake.size == current_snake.bodyX[body_index] and current_snake.y == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

                      # front
                      if current_snake.x == current_snake.bodyX[body_index] and current_snake.y - current_snake.size == current_snake.bodyY[body_index]:
                          inputs.append(100)
                          break
                      elif body_index == len(current_snake.bodyX) - 1:
                          inputs.append(0)

              output = action(inputs, brain, h_activation=sigmoid, o_activation=sigmoid)

              decision = "w"
              greatest_num = 0

              if output[0][1] > greatest_num:
                  greatest_num = output[0][1]
                  decision = "w"
              if output[1][1] > greatest_num:
                  greatest_num = output[1][1]
                  decision = "a"
              if output[2][1] > greatest_num:
                  greatest_num = output[1][1]
                  decision = "s"
              if output[3][1] > greatest_num:
                  greatest_num = output[1][1]
                  decision = "d"
              snake.changeDir(decision)
              snake.move()

              # ALLOW SNAKE TO EAT APPLE
              if snake.x == apple.x and snake.y == apple.y:
                  snake.length += 1
                  apple.x = round(random.randrange(0, displayWidth - snakeSize) / snakeSize) * snakeSize
                  apple.y = round(random.randrange(0, displayHeight - snakeSize) / snakeSize) * snakeSize

              # DETECT BODY/WALL COLLISION
              if snake.length > 1:
                  for body in range(len(snake.bodyX)):
                      if body == 0:
                          continue
                      if snake.x == snake.bodyX[body] and snake.y == snake.bodyY[body]:
                          snake.alive = False

              # DISPLAY TELEPORT
              if snake.x > displayWidth:
                snake.x = 0
              if snake.x < 0:
                snake.x = displayWidth
              if snake.y > displayHeight:
                snake.y = 0
              if snake.y < 0:
                snake.y = displayHeight

              # CHECK IF SNAKE IS DEAD
              if snake.alive == False:
                break

              replay_time += 1
              pygame.display.update()
              clock.tick(fps)

        generations_list.append(generations)
        avg_fitnesses.append(sum([network.fitness for network in snakeBrains.networks]) / len([network.fitness for network in snakeBrains.networks]))
        
        snakeBrains.evolve(1, 0.8)
        gameOver = False
    plt.plot(generations_list, avg_fitnesses)
    plt.show()
    while True:
        if input("Break? y/n") == "y":
            break

gameLoop()
pygame.quit()
quit()
