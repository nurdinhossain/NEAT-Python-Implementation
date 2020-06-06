import pygame
import random
import sys
sys.path.append("../NEAT")
from Network import Network
from Networks import Networks
from activations import sigmoid, relu, tanh
from action import action

pygame.init()

# VARIABLES
displayWidth = 600
displayHeight = 600
sky_blue = (186, 253, 255)
red = (255, 0, 0)
green = (0, 255, 0)
yellow = (237, 230, 33)

clock = pygame.time.Clock()
fps = 30

display = pygame.display.set_mode((displayWidth, displayHeight))
pygame.display.set_caption("Flappy Bird")

# FUNCTIONS
def drawBackground(display):
    display.fill(sky_blue)

def drawMessage(display, message, color, x, y):
    font = pygame.font.SysFont("comicsansms", 72)
    text = font.render(message, True, color)
    display.blit(text, (x, y))

# CLASSES
class Bird:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.gravity = 1.7
        self.velocity = 0
        self.alive = True

    def draw(self):
        if self.alive:
            pygame.draw.ellipse(display, yellow, [self.x, self.y, self.size, self.size])

    def moveDown(self):
        if self.alive:
            self.y += self.velocity
            self.velocity += self.gravity

    def moveUp(self):
        if self.alive:
            self.y -= 50
            self.velocity = -11

class Pipe:
    def __init__(self, x, y, height):
        self.x = x
        self.y = y
        self.height = height

    def draw(self):
        pygame.draw.rect(display, green, [self.x, self.y, 100, self.height])

# MAIN LOOP
def gameLoop():
    gameOver = False
    gameExit = False
    populationSize = 500
    brains = Networks(populationSize, 4, 1)
    generations = []
    max_fitnesses = []

    while not gameExit:
        ballX = displayWidth / 2
        ballY = displayHeight / 2
        gravity = 1.7
        velocity = 0
        birdSize = 25

        birds = [Bird(ballX, ballY, birdSize) for i in range(len(brains.networks))]        
        points = 0

        topPipeX = []
        topPipeY = []
        topPipeHeight = []
        bottomPipeX = []
        bottomPipeY = []
        bottomPipeHeight = []
        gapSize = 140

        gameSpeed = 10 # 20

        for brain in brains.networks:
          brain.fitness = 0
        
        while not gameExit and not gameOver:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gameExit = True

            drawBackground(display)
            
            for bird in birds:
                bird.draw()
            for bird in birds:
                bird.moveDown()

            # GENERATE FIRST PIPE
            if len(topPipeX) == 0:
                topPipeX.append(displayWidth - 100)
                topPipeY.append(0)
                topPipeHeight.append(random.randrange(5, displayHeight - gapSize - 5))

                bottomPipeX.append(displayWidth - 100)
                bottomPipeY.append(topPipeHeight[-1] + gapSize)
                bottomPipeHeight.append(displayHeight - (topPipeHeight[-1] + gapSize))   

            # INSERT DATA INTO NEURAL NETWORKS
            outputs = []
            for bird in range(len(birds)):
                output = action([(topPipeX[-1] - birds[bird].x) / 100, (birds[bird].y - (topPipeY[-1] + topPipeHeight[-1])) / 100], brains.networks[bird], sigmoid, sigmoid)
                outputs.append(output)

            # MOVE BIRD BASED ON NETWORK DECISION
            for bird in range(len(birds)):
                if outputs[bird][0][1] < 0.5:
                    birds[bird].moveUp()
                else:
                    birds[bird].moveDown()
                
            # GENERATE NEW PIPES / ADD POINTS TO SCOREBOARD
            birdsX = [birds[bird].x for bird in range(len(birds))]

            passedPipe = False
            for x in birdsX:
                if x >= topPipeX[-1] + 100:
                    passedPipe = True
                
            if passedPipe:
                topPipeX.append(displayWidth - 100)
                topPipeY.append(0)
                topPipeHeight.append(random.randrange(5, displayHeight - gapSize - 5))

                bottomPipeX.append(displayWidth - 100)
                bottomPipeY.append(topPipeHeight[-1] + gapSize)
                bottomPipeHeight.append(displayHeight - (topPipeHeight[-1] + gapSize))

                points += 1
                print("Points: " + str(points))

            # DRAW PIPES
            for pipe in range(len(topPipeY)):
                Pipe(topPipeX[pipe], topPipeY[pipe], topPipeHeight[pipe]).draw()
                Pipe(bottomPipeX[pipe], bottomPipeY[pipe], bottomPipeHeight[pipe]).draw()

            # MOVE PIPES BACKWARD
            for pipe in range(len(topPipeX)):
                topPipeX[pipe] -= gameSpeed
                bottomPipeX[pipe] -= gameSpeed

            # CHECK FOR COLLISION W/ PIPE + GROUND
            for bird in range(len(birds)):
                if (birds[bird].x + birdSize > topPipeX[-1] and birds[bird].x < (topPipeX[-1] + 100) and birds[bird].y < (topPipeY[-1] + topPipeHeight[-1])) or (birds[bird].x + birdSize > bottomPipeX[-1] and birds[bird].x < (bottomPipeX[-1] + 100) and birds[bird].y + birdSize > bottomPipeY[-1]):
                    birds[bird].alive = False
                    
                elif birds[bird].y + birdSize > displayHeight:
                    birds[bird].alive = False

                elif birds[bird].y < 0:
                    birds[bird].alive = False

            # END GAME WHEN NO BIRDS REMAIN
            life_statuses = [bird.alive for bird in birds]
            if not True in life_statuses:
                gameOver = True

            # UPDATE BIRD FITNESSES
            for bird in range(len(birds)):
                if birds[bird].alive:
                    brains.networks[bird].fitness += 1
            
            pygame.display.update()
            clock.tick(fps)
        
        brains.evolve(1, 0.8)
        gameOver = False

        pygame.display.update()     

gameLoop()
pygame.quit()
quit()
