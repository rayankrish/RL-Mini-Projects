import sys, pygame
import random

# --------- Settings --------------
NUM_TILES_PER_SIDE = 10
PIXELS_PER_TILE = 40
TIME_PER_FRAME = 100
RL_PLAYER = False

# ------- Game Mechanics ----------

pygame.init()

size = width, height = NUM_TILES_PER_SIDE*PIXELS_PER_TILE, NUM_TILES_PER_SIDE*PIXELS_PER_TILE
black = 0, 0, 0
white = 255, 255, 255
red = 255, 0, 0
font = pygame.font.SysFont('monogram', 18)
#fonts = pygame.font.get_fonts()
#print(fonts)

screen = pygame.display.set_mode(size)
screen.fill(black)

# tuple positions in tiles, 0 indexed, [tail ... head]
#snake = [(i, 1) for i in range(1, 6)]
snake = [(1,1)]
moves = ["N", "S", "E", "W"]
orient = "E"
apple = (NUM_TILES_PER_SIDE//4,  NUM_TILES_PER_SIDE//4)


def drawTile(pos, color):
    x, y = pos
    pygame.draw.rect(screen, color, pygame.Rect(x*PIXELS_PER_TILE, y*PIXELS_PER_TILE, PIXELS_PER_TILE, PIXELS_PER_TILE))


def drawSnake():
    for i, pos in enumerate(snake):
        #drawTile(pos, (i*10, i*20, i*20))
        drawTile(pos, white)


def moveSnake(old_snake, grow=False):
    new_snake = old_snake.copy()
    last_x, last_y = new_snake[-1]
    if orient == "E":
        new_snake.append((last_x+1, last_y))
    elif orient == "S":
        new_snake.append((last_x, last_y+1))
    elif orient == "W":
        new_snake.append((last_x-1, last_y))
    elif orient == "N":
        new_snake.append((last_x, last_y-1))
    if not grow: new_snake.pop(0)
    return new_snake


def isWithinScreen(_snake):
    head_x, head_y = _snake[-1]
    return 0 <= head_x < NUM_TILES_PER_SIDE and 0 <= head_y < NUM_TILES_PER_SIDE


def isNotOverlapping(_snake):
    head = _snake[-1]
    tail = _snake[:-1]
    return head not in tail

# ---------- RL Agent Stuff --------------
"""
state: (left right apple pos, up down apple pos, wall/snake above, below, left, right)
only 2^6 possible states
state: (signed horizontal delta, signed vertical delta, wall/snake above, below, left, right)
30*30*2^4 possible states
    should we have the left/right and up/down be -1 0 1 or binary?
    what if it was directional distance to the apple?
"""

state_space = [""]

for i in range(2):
    new_state_space = []
    for state in state_space:
        new_state_space.append(state+"0")
        new_state_space.append(state+"1")
        new_state_space.append(state+"2")
    state_space = new_state_space

for i in range(4):
    new_state_space = []
    for state in state_space:
        new_state_space.append(state+"0")
        new_state_space.append(state+"1")
    state_space = new_state_space


# for FUNctions and eligibility - 0 by default
class fastDict():
    def __init__(self):
        self.fast_dict = {}  # {state + action: eligibility}

    def update(self, s, a, val, prnt=False):
        key = s + " " + a
        if not val == 0 and prnt: print("-non zero update!", key, val)
        self.fast_dict[key] = val
        return not val == 0

    def get(self, s, a):
        key = s + " " + a
        if key not in self.fast_dict:
            return 0
        return self.fast_dict[key]


def basicCompFunction(left, right):
    if left > right: return "1"
    return "0"


def ternaryCompFunction(left, right):
    if left > right: return "0"
    if left < right: return "2"
    return "1"


def getState(_snake, _apple):
    state_str = ""
    head_x, head_y = _snake[-1]
    _apple_x, _apple_y = apple
    state_str += ternaryCompFunction(head_x, _apple_x)
    state_str += ternaryCompFunction(head_y, _apple_y)
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for delta_x, delta_y in deltas:
        # this is slightly wrong when we know that the tail will move out
        new_head = (head_x+delta_x, head_y+delta_y)
        if isWithinScreen([new_head]) and not new_head in _snake:
            state_str += "1"
        else:
            state_str += "0"
    return state_str


class QPolicy:
    def __init__(self, eps, lamb, gamma, alpha, init_state):
        self.eps = eps
        self.lamb = lamb
        self.gamma = gamma
        self.alpha = alpha
        self.S = init_state
        # all (s, a) are 0 by default
        self.eligibility = fastDict()
        self.Q = fastDict()
        self.A = "E"

    def resetEpisode(self, S, A="E"):
        self.eligibility = fastDict()
        self.S = S
        self.A = A

    def getBestAction(self, S):
        best_move = ("N", self.Q.get(S, "N"))
        for move in moves[1:]:
            if self.Q.get(S, move) > best_move[1]:
                best_move = (move, self.Q.get(S, move))
        return best_move[0]

    def getAction(self, S_prime, R):
        if random.uniform(0, 1) < self.eps:
            A_prime = moves[random.randint(0, 3)]
        else:
            A_prime = self.getBestAction(S_prime)

        A_star = self.getBestAction(S_prime)
        delta = R + self.gamma * self.Q.get(S_prime, A_star) - self.Q.get(self.S, self.A)
        self.eligibility.update(self.S, self.A, 1)  # replacing traces

        reset_eligibility = not A_prime == self.getBestAction(S_prime)
        # TODO: make more efficient
        for _state in state_space:
            for move in moves:
                self.Q.update(_state, move, self.Q.get(_state, move) + (self.alpha * delta * self.eligibility.get(_state, move)), prnt=True)
                """
                    print("actually changes something!, non-zero Q-elements:")
                    for k, v in QAgent.Q.fast_dict.items():
                        if not v==0: print("+",k, v)
                """

                if reset_eligibility:
                    self.eligibility.update(_state, move, 0)
                else:
                    # decay
                    self.eligibility.update(_state, move, self.gamma * self.lamb * self.eligibility.get(_state, move))

        # update local parameters
        self.S, self.A = S_prime, A_prime

        return self.A


if __name__ == "__main__":
    if RL_PLAYER: QAgent = QPolicy(0.05, 0.5, 0.5, 0.9, getState(snake, apple))
    top_score = 0

    for i in range(200):
        # reset the game
        snake = [(1, 1)]
        orient = "E"
        apple = (NUM_TILES_PER_SIDE//4,  NUM_TILES_PER_SIDE//4)

        if RL_PLAYER:
            QAgent.resetEpisode(getState(snake, apple))
            action = QAgent.A
        playable = True
        iteration = 1

        while playable:
            reward = -0.01
            if RL_PLAYER:
                orient = action  # take the agent's action
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and not orient == "E":
                        orient = "W"
                    if event.key == pygame.K_RIGHT and not orient == "W":
                        orient = "E"
                    if event.key == pygame.K_UP and not orient == "S":
                        orient = "N"
                    if event.key == pygame.K_DOWN and not orient == "N":
                        orient = "S"
            # if the human overrides the RL agent, reset the eligibility function
            if RL_PLAYER and not orient == action:
                QAgent.resetEpisode(getState(snake, apple), orient)

            screen.fill(black)

            # check if got an apple
            if snake[-1] == apple:
                snake = moveSnake(snake, True)
                apple = (random.randint(0, NUM_TILES_PER_SIDE-1), random.randint(0, NUM_TILES_PER_SIDE-1))
                reward = 1
            else:
                snake = moveSnake(snake)

            # check not losing position
            if not isWithinScreen(snake) or not isNotOverlapping(snake):
                print("fish sticks, you lose")
                print("score:", len(snake))
                reward = -1
                playable = False

            if RL_PLAYER:
                action = QAgent.getAction(getState(snake, apple), reward)

            drawTile(apple, red) # draw apple
            drawSnake()
            top_score = max(top_score, len(snake))

            displayString = 'score: ' + str(len(snake)) \
                            + "   game: " + str(i+1)\
                            + "   iteration: " + str(iteration)\
                            + "  top score: " + str(top_score)
            displayObj = font.render(displayString, True, white)
            screen.blit(displayObj, (7, NUM_TILES_PER_SIDE*PIXELS_PER_TILE-20))
            pygame.display.flip()
            iteration += 1
            if iteration >= 1000:
                playable = False

            pygame.time.wait(TIME_PER_FRAME)


