import numpy as np
import pygame

class GridEnvironment:


    def __init__(self):
        self.grid = np.zeros((7,7))
        self.starting_position = [0,0]
        self.goal_position = [5,4]
        self.reset()
        self.walls = [[3, 2], [2, 3], [1, 4]]
        #actions are up, down, left, right mapped to 0, 1, 2, 3


    def reset(self):
        self.agent = self.starting_position
        self.special_blocks = {
            -2: [4, 1],
            2: [2, 2],
            1: [2, 4],
            3: [4, 6]
        }
        return self.agent

    def valid_move(self, state, action):
        # Check for walls or outside grid
        new_position = list(state)
        if action == 0:
            new_position[0] -= 1
        elif action == 1:
            new_position[0] += 1
        elif action == 2:
            new_position[1] -= 1
        elif action == 3:
            new_position[1] += 1

        if new_position in self.walls or new_position[0] < 0 or new_position[0] > 6 or new_position[1] < 0 or new_position[1] > 6:
            return False
        return True

    def check_value(self, state):
        for reward_value, coordinates in self.special_blocks.items():
            if coordinates == state:
                return reward_value
        return 0

    def remove_value(self, state):
        for reward_value, coordinates in list(self.special_blocks.items()):
            if coordinates == state:
                self.special_blocks.pop(reward_value)
                break

    def next_state(self, action):
        new_position = list(self.agent)
        if action == 0:
            new_position[0] -= 1
        elif action == 1:
            new_position[0] += 1
        elif action == 2:
            new_position[1] -= 1
        elif action == 3:
            new_position[1] += 1

        if self.valid_move(self.agent, action):
            return new_position
        else:
            return self.agent

    def step(self, action):
        next_position = self.next_state(action)
        reward = -1  # default reward

        special_reward = self.check_value(next_position)
        if special_reward:
            reward += special_reward
            self.remove_value(next_position)

        self.agent = next_position  # updating the agent's position

        if self.agent == self.goal_position:
            reward += 10
            return next_position, reward, True

        return next_position, reward, False

    def render(self, q_table):
        pygame.init()
        self.reset()

        # Configurations
        BLOCK_SIZE = 60  # Increased block size
        WINDOW_SIZE = [self.grid.shape[0] * BLOCK_SIZE, self.grid.shape[1] * BLOCK_SIZE]
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption('Q-table Visualization')

        colors = {
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'gray': (200, 200, 200)
        }

        actions = ["up", "down", "left", "right"]
        font = pygame.font.SysFont(None, 25)  # Reduced font size

        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            screen.fill(colors['white'])

            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    state = (i, j)
                    max_action = np.argmax(q_table[state])

                    # Choose color based on max action value
                    max_val = q_table[state][max_action]
                    color = colors['green'] if max_val > 0 else colors['red'] if max_val < 0 else colors['blue']

                    pygame.draw.rect(screen, color, [j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])

                    # Display action text
                    action_text = font.render(actions[max_action], True, colors['white'])
                    screen.blit(action_text, (j * BLOCK_SIZE + 10, i * BLOCK_SIZE + 10))

                    # Check and draw special blocks and their values
                    for reward_value, coordinates in self.special_blocks.items():
                        if coordinates == [i, j]:
                            pygame.draw.circle(screen, colors['yellow'],
                                               (j * BLOCK_SIZE + BLOCK_SIZE // 2, i * BLOCK_SIZE + BLOCK_SIZE // 2),
                                               BLOCK_SIZE // 3)
                            reward_text = font.render(str(reward_value), True, colors['gray'])
                            screen.blit(reward_text,
                                        (j * BLOCK_SIZE + BLOCK_SIZE // 2 - 10, i * BLOCK_SIZE + BLOCK_SIZE // 2 - 10))

                    # Draw walls
                    if [i, j] in self.walls:
                        pygame.draw.rect(screen, colors['gray'],
                                         [j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()






