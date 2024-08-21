# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:14:09 2023

@author: Igor
"""
import math

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

def calc_dist(currentState, goalState, mapWidth):
    return (abs(math.floor(goalState/mapWidth) - math.floor(currentState/mapWidth)) + abs((goalState%mapWidth) - (currentState%mapWidth)))

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "droneMA_v0",
        "is_parallelizable": True,
        "render_fps": 10,
    }
    
    def __init__(self, render_mode=None):

        # Map Key
        # 0: Empty Space
        # 1: Start
        # 2: Goal
        # 3: Wall
        # 4: Full Shelf
        # 5: Empty Shelf

        # self.Map = [
        #             1,0,0,0,0,1,
        #             0,0,0,0,0,0,
        #             0,0,4,4,0,0,
        #             0,0,4,4,0,0,
        #             0,0,0,0,0,0,
        #             0,0,2,2,0,0
        #             ]
        
        self.Map = [
                    1,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,
                    0,0,4,0,0,4,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,4,0,0,4,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,2,2,0,0,0
                    ]
        self.MapWidth = 8
        self.Delivered = 0
        self.screen = None
        self.clock = None
        self.width, self.height = 100 * self.MapWidth, 100 * len(self.Map)/self.MapWidth
        
        self.depots = find_indices(self.Map, 4)
        
        self.currentReward = 0

        self.agents = ["drone_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.start = find_indices(self.Map, 1) 
        self.state = find_indices(self.Map, 1)
        self.distanceFromGoal = []
        for i in self.state:
            self.distanceFromGoal.append(calc_dist(i, self.depots[0], self.MapWidth))
        self.BatteryLife = [200 for agent in self.agents]
        self.Packages = [-1 for i in find_indices(self.Map, 4)]    
        self.PackageStates = [i for i in self.depots]
    
        # Action Space
        # 0: Move Up
        # 1: Move Right
        # 2: Move Down
        # 3: Move Left
        # 4: Pick Up
        # 5: Drop
        # 6: Do nothing
        
        self.action_spaces = {agent: Discrete(4) for agent in self.agents}
        self.observation_spaces = {
            agent: Discrete(len(self.Map) + 2) for agent in self.agents
        }

        self.render_mode = render_mode
        self.screen = None

        self.reinit()
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def reinit(self):
        
        # self.Map = [
        #             1,0,0,0,0,1,
        #             0,0,0,0,0,0,
        #             0,0,4,4,0,0,
        #             0,0,4,4,0,0,
        #             0,0,0,0,0,0,
        #             0,0,2,2,0,0
        #             ]
        
        self.Map = [
                    1,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,
                    0,0,4,0,0,4,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,4,0,0,4,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,2,2,0,0,0
                    ]
        
        self.MapWidth = 8
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.BatteryLife = [200 for agent in self.agents]

        self.state = list(reversed(find_indices(self.Map, 1)))#  find_indices(self.Map, 1)
        self.distanceFromGoal = []
        for i in self.state:
            self.distanceFromGoal.append(calc_dist(i, self.depots[0], self.MapWidth))
        self.Packages = [-1 for i in find_indices(self.Map, 4)]
        self.PackageStates = [i for i in self.depots]
        self.Delivered = 0
        
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            self.render_mode = "human"
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Drones")

        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        GREY = (128, 128, 128)
        
        image = pygame.image.load("drone.png").convert_alpha()
        boxImage = pygame.image.load("CardboardBox.png").convert()
        
        image_width, image_height = 90, 90
        image = pygame.transform.scale(image, (image_width, image_height))
        
        self.screen.fill(WHITE)
        
        square_size = self.width // self.MapWidth
        
        for i in range(self.MapWidth + 1):
            pygame.draw.line(self.screen, BLACK, (0, i * square_size), (self.width, i * square_size))
            pygame.draw.line(self.screen, BLACK, (i * square_size, 0), (i * square_size, self.height))
            
            
        deliverSpot = find_indices(self.Map, 2)
        for ele in deliverSpot:
            pygame.draw.rect(self.screen, GREEN, ((ele % self.MapWidth) * square_size, math.floor(ele / self.MapWidth) * square_size, square_size, square_size))
        
        for ele in self.depots:
            pygame.draw.rect(self.screen, BLUE, ((ele % self.MapWidth) * square_size, math.floor(ele / self.MapWidth)* square_size, square_size, square_size))
        
        startSpot = find_indices(self.Map, 1)
        for ele in startSpot:
            pygame.draw.rect(self.screen, GREY, ((ele % self.MapWidth) * square_size, math.floor(ele / self.MapWidth) * square_size, square_size, square_size))
        
        droneX = [5 + (int(i) % self.MapWidth) * square_size for i in self.state]
        droneY = [5 + math.floor(int(int(i)) / self.MapWidth) * square_size for i in self.state]
        
        for i in self.Packages:
            if i < 0:
                packageLocation = find_indices(self.Map, 4)
                for i in packageLocation:
                    package_size = 60
                    boxImage = pygame.transform.scale(boxImage, (60, 60))
                    package_x = 20 + (i % self.MapWidth) * square_size
                    package_y = 20 + math.floor(i / self.MapWidth) * square_size
            
                    pygame.draw.rect(self.screen, BLACK, (package_x - 2, package_y - 2, package_size + 4, package_size + 4), 0)
                    self.screen.blit(boxImage, (package_x, package_y))   
                    
        for i in self.Packages:
            if i >= 0:
                package_size = 80
                boxImage = pygame.transform.scale(boxImage, (80, 80))
                package_x = droneX[i] + 5
                package_y = droneY[i] + 5
                
                pygame.draw.rect(self.screen, BLACK, (package_x - 2, package_y - 2, package_size + 4, package_size + 4), 0)
                self.screen.blit(boxImage, (package_x, package_y))  
        for i in range(len(self.state)):
            self.screen.blit(image, (droneX[i], droneY[i]))
    
        # Update the display
        pygame.event.pump()
        self.clock.tick(60)
        pygame.display.flip()
        
    def observe(self, agent):
        if self.agents.index(agent) in self.Packages:
            package = [1]
        else:
            package = [0]
            
        my_list = self.Map + [self.state[self.agents.index(agent)]] + package 

        return np.array(my_list, dtype=np.int64)

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def reset(self, seed=None, options=None):
        self.reinit()

    def step(self, action):
        reward = 0
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        
        inx = self.agents.index(agent)
        
        if action == 0:
            if self.state[inx] > self.MapWidth-1:
                if self.Map[int(self.state[inx]) - self.MapWidth] != 3:
                    if int(self.state[inx]) - self.MapWidth not in self.state:
                        self.state[inx] -= self.MapWidth
            
        if action == 1:
            if self.state[inx] % self.MapWidth != self.MapWidth -1:
                if self.Map[int(self.state[inx]) + 1] != 3:
                    if int(self.state[inx]) + 1 not in self.state:
                        self.state[inx] += 1
                    
        if action == 2:
            if self.state[inx] < len(self.Map) - self.MapWidth-1:
                if self.Map[int(self.state[inx]) + self.MapWidth] != 3:
                    if int(self.state[inx]) + self.MapWidth not in self.state:
                        self.state[inx] += self.MapWidth
                
        if action == 3:
            if self.state[inx] % self.MapWidth != 0:
                if self.Map[int(self.state[inx]) - 1] != 3:
                    if int(self.state[inx]) - 1 not in self.state:
                        self.state[inx] -= 1
                    
        if inx not in self.Packages:
            if self.Map[int(self.state[inx])] == 4:
                self.Map[int(self.state[inx])] = 5
                
                package = self.PackageStates.index(self.state[inx])

                    
                self.Packages[package] = inx
                reward = 1
                
                self.distanceFromGoal[inx] = [i for i, n in enumerate(self.Map) if n == 2][0]

        else:
            if self.Map[int(self.state[inx])] == 2:
                
                package = self.Packages.index(inx)
                
                self.Packages[package] = -2
                reward = 1
                self.Delivered += 1
                
                if 4 in self.Map:
                    self.distanceFromGoal[inx] = [i for i, n in enumerate(self.Map) if n == 4][0]

        if inx in self.Packages:
            goalState = [i for i, n in enumerate(self.Map) if n == 2][0]
        elif 4 in self.Map:
            goalState = [i for i, n in enumerate(self.Map) if n == 4][0]
        else:
            goalState = self.start[inx]
            
        currentDistance = calc_dist(self.state[inx], goalState, self.MapWidth)
          
        if currentDistance < self.distanceFromGoal[inx]:
            reward = 1
            self.distanceFromGoal[inx] = currentDistance
                
        self.BatteryLife[inx] -= 1

        if self.BatteryLife[inx] <= 0:
            self.terminations[agent] = True
            
        if self.Delivered == 4:
            self.terminations[agent] = True
        
        self.rewards[self.agent_selection] += reward
        
        info = {}
        
        self.agent_selection = self._agent_selector.next()
        if self.render_mode == "human":
            self.render() 