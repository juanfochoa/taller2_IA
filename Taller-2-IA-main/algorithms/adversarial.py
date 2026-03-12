from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """
    
    def minimax(self, state: GameState, agent_index, depth):
        
        #si ya perdimos/ganamos o alcanzamos la profundidad maxima evaluamos el estado actual
        if state.is_win() or state.is_lose() or depth == self.depth:
            return self.evaluation_function(state)
        
        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents # se usa el modulo para que cuando llegue al final pueda volver al turno inicial
        
        # si se vuelve a llegar al drone (0), se incrementa la profundidad
        if next_agent == 0:
            next_depth = depth + 1
        else:
            next_depth = depth   
            
        actions = state.get_legal_actions(agent_index)
        
        # parte de los drones (max)
        if agent_index == 0:
            value = float("-inf") #para luego establecer el maximo
            for action in actions:
                sucessor = state.generate_successor(agent_index, action)
                value = max(value, self.minimax(sucessor, next_agent, next_depth))
            return value
        
        # turno de los cazadores (min)
        else:
            value = float("inf")
            for action in actions:
                sucessor = state.generate_successor(agent_index, action)
                value = min(value, self.minimax(sucessor, next_agent, next_depth))
            return value        
        
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        best_action = None
        best_value = float("-inf")
        
        for action in state.get_legal_actions(0):
            sucessor = state.generate_successor(0, action)
            value = self.minimax(sucessor, 1, 0)

            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def alpha_beta(self, state: GameState, agent_index, depth, alpha, beta):
        if state.is_win() or state.is_lose() or depth == self.depth:
            return self.evaluation_function(state)
        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents

        if next_agent == 0:
            next_depth = depth + 1
        else:
            next_depth = depth
        
        actions = state.get_legal_actions(agent_index)

        # MAX node (drone)
        if agent_index == 0:
            value = float("-inf")
            for action in actions:
                successor = state.generate_successor(agent_index, action)
                value = max(value, self.alpha_beta(successor, next_agent, next_depth, alpha, beta))

                #poda
                if value > beta:  # prune
                    return value
                alpha = max(alpha, value)  # update alpha

            return value
        
        # MIN node (cazadores)
        else:
            value = float("inf")

            for action in actions:
                successor = state.generate_successor(agent_index, action)
                value = min(value, self.alpha_beta(successor, next_agent, next_depth, alpha, beta))

                #poda
                if value < alpha:  # prune
                    return value
                beta = min(beta, value)  # update beta

            return value

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        aplpha = float("-inf")
        beta = float("inf")

        best_action = None
        best_value = float("-inf")

        actions = state.get_legal_actions(0)

        for action in actions:
            successor = state.generate_successor(0, action)
            value = self.alpha_beta(successor, 1, 0, aplpha, beta)

            if value > best_value:
                best_value = value
                best_action = action

            aplpha = max(aplpha, best_value)
            
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """
    #Se toma como base la función de minimax, pero se cambia el como se calculan los nodos de los cazadores
    def expectimax(self, state: GameState, agent_index, depth):
        
        #si ya perdimos/ganamos o alcanzamos la profundidad maxima evaluamos el estado actual
        if state.is_win() or state.is_lose() or depth == self.depth:
            return self.evaluation_function(state)
        
        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents # se usa el modulo para que cuando llegue al final pueda volver al turno inicial
        
        # si se vuelve a llegar al drone (0), se incrementa la profundidad
        if next_agent == 0:
            next_depth = depth + 1
        else:
            next_depth = depth   
            
        actions = state.get_legal_actions(agent_index)
        
        # parte de los drones (max)
        if agent_index == 0:
            value = float("-inf") #para luego establecer el maximo
            for action in actions:
                sucessor = state.generate_successor(agent_index, action)
                value = max(value, self.minimax(sucessor, next_agent, next_depth))
            return value
        
        # turno de los cazadores (min)
        else:
            # se guardan los valores de cada posible acción del hunter
            child_values = []
            for action in actions:
                sucessor = state.generate_successor(agent_index, action)
                values = self.expectimax(sucessor, next_agent, next_depth)
                child_values.append(values)
            
            # representa cuando un cazador actua aleatoriamente
            mean = sum(child_values) / len(child_values)
                
            return (1-self.prob) * min(child_values) + self.prob * mean(child_values)

    # se maneja la misma logica que para minimax
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        best_action = None
        best_value = float("inf")
        
        for action in state.generate_successor(0):
            sucessor = state.generate_successor(0, action)
            value = self.minimax(sucessor, 1, 0)

            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
