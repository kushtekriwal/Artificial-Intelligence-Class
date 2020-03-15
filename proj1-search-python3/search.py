# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    path = []
    visited = set()

    toVisit = Stack()
    #toVisit2 = []
    start = problem.getStartState()
    pos_and_dir = [start, path]
    toVisit.push(pos_and_dir)
    current = pos_and_dir

    while (not toVisit.isEmpty()):
      current = toVisit.pop()
      visited.add(current[0])
      if problem.isGoalState(current[0]):
        return current[1]
      successors = problem.getSuccessors(current[0])
      for s in successors:
        pos, direct, _ = s
        new_dir = current[1] + [direct]
        if (pos not in visited):
          toVisit.push([pos, new_dir])
          #toVisit2.append(pos)
    return current[1]

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    path = []
    visited = []

    toVisit = Queue()
    toVisit2 = []
    start = problem.getStartState()
    pos_and_dir = [start, path]
    toVisit.push(pos_and_dir)
    toVisit2.append(start)
    current = pos_and_dir

    while (not toVisit.isEmpty()):
      current = toVisit.pop()
      visited.append(current[0])
      if problem.isGoalState(current[0]):
        return current[1]
      successors = problem.getSuccessors(current[0])
      for s in successors:
        state, direct, _ = s
        new_dir = current[1] + [direct]
        if (state not in visited and state not in toVisit2):
          toVisit.push([state, new_dir])
          toVisit2.append(state)
    return current[1]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    path = []
    cost = 0
    visited = set()

    toVisit = PriorityQueue()
    toVisit2 = {}
    start = problem.getStartState()
    pos_and_dir_cost = [start, path, cost]
    toVisit.push(pos_and_dir_cost, cost)
    toVisit2[start] = 0
    current = pos_and_dir_cost

    while (not toVisit.isEmpty()):
      current = toVisit.pop()
      visited.add(current[0])
      if problem.isGoalState(current[0]):
        return current[1]
      successors = problem.getSuccessors(current[0])
      for s in successors:
        pos, direct, cost = s
        new_dir = current[1] + [direct]
        new_cost = current[2] + cost
        if (pos in toVisit2 and new_cost < toVisit2.get(pos)):
          toVisit.push([pos, new_dir, new_cost], new_cost)
          toVisit2[pos] = new_cost
        elif pos not in visited and pos not in toVisit2:
          toVisit.push([pos, new_dir, new_cost], new_cost)
          toVisit2[pos] = new_cost
    return current[1]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    path = []
    total = 0
    visited = []

    toVisit = PriorityQueue()
    toVisit2 = {}
    #toVisit2 = []
    start = problem.getStartState()
    pos_and_dir_and_tot = [start, path, total]
    toVisit.push(pos_and_dir_and_tot, total)
    toVisit2[start] = 0
    #toVisit2.append(start)
    current = pos_and_dir_and_tot

    while (not toVisit.isEmpty()):
      current = toVisit.pop()
      visited.append(current[0])
      if problem.isGoalState(current[0]):
        return current[1]
      successors = problem.getSuccessors(current[0])
      for s in successors:
        pos, direct, cost = s
        new_dir = current[1] + [direct]
        #print(current[0])
        h_o = heuristic(current[0], problem)
        h_n = heuristic(pos, problem)
        new_tot = current[2] + cost - h_o + h_n
        if (pos in toVisit2 and new_tot < toVisit2.get(pos)):
          toVisit.push([pos, new_dir, new_tot], new_tot)
          toVisit2[pos] = new_tot
        if(pos not in visited and pos not in toVisit2):
          toVisit.push([pos, new_dir, new_tot], new_tot)
          toVisit2[pos] = new_tot
          #toVisit2.append(pos)
    return current[1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
