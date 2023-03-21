import numpy as np
size = 5

goal = (2,3)
obstacle = [(0,3), (2,2), (1,1), (2,1), (3,1), (3,3)]

def setup(size, goal, obstacle):
    S = np.ones((size,size), dtype=int) * -1
    
    #Placement But
    S[goal[0], goal[1]] = 10
    
    #Placement des murs
    for obs in obstacle:
        S[obs[0], obs[1]] = -99

    SNew = np.zeros((size,size), dtype=float)
    return S, SNew

def reward(state):
    return S[state[0], state[1]]

def transition(state, action):

    if state[1] == 0 and action == "up":
        return state

    if state[0] == size and action == "right":
        return state
        
    if state[1] == size and action == "down":
        return state 

    if state[0] == 0 and action == "left":
        return state
    
    match action:
        case "up":
            return (state[0], state[1] - 1)
        case "right":
            return (state[0] + 1, state[1])
        case "left":
            return (state[0] - 1, state[1])
        case "down":
            return (state[0], state[1] + 1)
    return 0

def listerActions(i, j):
    listeActions = []
    if i >0:
        listeActions.append("left")
    if i < size-1:
        listeActions.append("right")
    if j > 0:
        listeActions.append("up")
    if j < size - 1:
        listeActions.append("down")

    return listeActions

def solve(S, SNew):
    for k in range(size*size*3):
        i = k%size
        j = (k//size)%size

        listeActions = listerActions(i, j)
        valueACote = []

        for action in listeActions:
            if action == "up":
                valueACote.append(0.9 * SNew[i, j-1])
            if action == "down":
                valueACote.append(0.9 * SNew[i, j+1])
            if action == "left":
                valueACote.append(0.9 * SNew[i-1, j])
            if action == "right":
                valueACote.append(0.9 * SNew[i+1, j])

        SNew[i, j] = S[i, j] + max(valueACote)
    
    return SNew

def choisirActions(SNew, state):
    listeActions = listerActions(state[0], state[1])
    ValeurACote = []
    i = state[0]
    j = state[1]
    for action in listeActions:
        if action == "up":
            ValeurACote.append(SNew[i, j-1])
        if action == "down":
            ValeurACote.append(SNew[i, j+1])
        if action == "left":
            ValeurACote.append(SNew[i-1, j])
        if action == "right":
            ValeurACote.append(SNew[i+1, j])



S, SNew = setup(size, goal, obstacle)
print("Environnement")
print(S)
print(solve(S, SNew))

A = ["up", "right", "down", "left"]