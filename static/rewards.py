import numpy as np
def reward1(state, action, t):
    c1, c2, s1, s2 = state[0],state[2],state[1],state[3]
    height = -c1-(c1*c2-s1*s2)
    speed = pow(state[4] + state[5]/2, 2)
    return 100*(height*100+speed*37)

def reward2(state, action, t):
    c1, c2, s1, s2 = state[0],state[2],state[1],state[3]
    height = -c1-(c1*c2-s1*s2)
    speed = pow(state[4], 2) + pow(state[5]/2, 2)
    return 100*(height*10+speed)

def diff_lim(state, action, t):
    c1, c2, s1, s2 = state[0],state[2],state[1],state[3]
    h1 = -c1
    h2 = -(c1*c2-s1*s2)
    height = h1+h2
    ang_lead = s2-c2
    speed = np.sqrt(pow(state[4], 2) + pow(state[5]/2, 2)) 
    r = height*10+ang_lead+speed
    return r

def loc_torque(state, action, t):
    c1, c2, s1, s2, o1, o2 = state[0],state[2],state[1],state[3], state[4], state[5]
    c12 = c1*c2 - s1*s2
    height = 20*(0 - c1/2 - (c1 + c12/2))
    speed = np.power(o1, 2) + np.power(o2, 2)/2
    torque1 = -action[0] * o1 * 7
    torque2 = action[0] * o2 * 2
    return height+speed+torque1+torque2

all_rewards = {"Acb rw1": reward1, "Acb rw2": reward2, "Acb rwdl": diff_lim, "Acb 3 factors": loc_torque}
