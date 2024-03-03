# diff and action are normalized
import tensorflow as tf

def h2(state):
    return state[0]*state[2] - state[1]*state[3]

def ht_speed(state, t):
    height = 1-state[0] + 3 - state[0]*2 - h2(state)
    speed = pow(state[4], 2) + pow(state[5], 2)
    return height*10+speed

def upright(state, t):
    angle = - state[2] 
    rot = -state[3]*state[2]
    time = t*(-pow(angle,2) + pow(0.2095, 2))
    return angle+rot+time

all_rewards = {"Acb 2 factors": ht_speed, "Cart": upright}
