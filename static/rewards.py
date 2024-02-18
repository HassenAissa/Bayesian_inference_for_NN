def reward1(state, action, t):
    c1, c2, s1, s2 = state[0],state[2],state[1],state[3]
    height = -c1-(c1*c2-s1*s2)
    speed = pow(state[4] + state[5]/2, 2)
    return 100*(height*100+speed*37)

all_rewards = {"reward1": reward1}
