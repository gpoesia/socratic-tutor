import json

from environment import *

with open("ex_config.json") as f:
    config = json.load(f)
# print(config)

env = Environment.from_config(config)
# print(env.abstractions)

# new_prob = env.generate_new(seed=20) # -6x = ((2 - 8x) + (-1))
new_prob = env.generate_new(seed=3) # 9x = 3
# print(new_prob.goals)
# new_prob = State(["(((5 + ((-3x - -3x) - (-5))) - -3x) - (-3)) = ((((x * (-1)) - (x * (-3))) - -3x) - (-3))"], [""], 0.0)
# print(env.ax_seq_apply(('comm','assoc','comm','assoc'), new_prob.facts[-1]))
# print(env.iter_step_abs(new_prob.facts[-1]))

choices0 = env.step([new_prob])
# action1 = choices0[-1]
# state1 = action1.next_state
