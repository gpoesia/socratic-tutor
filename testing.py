import json

from environment import *

with open("ex_config.json") as f:
    config = json.load(f)
# print(config)

env = Environment.from_config(config)
# print(env.default_domain)

new_prob = env.generate_new(seed=20)
print(new_prob)

# actions = env.step([new_prob])[0][1]
# print(actions) # list of Action objects
# print(actions[0].next_state.facts)
next_states = commoncore.step("fractions", new_prob.facts[-1:])
print(next_states[0])