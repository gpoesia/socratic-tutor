# Utilities.

import datetime
import random

def format_eta(elapsed_time, elapsed_steps, total_steps):
    remaining_time = elapsed_time / elapsed_steps * (total_steps - elapsed_steps)
    return str(remaining_time)

def random_id(length=5):
    return "".join(random.choice("0123456789abcdef") for _ in range(length))
