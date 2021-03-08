# Utilities.

import datetime

def format_eta(elapsed_time, elapsed_steps, total_steps):
    remaining_time = elapsed_time / elapsed_steps * (total_steps - elapsed_steps)
    return str(remaining_time)
