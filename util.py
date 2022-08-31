# Utilities.

import random
import datetime


def format_eta(elapsed_time, elapsed_steps, total_steps):
    remaining_time = elapsed_time / elapsed_steps * (total_steps - elapsed_steps)
    return str(remaining_time)


def random_id(length=5):
    return "".join(random.choice("0123456789abcdef") for _ in range(length))


# Helper decorator that registers the baseclass under the 'subtypes' attribute
# of the given class.
def register(superclass):
    def decorator(subclass):
        superclass.subtypes[subclass.__name__] = subclass
        return subclass
    return decorator


def now():
    'The current time as string, to be printed in log messages.'
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def batched(l: list, batch_size: int):
    i = 0
    while i < len(l):
        yield l[i:i+batch_size]
        i += batch_size
