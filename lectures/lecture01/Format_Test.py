# Check version of Python
import sys

print(sys.version)

# Import nypy and write a function that takes a string and returns it in uppercase and use it
import numpy as np


def to_uppercase(s):
    return s.upper()


print(to_uppercase("hello world"))

print(np.__version__)
