from typing import Callable
from scratch.linear_algebra.vector import Vector

def validate_vector(function: Callable) -> Callable:
    def wrapper(vector_1: Vector, vector_2: Vector) -> Vector:
        if not isinstance(vector_2, Vector):
            raise TypeError("Operation only supported between Vector instances")
        if len(vector_1) != len(vector_2):
            raise ValueError("Vectors must be of same length")            
        return function(vector_1, vector_2)
    return wrapper