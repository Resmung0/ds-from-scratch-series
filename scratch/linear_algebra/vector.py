from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from scratch.validation.vector import validate_vector


@dataclass(slots=True)
class Vector(Sequence):
    """
    A vector class that represents a sequence of floating-point numbers. This class provides 
    a convenient way to work with vectors in linear algebra. It supports common vector operations
    such as addition, subtraction, scalar multiplication, dot product, and more. The class ensures
    that the vector data is a non-empty list of floats. It also provides methods to calculate 
    the magnitude, normalize it, and compute the distance between two vectors.
    """
    data: list[float]
    
    def __post_init__(self):
        if not self.data:
            raise ValueError("Data cannot be empty")
        for element in self.data:
            if not isinstance(element, float):
                raise TypeError("Data must be a float or int")
    
    def __neg__(self) -> Vector:
        """
        Returns a new vector with the negation of each element in the current vector.
        
        Returns:
            Vector: A new vector with the negation of each element in the current vector.
        """
        return Vector([-x for x in self.data])
    
    def __len__(self) -> int:
        """
        Returns the number of elements in the vector's data.
      
        Returns:
          int: The length of the vector's data.
        """
        return len(self.data)

    def __getitem__(self, index) -> float:
        """
        Returns the element at the specified index in the vector's data.
        
        Args:
            index (int): The index of the element to retrieve.
        
        Returns:
            float: The element at the specified index.
        """
        return self.data[index]
    
    @validate_vector
    def __add__(self, other) -> Vector:
        """
        Add each element of the vector to the corresponding element in the other vector.
        
        Args:
            other (Vector): The other vector to add to this vector.
        
        Returns:
            Vector: A new vector with the result of adding the corresponding elements.
        """
        return Vector(
            [a + b for a, b in zip(self.data, other.data)]
        )
    
    @validate_vector
    def __sub__(self, other) -> Vector:
        """
        Subtract each element of the vector by the corresponding element in the other vector.
        
        Args:
            other (Vector): The other vector to subtract from this vector.
        
        Returns:
            Vector: A new vector with the result of subtracting the corresponding elements.
        """
        return Vector(
            [a - b for a, b in zip(self.data, other.data)]
        )
    
    def __mul__(self, scalar: float) -> Vector:
        """
        Multiply each element of the vector by the given scalar.
        
        Args:
            scalar (float): The scalar to multiply the vector by.
        
        Returns:
            Vector: A new vector with each element multiplied by the scalar.
        """
        return Vector([a * scalar for a in self.data])
    
    def __rmul__(self, scalar: float) -> Vector:
        """
        Reverse multiplication of a vector by a scalar.
        
        Args:
            scalar (float): The scalar to multiply the vector by.
        
        Returns:
            Vector: A new vector with each element multiplied by the scalar.
        """
        return self * scalar
    
    def __truediv__(self, scalar: float) -> Vector:
        """
        Divide each element of the vector by the given scalar.
        
        Args:
            scalar (float): The scalar to divide the vector by.

        Returns:
            Vector: A new vector with each element divided by the scalar.
        """
        return Vector([a / scalar for a in self.data])
    
    @validate_vector
    def __matmul__(self, other: Vector) -> float:
        """
        Dot product of two vectors.
        
        Args:
            other (Vector): The other vector to compute the dot product with.
        
        Returns:
            float: The dot product of the two vectors.
        """
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def __abs__(self) -> float:
        """
        Magnitude (length) of the vector.
        
        Returns:
            float: The magnitude of the vector.
        """
        return sqrt(self @ self)
    
    def normalize(self) -> Vector:
        """
        Normalize the vector to a unit vector.
        
        Returns:
            Vector: The normalized vector.
        """
        return Vector([a / abs(self) for a in self.data])
    
    def distance(self, other: Vector) -> float:
        """
        Euclidean distance between two vectors.
        
        Args:
            other (Vector): The other vector to calculate the distance to.
        
        Returns:
            float: The Euclidean distance between the two vectors.
        """
        return abs(self - other)
