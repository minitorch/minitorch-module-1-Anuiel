"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The same float number x.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The negation of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is less than y, otherwise False.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is equal to y, otherwise False.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Find the maximum of two numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close to each other within a small tolerance.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if the absolute difference between x and y is less than 1e-2, otherwise False.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The sigmoid of x.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The ReLU of x.

    """
    return max(0.0, x)


def log(x: float, scale: float = 1.0) -> float:
    """Compute the logarithm of a number scaled by a factor.

    Args:
    ----
        x: A float number.
        scale: A float number, scaling factor for the logarithm.

    Returns:
    -------
        The logarithm of x scaled by the factor.

    """
    return math.log(x) * scale


def exp(x: float, scale: float = 1.0) -> float:
    """Compute the exponential of a number scaled by a factor.

    Args:
    ----
        x: A float number.
        scale: A float number, scaling factor for the exponential.

    Returns:
    -------
        The exponential of x scaled by the factor.

    """
    return math.exp(x) * scale


def log_back(x: float, scale: float = 1.0) -> float:
    """Compute the gradient of the logarithm function with respect to x.

    Args:
    ----
        x: A float number.
        scale: A float number, scaling factor for the gradient.

    Returns:
    -------
        The gradient of the logarithm function with respect to x.

    """
    return scale / x


def inv(x: float) -> float:
    """Compute the inverse of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The inverse of x.

    """
    return 1.0 / x


def inv_back(x: float, scale: float = 1.0) -> float:
    """Compute the gradient of the inverse function with respect to x.

    Args:
    ----
        x: A float number.
        scale: A float number, scaling factor for the gradient.

    Returns:
    -------
        The gradient of the inverse function with respect to x.

    """
    return -scale / (x**2)


def relu_back(x: float, scale: float = 1.0) -> float:
    """Backward pass for ReLU activation function.

    Args:
    ----
        x: A float number, input to the ReLU function.
        scale: A float number, scaling factor for the gradient.

    Returns:
    -------
        The gradient of the ReLU function with respect to x.

    """
    return scale if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element in an iterable.

    Args:
    ----
        fn: A function that takes a float and returns a float.
        xs: An iterable of float numbers.

    Returns:
    -------
        An iterable with the function applied to each element.

    """
    return [fn(x) for x in xs]


def zipWith(
    fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Apply a function to pairs of elements from two iterables.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        xs: An iterable of float numbers.
        ys: An iterable of float numbers.

    Returns:
    -------
        An iterable with the function applied to each pair of elements.

    """
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(
    fn: Callable[[float, float], float], xs: Iterable[float], init: float
) -> float:
    """Reduce an iterable to a single value using a binary function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        xs: An iterable of float numbers.
        init: The initial value for the reduction.

    Returns:
    -------
        The result of the reduction.

    """
    result = init
    for x in xs:
        result = fn(result, x)
    return result


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list.

    Args:
    ----
        xs: An iterable of float numbers.

    Returns:
    -------
        An iterable with each element negated.

    """
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists.

    Args:
    ----
        xs: An iterable of float numbers.
        ys: An iterable of float numbers.

    Returns:
    -------
        An iterable with the sum of corresponding elements.

    """
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        xs: An iterable of float numbers.

    Returns:
    -------
        The sum of all elements.

    """
    return reduce(add, xs, 0)


def prod(xs: Iterable[float]) -> float:
    """Compute the product of all elements in a list.

    Args:
    ----
        xs: An iterable of float numbers.

    Returns:
    -------
        The product of all elements.

    """
    return reduce(mul, xs, 1)
