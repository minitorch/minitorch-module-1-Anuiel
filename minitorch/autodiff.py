from dataclasses import dataclass
from typing import Any, Generator, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    args_left = [val + (epsilon if idx == arg else 0.0) for idx, val in enumerate(vals)]
    args_right = [val - (epsilon if idx == arg else 0) for idx, val in enumerate(vals)]
    value_left = f(*args_left)
    value_right = f(*args_right)
    return (value_left - value_right) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    def _topological_sort_inner(variable: Variable) -> Generator[Variable, None, None]:
        visited: set[int] = set()
        stack: list[Variable] = [variable]
        while stack:
            current = stack[-1]
            if current.unique_id in visited:
                stack.pop()
                continue
            if current.is_constant():
                stack.pop()
                continue
            if current.is_leaf() or all(
                p.unique_id in visited for p in current.parents
            ):
                stack.pop()
                visited.add(current.unique_id)
                yield current
            else:
                for p in current.parents:
                    if p.unique_id not in visited:
                        stack.append(p)

    return list(_topological_sort_inner(variable))[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    derivative_log: dict[int, Any] = {variable.unique_id: deriv}
    for v in order:
        current_derivatve = derivative_log[v.unique_id]
        if not v.is_leaf():
            for p, d in v.chain_rule(current_derivatve):
                derivative_log[p.unique_id] = derivative_log.get(p.unique_id, 0) + d
        else:
            v.accumulate_derivative(current_derivatve)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
