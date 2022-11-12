from tensornetwork.backends import abstract_backend
from typing import Union, Sequence, Tuple, Optional, Any, Callable, List, Type
import paddle
import numpy as np

Tensor = paddle.Tensor


class PaddleBackend(abstract_backend.AbstractBackend):
    def __init__(self) -> None:
        super().__init__()
        # pylint: disable=global-variable-undefined
        global paddlelib
        try:
        # pylint: disable=import-outside-toplevel
            import paddle
        except ImportError as err:
            raise ImportError("PyTorch not installed, please switch to a different "
                            "backend or install PyTorch.") from err
        paddlelib = paddle
        self.name = "paddle"

    def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
        return paddlelib.tensordot(a, b, axes=axes)

    def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
        return paddlelib.reshape(tensor, tuple(np.array(shape).astype(int)))

    def transpose(self, tensor: Tensor, perm=None) -> Tensor:
        if perm is None:
            perm = tuple(range(tensor.ndim - 1, -1, -1))
        return tensor.transpose(perm)

    def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
        if len(start_indices) != len(slice_sizes):
            raise ValueError("Lengths of start_indices and slice_sizes must be"
                    "identical.")
        obj = tuple(
            slice(start, start + size)
            for start, size in zip(start_indices, slice_sizes))
        return tensor[obj]

    def shape_concat(self, values: Tensor, axis: int) -> Tensor:
        return np.concatenate(values, axis)

    def shape_tensor(self, tensor: Tensor) -> Tensor:
        return paddlelib.to_tensor(list(tensor.shape))

    def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
        return tuple(tensor.shape)

    def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
        return self.shape_tuple(tensor)

    def shape_prod(self, values: Tensor) -> int:
        return np.prod(np.array(values))

    def sqrt(self, tensor: Tensor) -> Tensor:
        return paddlelib.sqrt(tensor)

    def convert_to_tensor(self, tensor: Tensor) -> Tensor:
        return tensor

    def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return paddlelib.tensordot(tensor1, tensor2, axes=0)
    
    # pylint: disable=unused-argument
    def einsum(self,
            expression: str,
            *tensors: Tensor,
            optimize: bool = True) -> Tensor:
        return paddlelib.einsum(expression, *tensors)

    def norm(self, tensor: Tensor) -> Tensor:
        return paddlelib.norm(tensor)

    def eye(self,
        N: int,
        dtype: Optional[Any] = None,
        M: Optional[int] = None) -> Tensor:
        dtype = dtype if dtype is not None else paddlelib.float32
        if not M:
            M = N  #torch crashes if one passes M = None with dtype!=None
        return paddlelib.eye(n=N, m=M, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Tensor:
        dtype = dtype if dtype is not None else paddlelib.float32
        return paddlelib.ones(shape, dtype=dtype)

    def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None) -> Tensor:
        dtype = dtype if dtype is not None else paddlelib.float32
        return paddlelib.zeros(shape, dtype=dtype)

    def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None,
            seed: Optional[int] = None) -> Tensor:
        if seed:
            paddlelib.seed(seed)
        dtype = dtype if dtype is not None else paddlelib.float32
        return paddlelib.randn(shape, dtype=dtype)

    def random_uniform(self,
                    shape: Tuple[int, ...],
                    boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                    dtype: Optional[Any] = None,
                    seed: Optional[int] = None) -> Tensor:
        if seed:
            paddlelib.seed(seed)
        dtype = dtype if dtype is not None else paddlelib.float32
        return paddlelib.empty(shape, dtype=dtype).uniform_(*boundaries)

    def conj(self, tensor: Tensor) -> Tensor:
        return tensor.conj()

    def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
        return paddlelib.linalg.eigh(matrix)

    def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 + tensor2

    def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 - tensor2

    def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 * tensor2

    def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 / tensor2

    def index_update(self, tensor: Tensor, mask: Tensor,
                assignee: Tensor) -> Tensor:
        #make a copy
        t = tensor.clone()
        t[mask] = assignee
        return t

    def inv(self, matrix: Tensor) -> Tensor:
        if len(matrix.shape) > 2:
            raise ValueError(
                "input to pytorch backend method `inv` has shape {}. Only matrices are supported."
                .format(matrix.shape))
        return matrix.inverse()

    def broadcast_right_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
        if len(tensor2.shape) != 1:
            raise ValueError(
                "only order-1 tensors are allowed for `tensor2`, found `tensor2.shape = {}`"
                .format(tensor2.shape))

        return tensor1 * tensor2

    def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
        if len(tensor1.shape) != 1:
            raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                    " found `tensor1.shape = {}`".format(tensor1.shape))

        t1_broadcast_shape = self.shape_concat(
            [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
        return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

    def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
        return fun

    def sum(self,
        tensor: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False) -> Tensor:
        return paddlelib.sum(tensor, axis=axis, keepdim=keepdims)

    def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        if (tensor1.ndim <= 1) or (tensor2.ndim <= 1):
            raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")

        return paddlelib.einsum('...ab,...bc->...ac', tensor1, tensor2)

    def diagonal(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
        r"""Return specified diagonals.
        If tensor is 2-D, returns the diagonal of tensor with the given offset,
        i.e., the collection of elements of the form a[i, i+offset].
        If a has more than two dimensions, then the axes specified by
        axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
        returned. The shape of the resulting array can be determined by removing
        axis1 and axis2 and appending an index to the right equal to the size of the
        resulting diagonals.
        This function only extracts diagonals. If you
        wish to create diagonal matrices from vectors, use diagflat.
        
        Args:
            tensor: A tensor.
            offset: Offset of the diagonal from the main diagonal.
            axis1, axis2: Axis to be used as the first/second axis of the 2D
                sub-arrays from which the diagonals should be taken.
                Defaults to second-last and last axis (note this
                differs from the NumPy defaults).
        Returns:
            array_of_diagonals: A dim = min(1, tensor.ndim - 2) tensor storing
                the batched diagonals.
        """
        if axis1 == axis2:
            raise ValueError("axis1={axis1} and axis2={axis2} must be different.")
        return paddlelib.diagonal(tensor, offset=offset, axis1=axis1, axis2=axis2)

    def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
        r""" Flattens tensor and creates a new matrix of zeros with its elements
        on the k'th diagonal.
        
        Args:
            tensor: A tensor.
            k     : The diagonal upon which to place its elements.
        Returns:
            tensor: A new tensor with all zeros save the specified diagonal.
        """
        return paddlelib.diagflat(tensor, offset=k)

    def trace(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
        r"""Return summed entries along diagonals.
        If tensor is 2-D, the sum is over the
        diagonal of tensor with the given offset,
        i.e., the collection of elements of the form a[i, i+offset].
        If a has more than two dimensions, then the axes specified by
        axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
        summed.
        In the paddlepaddle backend the trace is always over the main diagonal of the
        last two entries.
        
        Args:
            tensor: A tensor.
            offset: Offset of the diagonal from the main diagonal.
                    This argument is not supported  by the PyTorch
                    backend and an error will be raised if they are
                    specified.
            axis1, axis2: Axis to be used as the first/second axis of the 2D
                            sub-arrays from which the diagonals should be taken.
                            Defaults to first/second axis.
                            These arguments are not supported by the PyTorch
                            backend and an error will be raised if they are
                            specified.
        Returns:
            array_of_diagonals: The batched summed diagonals.
        """
        if offset != 0:
            errstr = (f"offset = {offset} must be 0 (the default)"
                    f"with Paddle backend.")
            raise NotImplementedError(errstr)
        if axis1 == axis2:
            raise ValueError(f"axis1 = {axis1} cannot equal axis2 = {axis2}")
        N = len(tensor.shape)
        if N > 25:
            raise ValueError(f"Currently only tensors with ndim <= 25 can be traced"
                f"in the Paddle backend (yours was {N})")

        if axis1 < 0:
            axis1 = N+axis1
        if axis2 < 0:
            axis2 = N+axis2

        inds = list(map(chr, range(98, 98+N)))
        indsout = [i for n, i in enumerate(inds) if n not in (axis1, axis2)]
        inds[axis1] = 'a'
        inds[axis2] = 'a'
        return paddlelib.einsum(''.join(inds) + '->' +''.join(indsout), tensor)

    def abs(self, tensor: Tensor) -> Tensor:
        r"""
        Returns the elementwise absolute value of tensor.
        
        Args:
            tensor: An input tensor.
        Returns:
            tensor: Its elementwise absolute value.
        """
        return paddlelib.abs(tensor)

    def sign(self, tensor: Tensor) -> Tensor:
        r"""
        Returns an elementwise tensor with entries
        y[i] = 1, 0, -1 where tensor[i] > 0, == 0, and < 0 respectively.
        For complex input the behaviour of this function may depend on the backend.
        The Paddle version is not implemented in this case.
        
        Args:
            tensor: The input tensor.
        """
        return paddlelib.sign(tensor)

    def item(self, tensor):
        return tensor.item()
