import einops
from torch import Tensor
import torch
import typing

class EinChain:
    x: Tensor
    current_pattern: str

    def __init__(self, x: Tensor, pattern : str) -> None:
        num_dims = len(x.shape)
        pattern_dims = len(pattern.strip().split(' '))
        if num_dims != pattern_dims:
            raise ValueError(f"Number of dimensions in the tensor ({num_dims}) does not match the number of dimensions in the pattern ({pattern_dims})")
        self.x = x
        self.current_pattern = pattern

    @typing.overload
    def einsum(self, tensor: Tensor, target_pattern: str, /) -> 'EinChain': ...
    @typing.overload
    def einsum(self, tensor1: Tensor, tensor2: Tensor, target_pattern: str, /) -> 'EinChain': ...
    @typing.overload
    def einsum(self, tensor1: Tensor, tensor2: Tensor, tensor3: Tensor, target_pattern: str, /) -> 'EinChain': ...
    @typing.overload
    def einsum(self, tensor1: Tensor, tensor2: Tensor, tensor3: Tensor, tensor4: Tensor, target_pattern: str, /) -> 'EinChain': ...
    def einsum(self, *target_tensors_and_pattern: Tensor | str) -> 'EinChain':
        target_pattern = target_tensors_and_pattern[-1]
        tensors = [self.x] + list(target_tensors_and_pattern[:-1])
        pattern, new_current_pattern = self.transformation_to(target_pattern)
        res = einops.einsum(*tensors, pattern)
        return EinChain(res, new_current_pattern)

    def rearrange(self, target_pattern: str, **axes_lengths: int) -> 'EinChain':
        pattern, new_current_pattern = self.transformation_to(target_pattern)
        res = einops.rearrange(self.x, pattern, **axes_lengths)
        return EinChain(res, new_current_pattern)

    def reduce(self, target_pattern: str, reduction: str, **axes_lengths: int) -> 'EinChain':
        pattern, new_current_pattern = self.transformation_to(target_pattern)
        res = einops.reduce(self.x, pattern, reduction, **axes_lengths)
        return EinChain(res, new_current_pattern)

    def repeat(self, target_pattern: str, **axes_lengths: int) -> 'EinChain':
        pattern, new_current_pattern = self.transformation_to(target_pattern)
        res = einops.repeat(self.x, pattern, **axes_lengths)
        return EinChain(res, new_current_pattern)

    def transformation_to(self, target_pattern: str) -> tuple[str, str]:
        end_pattern = target_pattern.split('->')[-1].strip()

        if target_pattern.strip().startswith(','):
            return f"{self.current_pattern}{target_pattern}", end_pattern
        if target_pattern.strip().startswith('->'):
            return f"{self.current_pattern} {target_pattern}", end_pattern
        raise ValueError()

    def tensor(self) -> Tensor:
        return self.x

    def __repr__(self) -> str:
        return f"EinChain({self._format_dims()})"

    def _format_dims(self):
        dims = [
            f"{dim_name}={dim}" if not dim_name.isdigit() else str(dim)
            for dim_name, dim in zip(self.current_pattern.split(' '), self.x.shape)
        ]
        return ', '.join(dims)


if __name__ == "__main__":
    bchw = torch.randn(2, 3, 4, 5)
    hwc = torch.randn(4, 5, 3)

    t = EinChain(bchw, 'b c h w')\
        .rearrange("-> b h w c")\
        .einsum(hwc, ", h w c -> b h w")\
        .rearrange("-> b h w 1")\
        .repeat("-> b h w 3")\
    
    print(t)
    


