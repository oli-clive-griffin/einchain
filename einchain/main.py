import einops
from torch import Tensor
import torch

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

    def einsum(self, pattern: str, *given_tensors: Tensor) -> 'EinChain':
        tensors = [self.x] + list(given_tensors)
        res = einops.einsum(*tensors, self._transformation_with_self(pattern))
        return EinChain(res, self._end_pattern(pattern))

    def rearrange(self, target_pattern: str, **axes_lengths: int) -> 'EinChain':
        pattern = self._transformation_to(target_pattern)
        res = einops.rearrange(self.x, pattern, **axes_lengths)
        return EinChain(res, self._end_pattern(target_pattern))

    def reduce(self, target_pattern: str, reduction: str, **axes_lengths: int) -> 'EinChain':
        pattern = self._transformation_to(target_pattern)
        res = einops.reduce(self.x, pattern, reduction, **axes_lengths)
        return EinChain(res, self._end_pattern(target_pattern))

    def repeat(self, target_pattern: str, **axes_lengths: int) -> 'EinChain':
        pattern = self._transformation_to(target_pattern)
        res = einops.repeat(self.x, pattern, **axes_lengths)
        return EinChain(res, self._end_pattern(target_pattern))

    def _end_pattern(self, target_pattern: str) -> str:
        return target_pattern.split('->')[-1].strip()

    def _transformation_with_self(self, pattern: str) -> str:
        return pattern.replace('self', self.current_pattern)

    def _transformation_to(self, target_pattern: str) -> str:
        if not target_pattern.strip().startswith('->'):
            raise ValueError()
        return f"{self.current_pattern} {target_pattern}"

    def tensor(self) -> Tensor:
        return self.x

    def __repr__(self) -> str:
        dims_str = ', '.join(
            f"{dim_name}={dim}" if not dim_name.isdigit() else str(dim)
            for dim_name, dim in zip(self.current_pattern.split(' '), self.x.shape)
        )
        return f"EinChain({dims_str})"


if __name__ == "__main__":
    bchw = torch.randn(2, 3, 4, 5)
    hwc = torch.randn(4, 5, 3)

    t = (EinChain(bchw, 'b c h w')
        .rearrange("-> b h w c")
        .einsum("self, h w c -> b h w", hwc)
        .rearrange("-> b h w 1")
        .repeat("-> b h w 3"))

    print(t)
