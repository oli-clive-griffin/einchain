# einchain


```python
from einchain import EinChain

bchw = torch.randn(2, 3, 4, 5)
hwc = torch.randn(4, 5, 3)

x = (
    EinChain(bchw, 'b c h w')
        .rearrange("b h w c")
        .einsum("self, h w c -> b h w", hwc)
        .rearrange("b h w 1")
        .repeat("b h w 3")
        .tensor()
)

# is equivalent to:
x = einops.repeat(
    einops.rearrange(
        einops.einsum(
            einops.rearrange(bchw, 'b c h w -> b h w c'),
            hwc,
            'b h w c, h w c -> b h w'),
        'b h w -> b h w 1'
    ),
    'b h w 1 -> b h w 3'
)

# or more charitably
x = einops.rearrange(bchw, 'b c h w -> b h w c')
x = einops.einsum(x, hwc, 'b h w c, h w c -> b h w')
x = einops.rearrange(x, 'b h w -> b h w 1')
x = einops.repeat(x, 'b h w 1 -> b h w 3')
```