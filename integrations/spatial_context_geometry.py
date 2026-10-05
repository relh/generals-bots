"""The two explicitly supported spatial actor neighborhoods."""


def context_offsets(radius):
    if isinstance(radius, bool) or radius not in (1.01, 2.01):
        raise ValueError("Spatial context requires radius 1.01 or 2.01")
    extent = int(radius)
    return tuple((dy, dx) for dy in range(-extent, extent + 1)
                 for dx in range(-extent, extent + 1)
                 if dy * dy + dx * dx <= extent * extent)


def kernel_offsets(size):
    if size not in (3, 5):
        raise ValueError("Spatial context requires a 3x3 or 5x5 kernel")
    extent = size // 2
    return tuple((dy + extent, dx + extent)
                 for dy, dx in context_offsets(extent + .01))
