# image_fusion.py

import torch
from tqdm import tqdm
from pathlib import Path

from src.utils.pyramids import compute_gaussian_pyramid, compute_laplacian_pyramid, collapse_pyramid
from src.utils.weight_map import WeightMap
from src.utils.ImageStore import ImageStore

BATCH_SIZE = 4
OUTPUT_DIRNAME = Path("fused/")

def fuse(source: ImageStore, weight_maps: dict[str, WeightMap], n_levels: int = 1) -> None:
    """
    Exposure fusion for an iterator of images, one at a time, with late normalization.
    image_tensor_iterator: generator of torch tensors [C,H,W]
    weight_map_fn: function(contrast, saturation, well_exposedness, index) -> weight map for each image
    do_pyramid_decomposition: if False, skip pyramid blending and do simple weighted average
    """
    do_pyramid_decomposition = n_levels > 1
    cache = source.cache.child(OUTPUT_DIRNAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    levels = list(range(n_levels))
    blended_pyramid = None  # Will be [n_maps][levels][C,h,w]
    weight_sum_pyramid = None    # Will be [n_maps][levels][1,h,w]
    image_index = 0

    image_count = source.get_image_count()
    if image_count == 0:
        raise ValueError("No images provided to image_fusion.")
    
    progress_bar = tqdm(total=image_count, desc="Fusing images", unit="images")
    for batch in source.images(batch_size=BATCH_SIZE):  # [N,C,H,W]
        batch = batch.to(device) # [N,C,H,W]
        
        if do_pyramid_decomposition:
            img_gaussian_pyramid = compute_gaussian_pyramid(batch, n_levels=n_levels) # list of [N,C,H,W]
            img_laplacian_pyramid = compute_laplacian_pyramid(img_gaussian_pyramid) # list of [N,C,H,W]
        else:
            # Skip pyramid decomposition - work directly with original images
            img_laplacian_pyramid = [batch]  # Single level containing original images
            n_levels = 1
            levels = [0]

        if blended_pyramid is None:
            assert weight_sum_pyramid is None
            blended_pyramid = [
                [torch.zeros_like(img_laplacian_pyramid[k][0], device=device) for k in levels]
                for _ in range(len(weight_maps))
            ]  # [n_maps][levels][C,h,w]
            weight_sum_pyramid = [
                [torch.zeros_like(img_laplacian_pyramid[k][0], device=device) for k in levels]
                for _ in range(len(weight_maps))
            ]  # [n_maps][levels][1,h,w]

        for weight_map_index, weight_map in enumerate(weight_maps.values()):
            weights = weight_map(batch, image_index)  # [N,1,H,W]
            weights = torch.clamp(weights, min=1e-12, max=1e6)

            weight_gaussian_pyramid = compute_gaussian_pyramid(weights, n_levels) if do_pyramid_decomposition else [weights]  # list of [N,1,H,W]
            for k in levels:
                blended_pyramid[weight_map_index][k] += torch.sum(weight_gaussian_pyramid[k] * img_laplacian_pyramid[k], dim=0)  # [C,h,w]
                weight_sum_pyramid[weight_map_index][k] += torch.sum(weight_gaussian_pyramid[k], dim=0)  # [1,h,w]

        image_index += batch.shape[0]  # Increment by batch size
        progress_bar.update(batch.shape[0])
    progress_bar.close()

    for weight_map_index, weight_map_name in enumerate(weight_maps.keys()):
        normalized_blended_pyramid = [blended_pyramid[weight_map_index][k] / weight_sum_pyramid[weight_map_index][k] for k in levels]  # [C,h,w]
        normalized_blended_pyramid = [level.unsqueeze(0) for level in normalized_blended_pyramid]
        if do_pyramid_decomposition:
            fused_image = collapse_pyramid(normalized_blended_pyramid).squeeze(0)
        else:
            fused_image = normalized_blended_pyramid[0].squeeze(0)

        cache.save_image(fused_image, Path(weight_map_name + ".png"))

    return cache
