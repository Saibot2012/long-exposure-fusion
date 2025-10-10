"""
long_exposure_fusion.py
"""

from __future__ import annotations

import shutil
import argparse
import numpy
from pathlib import Path
from attr import dataclass

from src.utils.ImageStore import ImageStore
import src.utils.weight_map as weight_map
import src.pipeline.decode_video as decode_video
import src.pipeline.align_images as align_images
import src.pipeline.interpolate_images as interpolate_images
import src.pipeline.fuse_images as fuse_images
import src.pipeline.segment_picker as segment_picker

# ---------------------------------------------------------------------------#
# Constants
# ---------------------------------------------------------------------------#
# Batch size for image fusion
BATCH_SIZE = 4
# Batch size for fusion maps
WEIGHT_MAP_BATCH_SIZE = 32

# Cropping thresholds
INTERSECTION_THRESHOLD = 0.75  # Discard images with intersection below this threshold
DISCARD_RATIO_THRESHOLD = 0.1  # Abort if more than 10% of images are discarded

@dataclass
class LongExposureFusionConfig:
    reference_index: int
    weight_map_file: Path
    align: bool = False
    interpolate: int = None
    use_pyramid_decomposition: bool = False

    def __post_init__(self):
        if len(self.weight_maps) == 0:
            raise ValueError("At least one weight map must be provided.")

def run_long_exposure_fusion(
    source: ImageStore,
    config: LongExposureFusionConfig,
) -> ImageStore:
    # Optionally align
    if config.align:
        source = align_images.align(source, config.reference_index)
        config.reference_index = [
            int(path.stem) for path in source.get_image_filenames()
        ].index(config.reference_index)

    # Optionally interpolate
    if config.interpolate is not None:
        print(f"[INFO] Interpolating {config.interpolate}x frames using Practical-RIFE.")
        source = interpolate_images.interpolate(source, multi=config.interpolate)
        config.reference_index *= config.interpolate

    # Define segmentation masks
    segment_picker.run_segment_picker(source)

    if config.weight_map_file is None:
        raise ValueError("A weight map file must be provided. Use --maps <file>.")

    # Prepare dictionary of keys and functions for batch fusion
    # Initialize WeightMapGenerator with mask_dir, reference_index, and frame_count
    decoder = weight_map.WeightMapGenerator(
        source=source,
        reference_index=config.reference_index,
        frame_count=len(source.get_image_filenames())
    )
    weight_map_items = list(decoder.from_yaml_file(config.weight_map_file).items())

    # Fuse images using exposure fusion
    n_levels = 1
    if config.use_pyramid_decomposition:
        # Calculate pyramid levels based on image dimensions
        height, width = source.load_image_at(0).shape[1:3]
        n_levels = int(numpy.floor(numpy.log2(min(height, width)))) - 1

    for i in range(0, len(weight_map_items), WEIGHT_MAP_BATCH_SIZE):
        print(f"[INFO] Processing weight map batch {i // WEIGHT_MAP_BATCH_SIZE + 1} / {(len(weight_map_items) - 1) // WEIGHT_MAP_BATCH_SIZE + 1}")
        weight_map_batch = dict(weight_map_items[i:i + WEIGHT_MAP_BATCH_SIZE])
        destination = fuse_images.fuse(
            source,
            weight_map_batch,
            n_levels=n_levels,
        )

    indexed_filenames = source.get_indexed_image_filenames()
    source.copy_image_to(indexed_filenames[0], destination, Path("first.png"))
    source.copy_image_to(indexed_filenames[len(indexed_filenames) - 1], destination, Path("last.png"))
    source.copy_image_to(indexed_filenames[config.reference_index], destination, Path("reference.png"))

    return destination

# ---------------------------------------------------------------------------#
# Argument parsing
# ---------------------------------------------------------------------------#
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a burst of photos into a long-exposure image."
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs='?',
        help="Directory of images or path to a video file.",
    )
    parser.add_argument(
        "-m",
        "--maps",
        type=Path,
        help="Path to a YAML file containing weight map definitions to be added to the fusion."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for fused images (default: '.cache/<input_name>_<hash>/fused').",
    )
    parser.add_argument(
        "--reference",
        type=float,
        default=0,
        help="Index or ratio of reference image for alignment. If int >= 1, treated as index. If float in [0,1], treated as ratio of image count (default: 0).",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Align frames before fusion.",
    )
    parser.add_argument(
        "--interpolate",
        type=int,
        default=None,
        help="Interpolate intermediate frames (RIFE --multi argument, default: None=no interpolation).",
    )
    parser.add_argument(
        "--pyramid",
        action="store_true",
        help="Enable pyramid decomposition for blending (default: False, uses simple weighted average)."
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache and re-process all frames.",
    )

    args = parser.parse_args()
    
    # Handle clearing all caches when no input is provided
    if args.input is None and args.clear_cache:
        print("[INFO] Clearing all caches.")
        ImageStore.clear_all_caches()
        return args
    
    # Input is required for normal operation
    if args.input is None:
        parser.error("Input is required unless using --clear-cache to clear all caches.")
    
    if args.reference < 0:
        raise ValueError("Reference must be non-negative (index or ratio).")
    if args.interpolate is not None and args.interpolate < 1:
        raise ValueError("Interpolation must be at least 1.")
    
    if args.clear_cache:
        cache = ImageStore.create_cache(args.input)
        print(f"[INFO] Clearing cache at {cache.path}.")
        cache.clear()

    return args

# ---------------------------------------------------------------------------#
# Main entry‑point
# ---------------------------------------------------------------------------#
def main() -> None:
    """Parse arguments and run the long-exposure pipeline."""
    args = _parse_args()

    if args.input is None:
        return

    if args.input.is_file():
        args.input = decode_video.decode(args.input)
    else:
        args.input = ImageStore(args.input)

    image_count = len(args.input.get_image_filenames())
    if args.reference >= 1:
        # Treat as index
        args.reference = int(args.reference)
        if args.reference >= image_count:
            raise ValueError(f"Reference index {args.reference} is out of range for {image_count} images")
    else:
        # Treat as ratio in [0, 1[
        args.reference = int(args.reference * (image_count - 1))
        print(f"[INFO] Using image {args.reference}/{image_count}) as reference")

    config = LongExposureFusionConfig(
        reference_index=args.reference,
        weight_map_file=args.maps,
        align=args.align,
        interpolate=args.interpolate,
        use_pyramid_decomposition=args.pyramid,
    )

    output_cache = run_long_exposure_fusion(
        source=args.input,
        config=config,
    )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for path in output_cache.path.iterdir():
            if path.is_file():
                shutil.copy2(path, args.output)

    print(f"[INFO] Fused images saved to {args.output or output_cache.path}")

# ---------------------------------------------------------------------------#
# Script execution
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
