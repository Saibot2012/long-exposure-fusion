from collections.abc import Iterable
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from tqdm import tqdm
import cv2
import shapely
import largestinteriorrectangle
from lightglue import LightGlue, DISK
import torch
import kornia

from src.utils.ImageStore import ImageStore

ALIGNED_DIRNAME = Path("aligned/")
CROPPED_DIRNAME = Path("cropped/")
REFERENCE_INDEX_FILENAME = "reference.txt"

INTERSECTION_THRESHOLD = 0.75
MAX_NUM_KEYPOINTS = 12000
FEATURE_EXTRACTION_METHOD = 'disk'
HOMOGRAPHY_METHOD = cv2.USAC_MAGSAC
RANSAC_REPROJECTION_THRESHOLD = 3.0

def align(source: ImageStore, reference_index: int) -> ImageStore:
    """
    Aligns images to the reference image and crops them into the given cache under cropped/.
    Returns a directory containing the cropped images. Some images may be discarded.
    """

    aligned_cache = source.cache.child(ALIGNED_DIRNAME)
    cropped_cache = source.cache.child(CROPPED_DIRNAME)

    if cropped_cache.get_entry("reference_index") == reference_index:
        print(f"[INFO] Using cached cropped images from {cropped_cache.path}.")
        return cropped_cache

    # Align images
    indexed_image_paths = source.get_indexed_image_filenames()
    aligned_cache.clear()
    transformed_polygons = _align_images(
        images=(source.load_image(indexed_image_paths[i]) for i in range(max(indexed_image_paths) + 1)),
        reference_image=source.load_image(indexed_image_paths[reference_index]),
        destination=aligned_cache,
        image_count=len(indexed_image_paths)
    )

    # Crop images
    indexed_image_paths = aligned_cache.get_indexed_image_filenames()
    cropped_cache.clear()
    _crop_aligned_images(
        images=(source.load_image(indexed_image_paths[i]) for i in range(max(indexed_image_paths) + 1)),
        polygons=transformed_polygons,
        reference_index=reference_index,
        destination=cropped_cache,
        image_count=len(indexed_image_paths),
    )
    cropped_cache.save_entry("reference_index", reference_index)

    return cropped_cache

def _align_images(
    images: Iterable[torch.Tensor],
    reference_image: torch.Tensor,
    destination: ImageStore,
    image_count: int = None
) -> dict[int, Polygon]:
    print(f"[INFO] Aligning images to reference using DISK+LightGlue (feature matching only). Saving to {destination.path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = DISK(max_num_keypoints=MAX_NUM_KEYPOINTS).eval().to(device)
    matcher = LightGlue(features=FEATURE_EXTRACTION_METHOD).eval().to(device)

    reference_image = reference_image.to(device)
    reference_features = extractor.extract(reference_image)
    height, width = reference_image.shape[1:]

    reference_corners = torch.tensor(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        device=device,
        dtype=torch.float32
    )
    transformed_polygons = {}
    discarded = []
    for i, image in tqdm(
        iterable=((i, image.to(device)) for i, image in enumerate(images) if image is not None),
        total=image_count,
        desc="Aligning images",
        unit="img"
    ):
        target_features = extractor.extract(image)
        matches = matcher({"image0": reference_features, "image1": target_features})["matches"][0]
        reference_keypoints = reference_features["keypoints"][0]
        target_keypoints = target_features["keypoints"][0]

        if len(matches) < 4:
            discarded.append(i)
            continue

        H, _ = cv2.findHomography(
            target_keypoints[matches[:, 1]].cpu().numpy(),
            reference_keypoints[matches[:, 0]].cpu().numpy(),
            method=HOMOGRAPHY_METHOD,
            ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD
        )
        if H is None:
            discarded.append(i)
            continue

        # Transform image and corners
        # kornia is slightly faster than cv2 here.
        H = torch.from_numpy(H).to(device).float().unsqueeze(0)
        aligned = kornia.geometry.transform.warp_perspective(image.unsqueeze(0), H, dsize=(height, width))[0]
        corners = kornia.geometry.transform_points(H, reference_corners.unsqueeze(0))[0]
        transformed_polygons[i] = Polygon(corners.cpu().numpy())
        destination.save_image_at(aligned.cpu(), i)

    if len(discarded):
        print(f"[WARN] Some images could not be aligned: {discarded}.")
        print(f"[WARN] Total discarded: {len(discarded)} out of {image_count} images ({len(discarded) / image_count:.1%}).")
    print(f"[INFO] Alignment complete. {image_count - len(discarded)} aligned images saved to {destination.path}.")

    return transformed_polygons

def _crop_aligned_images(
    images: Iterable[torch.Tensor],
    polygons: dict[int, Polygon],
    reference_index: int,
    destination: ImageStore,
    image_count: int = None
) -> int:
    reference_polygon = polygons[reference_index]
    reference_area = reference_polygon.area

    def is_valid_polygon(polygon):
        if polygon is None or not polygon.is_valid:
            return False
        intersection_ratio = polygon.intersection(reference_polygon).area / reference_area
        return intersection_ratio >= INTERSECTION_THRESHOLD

    discarded = [i for i, polygon in polygons.items() if not is_valid_polygon(polygon)]
    if len(discarded):
        print(f"[WARN] Some images could not be cropped: {discarded}.")
        print(f"[WARN] Total discarded: {len(discarded)} out of {len(polygons)} images ({len(discarded) / len(polygons):.1%}).")
    polygons = {i: polygon for i, polygon in polygons.items() if i not in discarded}

    # Crop and save images
    common = shapely.intersection_all(list(polygons.values()))
    x, y, w, h = largestinteriorrectangle.lir(np.asarray(common.exterior.coords, np.int32)[None, :, :]).astype(int)
    print(f"[INFO] Cropping region from LIR: x={x}, y={y}, w={w}, h={h}.")
    for i, image in tqdm(
        ((i, image) for i, image in enumerate(images) if image is not None and i in polygons),
        total=image_count - len(discarded),
        desc="Cropping images",
        unit="img"
    ):
        cropped = image[:, y:y+h, x:x+w]
        destination.save_image_at(cropped, i)
    print(f"[INFO] Cropping complete. {len(polygons)} cropped images saved to {destination.path}.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align and crop a sequence of images.")
    parser.add_argument("source", type=ImageStore, help="Directory containing images to align.")
    parser.add_argument("--reference", type=int, default=0, help="Index of reference image. (default: 0)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and re-align frames.")
    args = parser.parse_args()

    if args.clear_cache:
        aligned_cache = args.source.cache.child(ALIGNED_DIRNAME)
        cropped_cache = args.source.cache.child(CROPPED_DIRNAME)
        print(f"[INFO] Clearing cache at {aligned_cache.path}.")
        aligned_cache.clear()
        print(f"[INFO] Clearing cache at {cropped_cache.path}.")
        cropped_cache.clear()

    align(args.image_dir, args.reference)

if __name__ == "__main__":
    main()
