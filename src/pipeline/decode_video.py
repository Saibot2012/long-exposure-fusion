import subprocess
from pathlib import Path

import src.utils.utils as utils
from src.utils.ImageStore import ImageStore

DECODED_DIRNAME = Path("decoded/")

def decode(video_path: Path) -> ImageStore:
    """
    Decodes a video file into a sequence of PNG images into the corresponding cache under decoded/.
    Returns an ImageStore containing the decoded frames.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    if not any(str(video_path).lower().endswith(extension) for extension in utils.VALID_VIDEO_EXTENSIONS):
        raise ValueError(f"Input file {video_path} is not a supported video format.")
    
    cache = ImageStore.create_cache(video_path).child(DECODED_DIRNAME)

    if not cache.is_empty():
        print(f"[INFO] Using cached frames in {cache.path}.")
        return cache

    cache.path.mkdir(parents=True, exist_ok=True)
    ffmpeg_command = [
        "ffmpeg", "-i", str(video_path),
        "-start_number", "0",
        str(cache.path / "%06d.png")
    ]
    print(f"[INFO] Decoding video frames with ffmpeg: {' '.join(ffmpeg_command)}.")
    subprocess.run(ffmpeg_command, check=True)
    return cache

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decode a video into PNG frames.")
    parser.add_argument("video_path", type=Path, help="Path to the input video file.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and re-decode video.")
    args = parser.parse_args()

    if args.clear_cache:
        cache = ImageStore.create_cache(args.video_path).child(DECODED_DIRNAME)
        print(f"[INFO] Clearing cache at {cache.path}.")
        cache.clear()

    output = decode(args.video_path)
    print(f"Frames saved to: {output.path}.")

if __name__ == "__main__":
    main()