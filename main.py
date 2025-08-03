from pathlib import Path
import shutil
import json
import colorsys
import argparse
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC

def extract_metadata(path):
    info = {
        "title": None,
        "artist": None,
        "duration": None,
        "cover": None
    }

    try:
        audio = MP3(path, ID3=ID3)

        if "TIT2" in audio:
            info["title"] = audio["TIT2"].text[0]
        if "TPE1" in audio:
            info["artist"] = audio["TPE1"].text[0]

        info["duration"] = int(audio.info.length)

        for tag in audio.tags.values():
            if isinstance(tag, APIC):
                info["cover"] = Image.open(BytesIO(tag.data)).convert("RGB")
                break

    except Exception as e:
        print(f"[error] couldn't extract info: {e}")

    return info

def crop_image(image):
    w, h = image.size
    s = min(w, h)
    return image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))

def get_accent_color(image):
    small_img = image.resize((10, 10))
    best_color = (255, 255, 255)
    best_score = 0

    def contrast_against_black(r, g, b):
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    for pixel in small_img.getdata():
        r, g, b = pixel
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        if v < 0.2: continue

        vibrancy = s * v
        contrast = contrast_against_black(r, g, b)

        score = vibrancy * 0.6 + contrast * 0.4

        if score > best_score:
            best_score = score
            best_color = (r, g, b)

    h, s, v = colorsys.rgb_to_hsv(*[x / 255.0 for x in best_color])

    s = min(s * 1.1, 1.0)
    v = min(v * 1.1, 1.0)

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in (r, g, b))

def adaptive_threshold(image_array, low_frac=0.3, high_frac=0.7):
    brightness = np.mean(image_array, axis=2)
    flat = brightness.flatten()
    sorted_vals = np.sort(flat)
    low_thresh = sorted_vals[int(len(sorted_vals) * low_frac)]
    high_thresh = sorted_vals[int(len(sorted_vals) * high_frac)]
    return low_thresh, high_thresh

def sharpen_image(image, accent):
    arr = np.array(image)
    low, high = adaptive_threshold(arr)
    brightness = np.mean(arr, axis=2)
    output = np.zeros_like(arr)

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            b = brightness[y, x]
            if b < low:
                output[y, x] = (0, 0, 0)
            elif b > high:
                output[y, x] = (255, 255, 255)
            else:
                output[y, x] = accent

    return Image.fromarray(output)

def enhance_contrast(image):
    arr = np.array(image).astype(np.float32) / 255.0
    min_val = np.percentile(arr, 2)
    max_val = np.percentile(arr, 98)
    arr = np.clip((arr - min_val) / (max_val - min_val), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def rgb888_to_rgb565(r, g, b):
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

def process_mp3(path: Path):
    metadata = extract_metadata(path)

    if not metadata:
        print("no metadata found")
        return

    # crop and resize image
    cover = crop_image(metadata["cover"])
    cover = cover.resize((86, 86), Image.LANCZOS)

    # extract dominant accent color
    accent = get_accent_color(cover)
    print(f"dominant color is {accent}")

    # enhance contrast
    cover = enhance_contrast(cover)
    cover = cover.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=2))

    # sharpen with adaptive thresholding
    cover = sharpen_image(cover, accent)

    return cover, accent, metadata

def write_mp3(path: Path, output_dir: Path, playlist: Path = None):
    if not path.is_file() or path.suffix.lower() not in [".mp3", ".wav"]:
        print(f"skipped {path.name}")
        return

    mp3_dir = output_dir / path.stem

    # don't write if mp3 folder already exists
    if mp3_dir.exists():
        print(f"\"{path.name}\" already exists, skipping...")
    else:
        print(f"adding \"{path.name}\" to library...")
        mp3_dir.mkdir(parents=True, exist_ok=True)

        # write mp3 file and metadata to individual folders
        shutil.copy(path, mp3_dir / ("audio" + path.suffix.lower()))
        cover, accent, metadata = process_mp3(path)

        # write cover to .raw file
        with open(mp3_dir / "cover.raw", "wb") as f:
            for pixel in cover.getdata():
                r = pixel[0] >> 3
                g = pixel[1] >> 2
                b = pixel[2] >> 3
                value = (r << 11) | (g << 5) | b
                f.write(value.to_bytes(2, "little"))
        del metadata["cover"]

        # write accent color to metadata
        metadata["color"] = rgb888_to_rgb565(*accent)

        # write metadata to json file
        with open(mp3_dir / "track.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # add to playlist if necessary
    if playlist:
        # add track to playlist json file
        if playlist.exists():
            with open(playlist, "r") as f:
                playlist_data = json.load(f)
        else:
            playlist_data = { "tracks": [] }

        if mp3_dir.name in playlist_data["tracks"]:
            print(f"\"{path.name}\" already in playlist, skipping...")
        else:
            print(f"adding \"{path.name}\" to playlist \"{playlist.stem}\"...")

            playlist_data["tracks"].append(mp3_dir.name)
            with open(playlist, "w") as f:
                json.dump(playlist_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="manage mp3 files and metadata in the yume library")
    parser.add_argument("input", help="path to the mp3 file or directory containing mp3 files to add")
    parser.add_argument("output", help="directory to copy the files to", default="/Volumes/YUME")
    parser.add_argument("--playlist", "-p", help="name of the playlist file to add the mp3 files to", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    playlist = Path(output_dir / (args.playlist + ".json")) if args.playlist else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_file():
        write_mp3(input_path, output_dir, playlist)
    else:
        # add all mp3 files in a directory
        print(f"adding all mp3 files in \"{input_path}\" to library...")
        for item in input_path.iterdir():
            write_mp3(item, output_dir, playlist)

    print("~ enjoy! ~")
