"""
Download a small sample of the PlantVillage dataset.

The full PlantVillage dataset is about 800 MB, so this script fetches
only a small subset: 90 images each of 3 tomato classes
(healthy, bacterial spot, early blight) from the official public
GitHub repo (spMohanty/PlantVillage-Dataset, CC0 license).
No login needed. Images land in data/<class>/ and are ignored by git.
"""

import json
import os
import urllib.parse
import urllib.request

CLASSES = ["Tomato___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight"]
IMAGES_PER_CLASS = 90
BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def list_files(cls):
    url = (
        "https://api.github.com/repos/spMohanty/PlantVillage-Dataset"
        f"/contents/raw/color/{cls}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ml-projects"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        entries = json.load(resp)
    return [e["name"] for e in entries if e["type"] == "file"]


def main():
    for cls in CLASSES:
        out_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        names = list_files(cls)
        print(f"{cls}: {len(names)} images available, downloading {IMAGES_PER_CLASS}...")
        done = 0
        for name in names:
            dest = os.path.join(out_dir, name)
            if os.path.exists(dest):
                done += 1
                continue
            url = f"{BASE}/{cls}/{urllib.parse.quote(name)}"
            try:
                urllib.request.urlretrieve(url, dest)
                done += 1
            except Exception as e:
                print(f"  skipped {name}: {e}")
            if done >= IMAGES_PER_CLASS:
                break
        print(f"  -> {done} images in {out_dir}")


if __name__ == "__main__":
    main()
