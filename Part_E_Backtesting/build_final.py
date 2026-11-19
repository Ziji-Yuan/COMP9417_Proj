"""Build a self-contained integration candidate for the group's final packager."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/part_e_final"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    output = args.output.resolve()
    package = output / "package"
    package.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    for name in ["model.py", "train.py"]:
        shutil.copy2(here / "final_package" / name, package / name)
    helpers = package / "Part_A"
    helpers.mkdir()
    for name in ["__init__.py", "data.py", "features.py"]:
        shutil.copy2(here.parent / "Part_A" / name, helpers / name)
    subprocess.run([sys.executable, str(package / "train.py"), str(args.data.resolve())], check=True, cwd=package)
    destination = output / "candidate_submission.zip"
    included = [package / "model.py", package / "train.py", *helpers.glob("*.py"), *(package / "artifacts").glob("*")]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(included):
            archive.write(path, str(path.relative_to(package)))
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in included}
    (output / "package_manifest.json").write_text(json.dumps({"candidate_only": True, "files_sha256": hashes,
                                                              "zip_bytes": destination.stat().st_size}, indent=2) + "\n")
    print(f"Candidate built: {destination}")


if __name__ == "__main__":
    main()
