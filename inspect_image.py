import json
import subprocess

result = subprocess.run(['docker', 'image', 'inspect', 'artery-stenosis-backend:latest'], capture_output=True, text=True)
if result.returncode != 0:
    print("Image not found:", result.stderr)
    sys.exit(1)

import sys
images = json.loads(result.stdout)
for img in images:
    print(f"Image ID: {img.get('Id', 'N/A')[:19]}...")
    print(f"RepoTags: {img.get('RepoTags', [])}")
    size_bytes = img.get('Size', 0)
    print(f"Size: {size_bytes:,} bytes ({size_bytes / 1024 / 1024:.2f} MB, {size_bytes / 1024 / 1024 / 1024:.2f} GB)")
    print(f"Created: {img.get('Created', 'N/A')}")
    print(f"Architecture: {img.get('Architecture', 'N/A')}")
    print(f"Os: {img.get('Os', 'N/A')}")
    print()

    print("Layer History:")
    for layer in img.get('RootFS', {}).get('Layers', []):
        layer_size = layer.get('Size', 0)
        print(f"  {layer_size:>12,} bytes ({layer_size / 1024 / 1024:.1f} MB)")