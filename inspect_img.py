import json
import subprocess

result = subprocess.run(['docker', 'image', 'inspect', 'artery-stenosis-backend:optimized'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error: {result.stderr}")
    exit(1)

img = json.loads(result.stdout)[0]
print(f"Image ID: {img['Id'][:19]}...")
print(f"Size: {img['Size']:,} bytes ({img['Size'] / 1024 / 1024:.2f} MB, {img['Size'] / 1024 / 1024 / 1024:.2f} GB)")
print(f"Created: {img['Created']}")
print(f"Architecture: {img['Architecture']}")
print(f"Os: {img['Os']}")
print()
print("Layer History:")
for layer in img.get('RootFS', {}).get('Layers', []):
    size = layer.get('Size', 0)
    print(f"  {size:>12,} bytes ({size / 1024 / 1024:.1f} MB)")