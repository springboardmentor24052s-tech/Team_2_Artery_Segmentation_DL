import json
import subprocess

result = subprocess.run(['docker', 'image', 'inspect', 'a5924fba8196'], capture_output=True, text=True)
if result.returncode != 0:
    print("Error:", result.stderr)
    exit(1)

img = json.loads(result.stdout)[0]
print(f"Image ID: {img['Id'][:19]}...")
print(f"Size: {img.get('Size', 0):,} bytes ({img.get('Size', 0) / 1024 / 1024:.2f} MB, {img.get('Size', 0) / 1024 / 1024 / 1024:.2f} GB)")
print(f"RepoTags: {img.get('RepoTags', [])}")
print(f"Config.User: {img.get('Config', {}).get('User', 'root')}")
print(f"Config.Cmd: {c.get('Config', {}).get('Cmd', [])}")
print()
print("Config.Env:")
for e in c.get('Config', {}).get('Env', []):
    print(f"  {e}")
print()
print("Config.Healthcheck:")
print(json.dumps(c.get('Config', {}).get('Healthcheck', {}), indent=2))