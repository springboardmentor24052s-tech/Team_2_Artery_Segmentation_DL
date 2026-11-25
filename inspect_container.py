import json
import subprocess

result = subprocess.run(['docker', 'inspect', 'artery-backend'], capture_output=True, text=True)
if result.returncode != 0:
    print("Error:", result.stderr)
    exit(1)

c = json.loads(result.stdout)[0]
print(f"Name: {c.get('Name', 'N/A')}")
print(f"Image: {c.get('Image', 'N/A')[:19]}...")
print(f"Status: {c.get('State', {}).get('Status', 'N/A')}")
h = c.get('State', {}).get('Health', {})
print(f"Health Status: {h.get('Status', 'N/A')}")
print(f"Health Log (last 3):")
for log in h.get('Log', [])[-3:]:
    print(f"  {log.get('Start', '')} - {log.get('Status', '')} - {log.get('Output', '')[:100]}")
print()
print("Config.Env:")
for e in c.get('Config', {}).get('Env', []):
    print(f"  {e}")
print()
print(f"Config.Cmd: {c.get('Config', {}).get('Cmd', [])}")
print(f"Config.User: {c.get('Config', {}).get('User', 'root')}")
print(f"Config.WorkingDir: {c.get('Config', {}).get('WorkingDir', 'N/A')}")
print(f"Config.ExposedPorts: {c.get('Config', {}).get('ExposedPorts', {})}")
print(f"HostConfig.PortBindings: {c.get('HostConfig', {}).get('PortBindings', {})}")
print(f"Mounts:")
for m in c.get('Mounts', []):
    print(f"  {m.get('Name', 'N/A')} -> {m.get('Destination', 'N/A')} ({m.get('Type', 'N/A')})")