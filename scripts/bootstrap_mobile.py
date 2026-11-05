"""Generate Android platform files with the installed Flutter SDK without overwriting lib/."""
from pathlib import Path
import shutil
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
flutter=shutil.which("flutter")
if not flutter: raise SystemExit("Install Flutter stable and add flutter to PATH first.")
if not (root/"mobile/android").exists():
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)/"kulitku"
        subprocess.run([flutter,"create","--platforms","android","--project-name","kulitku","--org","id.ac.felix",str(target)],check=True)
        shutil.copytree(target/"android",root/"mobile/android")
        shutil.copy2(target/".metadata",root/"mobile/.metadata")
manifest=root/"mobile/android/app/src/main/AndroidManifest.xml"
text=manifest.read_text()
if "android.permission.INTERNET" not in text:
    text=text.replace("<application",'<uses-permission android:name="android.permission.INTERNET"/>\n    <application',1)
manifest.write_text(text)
# Cleartext allowed ONLY in debug builds to reach a local development API.
debug=root/"mobile/android/app/src/debug/AndroidManifest.xml"
debug.parent.mkdir(parents=True,exist_ok=True)
debug.write_text('<manifest xmlns:android="http://schemas.android.com/apk/res/android"><uses-permission android:name="android.permission.INTERNET"/><application android:usesCleartextTraffic="true"/></manifest>')
subprocess.run([flutter,"pub","get"],cwd=root/"mobile",check=True)
print("Android scaffold ready. Use HTTPS for release builds.")
