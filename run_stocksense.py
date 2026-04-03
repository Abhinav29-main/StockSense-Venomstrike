import sys

# Check Python version
if sys.version_info.major == 3 and sys.version_info.minor >= 13:
    print("=" * 60)
    print("❌ ERROR: Python 3.13 is not supported yet!")
    print("=" * 60)
    print("Please install Python 3.11 or 3.12:")
    print("https://www.python.org/downloads/release/python-31111/")
    print("=" * 60)
    input("Press Enter to exit...")
    sys.exit(1)
import subprocess, sys, os, time, webbrowser, importlib

PACKAGES = ["flask", "flask_cors", "yfinance", "transformers", "torch", "sklearn", "numpy", "pandas", "feedparser", "requests"]
PIP_NAMES = ["flask", "flask-cors", "yfinance", "transformers", "torch", "scikit-learn", "numpy", "pandas", "feedparser", "requests"]

print("=" * 50)
print("  StockSense — Starting up")
print("=" * 50)

missing = []
for pkg, pip_name in zip(PACKAGES, PIP_NAMES):
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pip_name)

if missing:
    print(f"\nInstalling: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing, stdout=subprocess.DEVNULL)
    print("Done.")

here = os.path.dirname(os.path.abspath(__file__))
backend = os.path.join(here, "stocksense_backend.py")
frontend = os.path.join(here, "stocksense_frontend.html")

print("\nStarting backend (FinBERT loads on first search ~20s)...")
proc = subprocess.Popen([sys.executable, backend], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for _ in range(20):
    time.sleep(0.5)
    if proc.poll() is not None:
        print("Backend failed to start. Check stocksense_backend.py is in the same folder.")
        sys.exit(1)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5050/api/health", timeout=1)
        break
    except:
        pass

print("Backend ready at http://localhost:5050")
print(f"Opening frontend...")
webbrowser.open(f"file://{frontend}")
print("\nPress Ctrl+C to stop.\n")

try:
    for line in proc.stdout:
        print(line, end="")
except KeyboardInterrupt:
    proc.terminate()
    print("\nStopped.")
