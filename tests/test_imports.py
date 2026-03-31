#!/usr/bin/env python3
"""
Quick test to verify all imports work.
Run this if you're still seeing Pylance errors.
"""

import sys

print("=" * 60)
print("IMPORT TEST")
print("=" * 60)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print()

tests = [
    ("torch", "torch"),
    ("numpy", "np"),
    ("yaml", "yaml"),
    ("tqdm", "tqdm"),
    ("traci", "traci"),
]

all_ok = True
for module_name, import_as in tests:
    try:
        module = __import__(module_name)
        if hasattr(module, "__version__"):
            version = module.__version__
            print(f"✓ {module_name:<20} {version}")
        else:
            print(f"✓ {module_name:<20} installed")
    except ImportError as e:
        print(f"✗ {module_name:<20} ERROR: {e}")
        all_ok = False

print()
print("=" * 60)
if all_ok:
    print("✅ All imports successful!")
    print()
    print("If Pylance still shows errors:")
    print("  1. Press Cmd+Shift+P")
    print("  2. Type 'Pylance: Clear Cache'")
    print("  3. Press Enter")
    print("  4. Reload VS Code: Cmd+Shift+P > 'Reload Window'")
else:
    print("❌ Some imports failed!")
    print("Install missing packages with: pip install -r requirements.txt")
print("=" * 60)
