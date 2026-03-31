#!/usr/bin/env python3
"""Validate and setup the DQN Traffic Light Control project."""

import os
import sys
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are installed."""
    print("🔍 Checking Python packages...")
    required = ["torch", "numpy", "yaml", "tqdm"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def check_sumo():
    """Check if SUMO is properly installed."""
    print("\n🔍 Checking SUMO installation...")
    
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        print("  ✗ SUMO_HOME not set")
        print("  Set with: export SUMO_HOME=/path/to/sumo/share/sumo")
        return False
    
    print(f"  ✓ SUMO_HOME: {sumo_home}")
    
    # Check if traci is available
    try:
        import traci
        print("  ✓ traci module")
    except ImportError:
        print("  ✗ traci not available (install sumolib and traci)")
        return False
    
    return True


def check_scenario_files():
    """Check if SUMO scenario files exist."""
    print("\n🔍 Checking scenario files...")
    
    scenario_dir = Path("data/scenarios/hn_sample")
    required_files = [
        "config.sumocfg",
        "intersection.net.xml",
        "routes.rou.xml",
        "nodes.nod.xml",
        "edges.edg.xml"
    ]
    
    missing = []
    for file in required_files:
        path = scenario_dir / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing.append(file)
    
    if missing:
        print("\n⚠️  Missing scenario files. Generate with:")
        print("  python src/utils/generate_scenario.py")
        if "intersection.net.xml" in missing:
            print("\nThen build network with:")
            print("  netconvert --node-files=data/scenarios/hn_sample/nodes.nod.xml \\")
            print("             --edge-files=data/scenarios/hn_sample/edges.edg.xml \\")
            print("             --output-file=data/scenarios/hn_sample/intersection.net.xml")
        return False
    
    return True


def check_project_structure():
    """Check if project structure is correct."""
    print("\n🔍 Checking project structure...")
    
    required_paths = [
        "src/dqn/model.py",
        "src/dqn/agent.py",
        "src/dqn/replay_buffer.py",
        "src/env/sumo_env.py",
        "src/utils/schedules.py",
        "scripts/train.py",
        "scripts/validate.py",
        "config.yaml",
        "requirements.txt"
    ]
    
    all_ok = True
    for path in required_paths:
        if Path(path).exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")
            all_ok = False
    
    return all_ok


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("DQN Traffic Light Control - Project Validation")
    print("=" * 60)
    
    checks = [
        ("Python Packages", check_python_packages),
        ("SUMO Installation", check_sumo),
        ("Scenario Files", check_scenario_files),
        ("Project Structure", check_project_structure)
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed! Ready to train.")
        print("\nQuick start:")
        print("  ./run.sh validate           # Validate setup (this)")
        print("  ./run.sh train              # Start training")
        print("  ./run.sh demo               # Run demo simulation")
        print("  ./run.sh dqn                # Run trained DQN in GUI")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
