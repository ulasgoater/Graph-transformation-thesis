# test_determinism.py
"""
Verify that the pipeline produces identical outputs across multiple runs.
This is a critical requirement for thesis validation.
.\venv\Scripts\Activate.ps1
"""
import hashlib
import sys
from pathlib import Path
from run_pipeline import main
from config import OUT_NODES, OUT_EDGES, OUT_GRAPHML

def hash_file(filepath):
    """Compute MD5 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def hash_outputs():
    """Hash all output files."""
    hashes = {}
    if OUT_NODES.exists():
        hashes['nodes'] = hash_file(OUT_NODES)
    if OUT_EDGES.exists():
        hashes['edges'] = hash_file(OUT_EDGES)
    if OUT_GRAPHML.exists():
        hashes['graphml'] = hash_file(OUT_GRAPHML)
    return hashes

def test_determinism(num_runs=3):
    """Run pipeline multiple times and verify identical outputs."""
    print(f"Testing determinism with {num_runs} runs...")
    print("=" * 60)
    
    # Use fixed test bbox (Milano)
    test_bbox = [45.386, 9.040, 45.535, 9.278]
    
    all_hashes = []
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}...")
        try:
            main(run_test_subset=True, bbox=test_bbox, prompt_bbox=False)
            hashes = hash_outputs()
            all_hashes.append(hashes)
            print(f"  Nodes hash:   {hashes.get('nodes', 'N/A')}")
            print(f"  Edges hash:   {hashes.get('edges', 'N/A')}")
            print(f"  GraphML hash: {hashes.get('graphml', 'N/A')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("Comparing outputs...")
    
    # Check if all runs produced identical hashes
    first_hashes = all_hashes[0]
    for i, hashes in enumerate(all_hashes[1:], start=2):
        if hashes != first_hashes:
            print(f"❌ FAILED: Run {i} differs from Run 1")
            print(f"   Run 1: {first_hashes}")
            print(f"   Run {i}: {hashes}")
            return False
    
    print("✅ SUCCESS: All runs produced identical outputs!")
    print(f"   Pipeline is deterministic across {num_runs} runs.")
    return True

if __name__ == "__main__":
    success = test_determinism(num_runs=3)
    sys.exit(0 if success else 1)
