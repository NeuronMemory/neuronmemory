"""
NeuronMemory Examples Test Runner

This script runs all examples to verify they work correctly.
Use this to test your NeuronMemory installation and examples.
"""

import asyncio
import sys
import os
import importlib.util
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

async def run_example(example_name: str, example_path: str) -> bool:
    """Run a single example and return success status"""
    try:
        print(f"\n{'='*60}")
        print(f"🧪 Running Example: {example_name}")
        print(f"{'='*60}")
        
        # Import the example module
        spec = importlib.util.spec_from_file_location(example_name, example_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for the main demo function
        demo_functions = [
            'demo_' + example_name.replace('_', '_'),
            'run_' + example_name.replace('_', '_'),
            'main',
            'demo'
        ]
        
        demo_function = None
        for func_name in demo_functions:
            if hasattr(module, func_name):
                demo_function = getattr(module, func_name)
                break
        
        # Try common demo function names
        if not demo_function:
            for attr_name in dir(module):
                if attr_name.startswith('demo_') or attr_name.startswith('run_'):
                    demo_function = getattr(module, attr_name)
                    break
        
        if demo_function and callable(demo_function):
            print(f"▶️  Running {demo_function.__name__}...")
            if asyncio.iscoroutinefunction(demo_function):
                await demo_function()
            else:
                demo_function()
            
            print(f"✅ {example_name} completed successfully!")
            return True
        else:
            print(f"⚠️  No demo function found in {example_name}")
            return False
            
    except Exception as e:
        print(f"❌ {example_name} failed: {e}")
        # Print shorter traceback for readability
        import traceback
        traceback.print_exc(limit=3)
        return False

async def main():
    """Run all NeuronMemory examples"""
    
    print("="*70)
    print("🧠 NeuronMemory Examples Test Runner")
    print("="*70)
    print("This will run all examples to verify they work correctly.")
    print("Each example should complete without errors.")
    
    # Get all Python files in examples directory
    examples_dir = Path(__file__).parent
    example_files = []
    
    for file_path in examples_dir.glob("*.py"):
        if file_path.name not in ["__init__.py", "run_all_examples.py"]:
            example_files.append(file_path)
    
    example_files.sort()
    
    print(f"\nFound {len(example_files)} examples to run:")
    for i, file_path in enumerate(example_files, 1):
        print(f"{i:2d}. {file_path.stem}")
    
    print(f"\nStarting example runs...")
    
    # Track results
    results = {}
    
    # Run each example
    for file_path in example_files:
        example_name = file_path.stem
        success = await run_example(example_name, str(file_path))
        results[example_name] = success
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 Test Results Summary")
    print(f"{'='*70}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    
    if successful < total:
        print(f"❌ Failed: {total - successful}/{total}")
        print("\nFailed examples:")
        for example_name, success in results.items():
            if not success:
                print(f"  • {example_name}")
    
    print(f"\nDetailed Results:")
    for example_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {example_name}")
    
    # Overall result
    if successful == total:
        print(f"\n🎉 All examples completed successfully!")
        print("Your NeuronMemory installation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some examples failed. Check the error messages above.")
        print("This might indicate installation issues or missing dependencies.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 