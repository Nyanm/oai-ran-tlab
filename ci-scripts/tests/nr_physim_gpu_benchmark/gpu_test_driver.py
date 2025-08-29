import sys
import subprocess
import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description="Test driver for the GPU benchmark suite.")
    parser.add_argument('--executable', required=True, help="Path to the test_channel_scalability executable")
    cli_args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    matrix_file_path = os.path.join(script_dir, "gpu_test_matrix.yaml")
    
    try:
        with open(matrix_file_path) as f:
            test_matrix = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Test matrix file not found at {matrix_file_path}")
        sys.exit(1)

    failed_tests = []
    
    for test_case in test_matrix['tests']:
        name = test_case['name']
        args = test_case['args']
        
        print(f"\n{'='*80}")
        print(f"--- Running Test Case: {name} ---")
        print(f"{'='*80}\n")
        
        command = [
            "python3",
            os.path.join(script_dir, "run_gpu_benchmark.py"),
            "--executable", cli_args.executable,
            "--args", args,
            "--test-case-name", name 
        ]
        
        result = subprocess.run(command)
        
        if result.returncode != 0:
            print(f"\n--- !!! Test Case FAILED: {name} !!! ---\n")
            failed_tests.append(name)
        else:
            print(f"\n--- Test Case PASSED: {name} ---\n")

    print(f"\n{'='*80}")
    print("--- Test Suite Summary ---")
    if not failed_tests:
        print("\033[92mAll test cases PASSED.\033[0m")
        sys.exit(0)
    else:
        print(f"\033[91mThe following {len(failed_tests)} test case(s) FAILED:\033[0m")
        for test_name in failed_tests:
            print(f"  - {test_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()