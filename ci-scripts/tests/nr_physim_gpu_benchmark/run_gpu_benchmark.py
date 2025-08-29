import sys
import json
import subprocess
import argparse
import re
import time
import os
import yaml

def parse_benchmark_output(log_text):
    metrics = {}
    pattern = re.compile(r"\|\s*(?P<key>[\w\s\(\)\-]+?)\s*\|\s*(?P<value>[\d\.]+)x?\s*\|")
    for line in log_text.splitlines():
        match = pattern.search(line)
        if match:
            key = match.group('key').strip()
            value = float(match.group('value'))
            metrics[key] = value
    return metrics

def analyze_gpu_log(log_file):
    util_vals, power_vals = [], []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "utilization.gpu" in line: continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        util_vals.append(float(parts[0].strip()))
                        power_vals.append(float(parts[1].strip()))
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        return "N/A", "N/A"
    if not util_vals or not power_vals:
        return "N/A", "N/A"
    avg_util = sum(util_vals) / len(util_vals)
    max_util = max(util_vals)
    avg_power = sum(power_vals) / len(power_vals)
    max_power = max(power_vals)
    return (f"Avg: {avg_util:.1f}%, Max: {max_util:.1f}%", f"Avg: {avg_power:.1f}W, Max: {max_power:.1f}W")


def main():
    parser = argparse.ArgumentParser(description="Run a single GPU benchmark and compare against a baseline.")
    parser.add_argument('--executable', required=True, help="Path to the test_channel_scalability executable")
    parser.add_argument('--args', required=True, help="Arguments to pass to the executable, in quotes")
    parser.add_argument('--test-case-name', required=True, help="The name of the test case to look up in the baseline file")
    cli_args = parser.parse_args()

    gpu_log_file = "nvidia-smi.log"
    monitor_command = f"nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > {gpu_log_file}"
    monitor_process = subprocess.Popen(monitor_command, shell=True)
    command = [cli_args.executable] + cli_args.args.split()
    print(f"Running command: {' '.join(command)}")
    # result = subprocess.run(command, capture_output=True, text=True)
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    monitor_process.terminate()
    time.sleep(1)

    if result.returncode != 0:
        print("--- Benchmark FAILED to run ---")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("--- Benchmark Ran Successfully ---")
    print(result.stdout)
    
    gpu_util_summary, gpu_power_summary = analyze_gpu_log(gpu_log_file)
    print("\n--- GPU Monitoring Summary ---")
    print(f"GPU Utilization: {gpu_util_summary}")
    print(f"GPU Power Draw:  {gpu_power_summary}")

    measured_metrics = parse_benchmark_output(result.stdout)
    if not measured_metrics:
        print("ERROR: Could not parse any metrics from the benchmark's text output.")
        sys.exit(1)

    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        baseline_file_path = os.path.join(script_dir, "gpu_performance_baselines.yaml")
        with open(baseline_file_path) as f:
            all_baselines = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Master baseline file not found at {baseline_file_path}")
        sys.exit(1)

    baseline_metrics = all_baselines.get(cli_args.test_case_name)
    if not baseline_metrics:
        print(f"ERROR: Could not find baseline entry for '{cli_args.test_case_name}' in {baseline_file_path}")
        sys.exit(1)
        
    all_metrics_ok = True
    print("\n--- Performance Validation ---")
    for key, baseline_value in baseline_metrics.items():
        if key == 'Threshold': continue

        if key in measured_metrics:
            measured_value = measured_metrics[key]
            threshold = baseline_metrics.get('Threshold', 0.25)
            upper_bound = baseline_value * (1 + threshold)

            print(f"Metric: '{key}'")
            print(f"  - Measured: {measured_value:.2f}")
            print(f"  - Baseline: {baseline_value:.2f}")
            print(f"  - Allowed Upper Bound: {upper_bound:.2f}")

            if measured_value > upper_bound:
                print("  - STATUS: \033[91mFAILED (Exceeded threshold)\033[0m")
                all_metrics_ok = False
            else:
                print("  - STATUS: \033[92mPASSED\033[0m")
        else:
            print(f"Metric '{key}' from baseline not found in run output. SKIPPING.")

    if all_metrics_ok:
        print("\nOverall Performance: PASSED")
        sys.exit(0)
    else:
        print("\nOverall Performance: FAILED due to regression.")
        sys.exit(1)

if __name__ == "__main__":
    main()