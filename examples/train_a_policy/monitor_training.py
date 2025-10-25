#!/usr/bin/env python3
"""
Monitor training progress in real-time
"""
import os
import time
import sys

def monitor_training(config_folder, update_interval=5):
    """
    Monitor training by tailing the log file
    """
    log_file = os.path.join(config_folder, "progress.csv")
    
    print(f"Monitoring training in: {config_folder}")
    print(f"Tensorboard logs: {config_folder}/")
    print(f"To view tensorboard: tensorboard --logdir={config_folder}")
    print("=" * 80)
    print()
    
    if not os.path.exists(log_file):
        print(f"Waiting for training to start (looking for {log_file})...")
        while not os.path.exists(log_file):
            time.sleep(2)
        print("Training started!")
        print()
    
    # Read and display log file
    last_position = 0
    header_printed = False
    
    try:
        while True:
            with open(log_file, 'r') as f:
                f.seek(last_position)
                lines = f.readlines()
                last_position = f.tell()
                
                for line in lines:
                    if not header_printed and line.startswith('r,'):
                        # Print header
                        print(line.strip())
                        print("-" * 80)
                        header_printed = True
                    elif not line.startswith('r,'):
                        # Print data
                        print(line.strip())
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        print(f"\nTo view detailed logs with tensorboard:")
        print(f"  tensorboard --logdir={config_folder}")

if __name__ == "__main__":
    config_folder = "carracing/no_shield/seed1"
    if len(sys.argv) > 1:
        config_folder = sys.argv[1]
    
    monitor_training(config_folder)
