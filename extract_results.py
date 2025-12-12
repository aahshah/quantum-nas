import re

# Read the last run output
try:
    with open('verification_results.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract final results
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'FINAL VERIFICATION RESULTS' in line:
            # Print next 15 lines
            for j in range(i, min(i+15, len(lines))):
                print(lines[j])
            break
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    
    # Just run verify_real.py and capture output
    import subprocess
    result = subprocess.run(['python', 'verify_real.py'], 
                          capture_output=True, text=True, encoding='utf-8')
    
    # Find and print final results
    output_lines = result.stdout.split('\n')
    for i, line in enumerate(output_lines):
        if 'FINAL' in line or 'Acc=' in line or 'SUCCESS' in line:
            print(line)
