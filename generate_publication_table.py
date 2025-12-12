import json

def generate_markdown_table():
    with open('multi_noise_results.json') as f:
        data = json.load(f)
    
    print("### Noise Resilience Results (Signal Fidelity at 10% Noise)")
    print("")
    print("| Noise Model | Noise Level | Bayesian (Depth 6) | Ours (Depth 2) | Improvement |")
    print("|-------------|-------------|-------------------|----------------|-------------|")
    
    titles = {
        'depolarizing': 'Depolarizing',
        'amplitude_damping': 'Amplitude Damping (T1)',
        'phase_damping': 'Phase Damping (T2)'
    }
    
    for nt in ['depolarizing', 'amplitude_damping', 'phase_damping']:
        res = data['results'][nt]
        d6 = res['depth6'][-1]
        d2 = res['depth2'][-1]
        imp = d2 / d6 if d6 > 0 else 0
        
        # Bold the improvement if it's significant
        imp_str = f"**{imp:.1f}x**" if imp >= 1.5 else f"{imp:.1f}x"
        
        print(f"| {titles[nt]} | 10% | {d6:.3f} | {d2:.3f} | {imp_str} |")

if __name__ == "__main__":
    generate_markdown_table()
