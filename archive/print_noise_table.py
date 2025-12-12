import json

with open('multi_noise_results.json') as f:
    data = json.load(f)

print(f"{'Noise Model':<20} | {'Level':<6} | {'Bayesian':<10} | {'Ours':<10} | {'Improvement':<10}")
print("-" * 66)

for nt, res in data['results'].items():
    d6 = res['depth6'][-1]
    d2 = res['depth2'][-1]
    imp = d2 / d6 if d6 > 0 else 0
    print(f"{nt:<20} | 10%    | {d6:<10.3f} | {d2:<10.3f} | {imp:<10.1f}x")
