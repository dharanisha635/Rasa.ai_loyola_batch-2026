# Run this to fix the CI in app_fixed.py
code = open('app_fixed.py', encoding='utf-8').read()

old = """def bootstrap_ci(arr, n=100):
    base  = float(model.predict(arr, verbose=0)[0][0])
    preds = []
    for _ in range(n):
        noise = np.random.normal(0, 0.1, arr.shape)
        p     = float(model.predict(arr+noise, verbose=0)[0][0])
        preds.append(p)
    low  = min(np.percentile(preds, 2.5),  base)
    high = max(np.percentile(preds, 97.5), base)
    return (round(np.mean(preds),4), round(low,4), round(high,4))"""

new = """def bootstrap_ci(arr, n=100):
    base = float(model.predict(arr, verbose=0)[0][0])
    preds = []
    for _ in range(n):
        # Use larger noise scaled to feature range
        noise = np.random.normal(0, 0.3, arr.shape)
        noisy = arr + noise
        p = float(model.predict(noisy, verbose=0)[0][0])
        preds.append(p)
    preds.append(base)
    preds = sorted(preds)
    mean  = float(np.mean(preds))
    # Calculate CI around the base prediction
    std   = float(np.std(preds))
    low   = max(0.0, base - 1.96 * std)
    high  = min(1.0, base + 1.96 * std)
    # Ensure minimum spread of 3%
    if high - low < 0.03:
        low  = max(0.0, base - 0.03)
        high = min(1.0, base + 0.03)
    return (round(base,4), round(low,4), round(high,4))"""

if old in code:
    code = code.replace(old, new)
    open('app_fixed.py', 'w', encoding='utf-8').write(code)
    print("Fixed!")
else:
    print("Pattern not found - fixing differently...")
    # Find and replace the whole function differently
    import re
    pattern = r'def bootstrap_ci\(arr, n=100\):.*?return \(round\(.*?\)\)'
    replacement = new
    code_new = re.sub(pattern, replacement, code, flags=re.DOTALL)
    if code_new != code:
        open('app_fixed.py', 'w', encoding='utf-8').write(code_new)
        print("Fixed with regex!")
    else:
        print("Please fix manually - see instructions below")
