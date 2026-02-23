import json, sys

f = sys.argv[1]
data = json.load(open(f))

celr = data['celr_results']
direct = data['direct_results']

celr_acc = sum(1 for r in celr if r['accuracy'])
direct_acc = sum(1 for r in direct if r['accuracy'])

print(f"CELR Model:   {data['model']}")
print(f"Direct Model: {data['direct_model']}")
print(f"CELR:   {celr_acc}/12")
print(f"Direct: {direct_acc}/12")
print()
print(f"{'Task':<22} {'CELR':>6} {'Direct':>6}  Steps  Retries")
print("-" * 60)

for r, d in zip(celr, direct):
    c = "PASS" if r['accuracy'] else "FAIL"
    dr = "PASS" if d['accuracy'] else "FAIL"
    print(f"{r['task_id']:<22} {c:>6} {dr:>6}  {r['steps_executed']:>5}  {r['retries']:>5}")
