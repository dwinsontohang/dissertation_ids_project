import csv, math, os

OPS = 'evaluation_results/ops_metrics.csv'
SYS = 'evaluation_results/sysmon.csv'

def pct(data, p):
    data = sorted(data)
    k = (len(data)-1)*(p/100)
    a, b = int(math.floor(k)), int(math.ceil(k))
    return data[a] if a==b else data[a]*(b-k)+data[b]*(k-a)

def avg(a): return sum(a)/len(a) if a else 0.0

# --- read ops metrics ---
latencies, ingest_ts = [], []
with open(OPS, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            latencies.append(float(row['latency_ms']))
            ingest_ts.append(int(row['ingest_ts']))
        except Exception:
            pass

if not latencies:
    print("No decisions found in", OPS)
    raise SystemExit(0)

n = len(latencies)
t0, t1 = min(ingest_ts), max(ingest_ts)
dur_s = max(1e-9, (t1 - t0)/1000.0)
throughput = n / dur_s

# --- read sysmon in same window ---
proc_cpu = []; sys_cpu = []; proc_rss = []
if os.path.exists(SYS):
    with open(SYS, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row['ts'])
                if t0 <= t <= t1:
                    proc_cpu.append(float(row['proc_cpu_percent']))
                    sys_cpu.append(float(row['sys_cpu_percent']))
                    proc_rss.append(float(row['proc_rss_mb']))
            except Exception:
                pass

print("Operational Performance Summary")
print(f"- Decisions: {n} over {dur_s:.1f}s  → Throughput ≈ {throughput:.2f} decisions/s")
print("- Decision latency (ms): "
      f"p50={pct(latencies,50):.1f}, p95={pct(latencies,95):.1f}, "
      f"avg={avg(latencies):.1f}, max={max(latencies):.1f}")

if proc_cpu:
    print("- Resource usage (same window):")
    print(f"  • Process CPU%: avg={avg(proc_cpu):.1f}, max={max(proc_cpu):.1f}")
    print(f"  • System  CPU%: avg={avg(sys_cpu):.1f}, max={max(sys_cpu):.1f}")
    print(f"  • Process RSS : avg={avg(proc_rss):.2f} MB, max={max(proc_rss):.2f} MB")
else:
    print("- Resource usage: no sysmon rows in the same time window (run the probe longer if needed).")

