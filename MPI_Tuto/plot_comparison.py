import matplotlib.pyplot as plt
import csv
import numpy as np

sizes = []
comm_only = []
comp_only = []
blocking_total = []
nonblocking_total = []

with open("nonblocking_comparison.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sizes.append(row["Size"])
        comm_only.append(float(row["Comm_Only_s"]) * 1000.0) # ms
        comp_only.append(float(row["Comp_Only_s"]) * 1000.0)
        blocking_total.append(float(row["Blocking_Total_s"]) * 1000.0)
        nonblocking_total.append(float(row["NonBlocking_Total_s"]) * 1000.0)

x = np.arange(len(sizes))
width = 0.35

plt.figure(figsize=(9, 5), dpi=300)
rects1 = plt.bar(x - width/2, blocking_total, width, label='Sequential (Blocking Comm + Comp)', color='#d62728', alpha=0.85)
rects2 = plt.bar(x + width/2, nonblocking_total, width, label='Overlapped (Non-Blocking Isend/Irecv + Comp)', color='#2ca02c', alpha=0.85)

plt.ylabel('Elapsed Time (milliseconds)', fontsize=11)
plt.title('Problem 3: Computation-Communication Overlap Comparison', fontsize=12, fontweight='bold', pad=12)
plt.xticks(x, sizes, fontsize=10)
plt.legend(frameon=True, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f} ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("nonblocking_overlap_comparison.png")
print("Saved nonblocking comparison plot to 'nonblocking_overlap_comparison.png'")

