import matplotlib.pyplot as plt
import csv

sizes = []
averages = []
trials_1 = []
trials_2 = []
trials_3 = []

with open("blocking_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sizes.append(float(row["Size_MB"]))
        trials_1.append(float(row["Trial1_s"]))
        trials_2.append(float(row["Trial2_s"]))
        trials_3.append(float(row["Trial3_s"]))
        averages.append(float(row["Average_s"]))

plt.figure(figsize=(8, 5), dpi=300)
plt.plot(sizes, averages, marker='o', linewidth=2.2, color='#1f77b4', label='Average Communication Time')
plt.scatter(sizes, trials_1, color='#ff7f0e', alpha=0.7, s=40, label='Trial 1')
plt.scatter(sizes, trials_2, color='#2ca02c', alpha=0.7, s=40, label='Trial 2')
plt.scatter(sizes, trials_3, color='#d62728', alpha=0.7, s=40, label='Trial 3')

for s, avg in zip(sizes, averages):
    plt.annotate(f"{avg*1000:.2f} ms", (s, avg), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

plt.title("MPI Blocking Communication: Message Size vs. Communication Time", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Message Size (MB)", fontsize=11)
plt.ylabel("Communication Time (seconds)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks([1, 10, 20, 30, 40, 50])
plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.savefig("blocking_communication_time.png")
print("Plot saved successfully to 'blocking_communication_time.png'")

