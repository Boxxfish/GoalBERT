from matplotlib import pyplot as plt

LABELS = [
    "1",
    "2",
    "3",
    "4",
    "6",
    "All",
]

# Sentence EM
EM = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

# Sentence F1
F1 = [
    1.0,
    6.7,
    10.0,
    12.8,
    16.1,
    23.3
]

import matplotlib
matplotlib.use("TkAgg")

plt.title("# of [MASK]s vs. F1")
plt.plot(LABELS, F1)
plt.xlabel("# of [MASK]s")
plt.ylabel("F1")
plt.savefig("f1_masks.pdf")
plt.cla()

plt.title("# of [MASK]s vs. EM")
plt.plot(LABELS, EM)
plt.xlabel("# of [MASK]s")
plt.ylabel("EM")
plt.savefig("em_masks.pdf")