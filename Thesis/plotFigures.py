import matplotlib.pyplot as plt
import numpy as np
from click import style
import matplotlib.ticker as ticker

parameter_groups = ["t_Cu", "w_Cu", "t_LamCore", "n_LamCore", "AlN", "t_Su8"]
value_Q_100 = [2.987, 6.282, 2.726, 3.709, 27.665, 1.892]
value_R_100 = [3.059, 5.318, 3.328, 5.363, 159.909, 1.971]
value_L_100 = [0.709, 3.998, 3.860, 2.913, 24.693, 0.782]


# yticsk_number = [0, np.log10(5), np.log10(10), np.log10(20), np.log10(30), np.log10(100), np.log10(300)]
# ytick_label = ["0%", "5%", "10%", "20%", "30%", "100%", "300%"]

bar_width = 0.3
index = np.arange(len(parameter_groups))
# plt.ylim([0,])

plt.figure(figsize=(10, 6))
plt.bar(index, value_Q_100, width=bar_width, edgecolor = "black", color = "#FADCAA", label="Q")
plt.bar(index + bar_width, value_R_100, width=bar_width, edgecolor = "black", color = "#A6D0DD", label="R")
plt.bar(index + 2 * bar_width, value_L_100, width=bar_width, edgecolor = "black", color = "#82A0D8", label="L")
plt.title("Influence of Single Design Parameter on Error Rate", fontsize=14)
# plt.ylim([0,1000000])
# plt.yticks(yticsk_number, ytick_label)
plt.xticks(index + 1 * bar_width, parameter_groups, style="italic")
plt.ylabel("Mean Error Rate", fontsize=12)
plt.xlabel("Design Parameters", fontsize=12)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=5)

ax = plt.gca()
ax.set_yscale("log")

major_ticks = [0,1, 10, 100,1000]
major_labels = ["0%", "1%", "10%","100%", "1000%"]

ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
ax.yaxis.set_major_formatter(ticker.FixedFormatter(major_labels))


# ax.yaxis.set

ax.tick_params(axis='y', which='minor', length=5, color='gray')  # 可以控制虚线长度和颜色
ax.grid(which='minor', axis='y', linestyle='--', color='gray', alpha=0.7)  # 只显示虚线


plt.grid(True)
plt.savefig("matFigures/MeanError.svg", dpi = 800, bbox_inches='tight')
plt.show()


