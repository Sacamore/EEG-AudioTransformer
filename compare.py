import matplotlib.pyplot as plt
import numpy as np

# 数据
subs = [f'sub-{i:02d}' for i in range(1, 11)]
linear_regression_means = [0.495404, 0.614066, 0.828894, 0.766885, 0.523527, 0.854428, 0.700102, 0.703663, 0.638907, 0.680938]
mywork_means = [0.631172, 0.661815, 0.838076, 0.809031, 0.626120, 0.854478, 0.787099, 0.746497, 0.725005, 0.820085]
mywork_maxes = [0.659414, 0.680198, 0.849987, 0.822582, 0.642836, 0.865612, 0.810954, 0.762955, 0.746951, 0.828626]

# 配置图表
x = np.arange(len(subs))
width = 0.25  # 每组柱子的宽度

fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(x - width, linear_regression_means, width, label='Linear Regression Mean')
bar2 = ax.bar(x, mywork_means, width, label='Mywork 1000 Epoch Mean')
bar3 = ax.bar(x + width, mywork_maxes, width, label='Mywork 1000 Epoch Max')

# 标签和标题
ax.set_xlabel('Subject')
ax.set_ylabel('Correlation')
ax.set_title('Comparison of Mean and Max Correlation for Linear Regression and Mywork')
ax.set_xticks(x)
ax.set_xticklabels(subs)
ax.legend()

# 显示每个柱状条上的数值
for bars in [bar1, bar2, bar3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('./fold10res/comp.png')
plt.close()

print(np.mean(linear_regression_means),np.mean(mywork_maxes),np.mean(mywork_means))