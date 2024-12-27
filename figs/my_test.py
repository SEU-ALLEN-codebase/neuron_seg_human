# import numpy as np
# from scipy import stats
#
# # 示例数据
# type_a_values = np.array([1479.52,2185.84,1666.74,1719.2,2809.76])
# type_b_values = np.array([1559.71,2313.69,1625.84,1801.73,2990.35])
# # type_b_values = np.array([5, 4, 3, 2, 1])
#
# # 计算皮尔逊相关系数和p值
# # corr, p_value = stats.spearmanr(type_a_values, type_b_values)
# chi2_stat, p_value = stats.chisquare(f_obs=type_a_values, f_exp=type_b_values)
# # t_stat, p_value = stats.ttest_rel(type_a_values, type_b_values)
#
# # print("Pearson correlation coefficient:", corr)
# # print("P-value:", p_value)
# # print("Spearman correlation coefficient:", corr)
# # print("P-value:", p_value)
# print("Chi-square statistic:", chi2_stat)
# print("P-value:", p_value)
#
# # print("T-statistic:", t_stat)
# # print("P-value:", p_value)

import numpy as np
from scipy import stats

# 观察到的频数
observed = [9, 10, 11] * 10

# 期望的频数（假设均匀分布）
expected = [10, 10, 10] * 10

# 进行卡方检验
chi2_stat, p_val = stats.chisquare(observed, expected / np.sum(expected) * np.sum(observed))

# 输出结果
print("Chi-squared statistic:", chi2_stat)
print("p-value:", p_val)

# 判断是否显著
if p_val < 0.05:
    print("拒绝零假设，存在显著差异")
else:
    print("无法拒绝零假设，没有显著差异")
