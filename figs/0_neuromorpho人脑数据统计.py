import os
import matplotlib.pyplot as plt

# 分辨率
plt.rcParams['figure.dpi'] = 800

swc_dir_root = "/data/kfchen/trace_ws/neuron_nmo"
datasets = {}
swc_dirs = os.listdir(swc_dir_root)
swc_dirs = [os.path.join(swc_dir_root, swc_dir, "CNG version") for swc_dir in swc_dirs]
swc_files = []
swc_labels = []
swc_nums = []


for swc_dir in swc_dirs:
    if(not os.path.isdir(swc_dir)):
        continue

    datasets[swc_dir] = {}
    current_swc_files = [os.path.join(swc_dir, swc_file) for swc_file in os.listdir(swc_dir) if swc_file.endswith(".swc")]
    swc_files.extend(current_swc_files)
    datasets[swc_dir]["swc_files"] = current_swc_files
    datasets[swc_dir]["swc_num"] = len(current_swc_files)

    swc_labels.append(swc_dir.split("/")[-2])
    swc_nums.append(len(current_swc_files))

print("Total swc files: ", len(swc_files))
datasets['proposed'] = {}
datasets['proposed']['swc_num'] = 8676
swc_labels.append("proposed")
swc_nums.append(8676)

# 小于3%的数据集合并
swc_labels_new = []
swc_nums_new = []
swc_labels_new.append('others')
swc_nums_new.append(0)

for i in range(len(swc_nums)):
    if(swc_nums[i] < 1000): #  0.01 * (len(swc_files) + 8676)
        swc_nums_new[0] += swc_nums[i]
    else:
        swc_labels_new.append(swc_labels[i])
        swc_nums_new.append(swc_nums[i])
print(swc_labels_new)
print(swc_nums_new)


# # pie cha
# plt.figure(figsize=(4, 4))
# plt.pie(swc_nums_new, labels=swc_labels_new, autopct="%1.1f%%")
# # plt.title("Neuromorpho datasets")
# # 设置字体大小
# # plt.rcParams.update({'font.size': 12})
# plt.show()
# plt.close()

sizes = swc_nums_new
labels = swc_labels_new
colors = plt.cm.get_cmap('Set3').colors[:len(sizes)]
# 简写map
labels_abbr = [label.split(' ')[0].split('_')[0] for label in labels]
# 首字母大写
labels_abbr = [label.capitalize() for label in labels_abbr]
# for i in range(len(labels_abbr)):
#     if not labels_abbr[i] == 'proposed':
#         labels_abbr[i] = labels_abbr[i].capitalize()

sorted_sizes, sorted_labels, sorted_colors = zip(*sorted(zip(sizes, labels_abbr, colors), reverse=False))
sizes, labels_abbr, colors = list(sorted_sizes), list(sorted_labels), list(sorted_colors)
# 创建饼图，不显示百分比和标签
fig, ax = plt.subplots(figsize=(9, 4))
ax.pie(sizes, labels=['' for i in range(len(sizes))], colors=colors, autopct='',
       wedgeprops={'width': 0.5, 'edgecolor': 'black', 'linewidth': 0.5},
       startangle=90)

# 计算每个扇区的百分比并在 legend 中显示
legend_labels = []
for i in range(len(sizes)):
    percentage = sizes[i] / sum(sizes) * 100
    # 创建 legend 标签: 名称、数量、百分比
    legend_labels.append(f'{labels_abbr[i]}; n={sizes[i]} ({percentage:.2f}%)')

# 将 legend 放置在图的右侧
ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=2, frameon=False,
          shadow=True)

plt.subplots_adjust(left=-0.5)
plt.savefig("/data/kfchen/trace_ws/neuron_nmo/neuron_nmo_pie.png", bbox_inches='tight')
# plt.show()
plt.close()

# 总结：样本少，特别是早期实验，基本都是只能使用死亡样本。然后，一般选择Neurolucida or ImageJ及衍生工具进行人工精准标注
# Jacobs: https://academic.oup.com/cercor/article/11/6/558/370645?login=true 2001, 改良的 cresyl echt violet染色，早期工作，手工重建
# PL, FL, STG, IFG, FL, PL, SFG, IFG,

# Ellis: https://www.nature.com/articles/s41593-019-0365-8 2019，免疫荧光标记，样本制备时间超过9周&非常复杂， Simple Neurite Tracer（fiji）半自动重建
# 涉及到样本培养？无法定位到具体的脑区，

# Vdheuvel: https://www.jneurosci.org/content/42/20/4147 2022， 染色，Neurolucida人工重建（看了一下，这个软件还挺不错的，付费）
# 文章付费，暂时没有看到细节

# Allen：https://www.cell.com/neuron/fulltext/S0896-6273(16)30720-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627316307206%3Fshowall%3Dtrue，2018，Biocytin染色，v3d标注
# neuromorpho上面的参考文献是错误的

# Mechawar: https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2021.640963/full, 2021，免疫染色，Neurolucida标注
# FL，其他几个组织不在大脑皮层

# Helmstaedter： https://www.science.org/doi/10.1126/science.abo0924?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed Osmium tetraoxide染色，没有全文访问权限
# TL, FL，同时还做了猴脑的实验

# Ataman: https://www.nature.com/articles/nature20111 2016，免疫染色（RNA），ImageJ 标注
# 同样涉及到神经元培养

