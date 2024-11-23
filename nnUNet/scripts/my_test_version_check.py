import os

# origin_anno_dir = r"/data/kfchen/trace_ws/paper_trace_result/manual/origin_anno"
origin_anno_dir = r"/data/kfchen/trace_ws/paper_trace_result/manual/one_checked_anno"
my_using_anno_dir = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/label/origin_eswc"

origin_anno_files = os.listdir(origin_anno_dir)
my_using_anno_files = os.listdir(my_using_anno_dir)

origin_ids = [int(file.split("_")[0].split('.')[0]) for file in origin_anno_files]
my_using_ids = [int(file.split("_")[0].split('.')[0]) for file in my_using_anno_files]

common_ids = list(set(origin_ids).intersection(set(my_using_ids)))
# print("Common IDs: ", common_ids)
print("Common IDs Length: ", len(common_ids))
common_file_list, no_common_file_list = [], []

for origin_anno_file in origin_anno_files:
    origin_id = int(origin_anno_file.split("_")[0].split('.')[0])
    if origin_id not in common_ids:
        continue
    my_using_anno_file = f"{origin_id}.eswc"
    # origin_anno, my_using_anno = None, None

    # md5sum
    origin_md5sum = os.popen(f'md5sum "{os.path.join(origin_anno_dir, origin_anno_file)}"').read().split()[0]
    my_using_md5sum = os.popen(f'md5sum "{os.path.join(my_using_anno_dir, my_using_anno_file)}"').read().split()[0]

    if origin_md5sum == my_using_md5sum:
        common_file_list.append(origin_anno_file)
        # print(f"Common: {origin_anno_file}")
    else:
        no_common_file_list.append(origin_anno_file)
        print(f"No Common: {origin_anno_file}")

print("Common File List: ", common_file_list)
print("No Common File List: ", no_common_file_list)

print("Common File List Length: ", len(common_file_list))
print("No Common File List Length: ", len(no_common_file_list))

'''
原始标注
Common File List Length:  1368
No Common File List Length:  9

一次检查
Common File List Length:  3
No Common File List Length:  493

两次检查
Common File List Length:  0
No Common File List Length:  397
'''

'''
和原始重建不同
['02989_P025_T02_-S009_LTL_R0460_YW-20230204_NYT_SXZ.eswc', 
'05987_P031_T02_(3)-S005__RTL_R0919_YS-20230522_YW_SXZ.eswc', 
'05989_P031_T02_(3)-S005__RTL_R0919_YS-20230522_YW_SXZ.eswc', 
'05988_P031_T02_(3)-S005__RTL_R0919_YS-20230522_YW_SXZ.eswc', 
'02997_P025_T02_-S009_LTL_R0613_YW-20230204_NYT_SXZ.eswc', 
'02998_P025_T02_-S001_LTL_R0460_YW-20230204_OMZ_SXZ.eswc', 
'02948_P025_T02_-S062_LTL_R0460_YW-20230204_HZY_SXZ.eswc', 
'05986_P031_T02_(3)-S005__RTL_R0919_YS-20230522_YW_SXZ.eswc', 
'04944_P019_T02_-S005_FP.R_R0460_YS-20230426_YS_SXZ.eswc']
'''

'''
和一次检查相同
['02396_P023_T01-S020_LIFG_R0613_LJ-20221127_LD_SBJ_CX.eswc', 
'02426_P024_T01_(2)-S036_LFL_R0919_RJ-20230201_YW_JFX_CX.eswc', 
'02375_P021_T01-S048_RFL_R0613_LJ-20221103_YW_LTC_CX.eswc']
'''
