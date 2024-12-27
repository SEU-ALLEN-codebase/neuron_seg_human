# 找到所有被重建的数据，然后
import os
import pandas as pd
from sympy.physics.units import percent
import seaborn as sns
import matplotlib.pyplot as plt


def get_final_id_list(final_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"):

    if(not os.path.exists(final_recon_list_file)):
        recon_swc_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc"
        recon_swc_list = [f for f in os.listdir(recon_swc_dir) if f.endswith(".swc")]
        ids = [f.split("_")[0] for f in recon_swc_list]
        ids = [int(i) for i in ids]
        print(len(ids))

        good_sample_list_file = r"/data/kfchen/trace_ws/paper_trace_result/good_sample_list.csv"
        good_sample_list = pd.read_csv(good_sample_list_file)["id"].tolist()
        good_sample_ids = [int(i) for i in good_sample_list]
        print(len(good_sample_ids))

        muti_neuron_list_file = r"/data/kfchen/trace_ws/paper_trace_result/mutineuron_list.csv"
        muti_neuron_list = pd.read_csv(muti_neuron_list_file)["id"].tolist()
        muti_neuron_ids = [int(i) for i in muti_neuron_list]
        print(len(muti_neuron_ids))

        # in good_sample and not in muti_neuron
        recon_ids = list(set(ids).intersection(set(good_sample_ids)).difference(set(muti_neuron_ids)))
        recon_ids.sort()
        print(len(recon_ids))

        final_id_list = [str(i) for i in recon_ids]
        final_id_list_df = pd.DataFrame(final_id_list, columns=["id"])
        final_id_list_df.to_csv(final_recon_list_file, index=False)
    else:
        print("final_recon_list_file exists")
        final_id_list_df = pd.read_csv(final_recon_list_file)
        final_id_list = final_id_list_df["id"].tolist()
    print(f"final_id_list: {len(final_id_list)}")

    return final_id_list

def check_final_list(final_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"):
    final_id_list_df = pd.read_csv(final_recon_list_file)
    final_id_list = final_id_list_df["id"].tolist()
    final_id_list = [int(i) for i in final_id_list]

    train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train_val_list.csv"
    train_val_list = pd.read_csv(train_val_list_file)["id"].tolist()
    train_val_ids = [int(i) for i in train_val_list]

    test_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv"
    test_list = pd.read_csv(test_list_file)["id"].tolist()
    test_ids = [int(i) for i in test_list]

    unlabel_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_without_gs.csv"
    unlabel_list = pd.read_csv(unlabel_list_file)["id"].tolist()
    unlabel_ids = [int(i) for i in unlabel_list]

    muti_neuron_list_file = r"/data/kfchen/trace_ws/paper_trace_result/mutineuron_list.csv"
    muti_neuron_list = pd.read_csv(muti_neuron_list_file)["id"].tolist()
    muti_neuron_ids = [int(i) for i in muti_neuron_list]



    set1 = set(final_id_list)
    set2 = set(test_ids) | set(train_val_ids) | set(unlabel_ids)
    print(len(set1))
    print(len(set2))

    # 检查不一样的
    print(set1.difference(set2))
    print(len(set1.difference(set2)))
    print(set2.difference(set1))
    print(len(set2.difference(set1)))

    print(len(train_val_ids), len(test_ids), len(unlabel_ids))

    """
    {6008, 2578, 2796, 6007}重建出来train_val有，但是final里面没有
    都不是多neuron
    6007 6008不是good sample, 2578 2796看起来是重建失败
    
    在good_sample中添加6007 6008
    
    
    在seg0中都有了，即都成功被预测
    
    train_val test是完全正确的，和多neuron不交
    
    应该ok了
    """
def get_unlabeled_list(unlabeled_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/unlabeled_list.csv"):
    if(not os.path.exists(unlabeled_recon_list_file)):
        final_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"
        final_id_list_df = pd.read_csv(final_recon_list_file)
        final_id_list = final_id_list_df["id"].tolist()
        final_id_list = [int(i) for i in final_id_list]

        train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train_val_list.csv"
        train_val_list = pd.read_csv(train_val_list_file)["id"].tolist()
        train_val_ids = [int(i) for i in train_val_list]

        test_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv"
        test_list = pd.read_csv(test_list_file)["id"].tolist()
        test_ids = [int(i) for i in test_list]

        # final_id_list - train_val - test
        unlabel_list = list(set(final_id_list).difference(set(train_val_ids)).difference(set(test_ids)))
        unlabel_list.sort()
        print(len(unlabel_list))

        unlabel_list_id = [str(i) for i in unlabel_list]
        unlabel_list_df = pd.DataFrame(unlabel_list_id, columns=["id"])
        unlabel_list_df.to_csv(unlabeled_recon_list_file, index=False)


    else:
        print("unlabeled_recon_list_file exists")
        final_id_list_df = pd.read_csv(unlabeled_recon_list_file)
        final_id_list = final_id_list_df["id"].tolist()
    print(f"unlabeled_id_list: {len(final_id_list)}")

    return final_id_list

def get_new_neuron_info():
    def get_ids_from_csv(file_path=r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"): # 从最终重建获取id
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 假设 'id' 列包含样本 ID
        if 'id' in df.columns:
            ids = df['id'].tolist()
            ids = [int(i) for i in ids]
            # print(ids)
            print(f"{len(ids)} samples")
            # 是否有重复？没有重复
            # print((len(ids) == len(set(ids))))
            return ids

    def get_patient_and_tissue_info(ids, excel_file=r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"):
        # 读取 Excel 文件
        df = pd.read_csv(excel_file, encoding='gbk')
        filtered_df = df[df['Cell ID'].isin(ids)][['Cell ID', '病人编号', '组织块编号', '切片厚度(微米)']]
        filtered_df.columns = ['id', 'patient_id', 'tissue_id', 'slice_thickness']

        filtered_df = filtered_df.groupby('id').first().reset_index()
        print(f"{len(filtered_df)} samples")
        return filtered_df

    def get_additional_info_from_excel(patient_tissue_df, excel_file=r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/sample_info10302024.xlsx"):
        df = pd.read_excel(excel_file)
        results = []
        for _, row in patient_tissue_df.iterrows():
            id = row['id']
            patient_id = row['patient_id']
            tissue_id = row['tissue_id']
            slice_thickness = row['slice_thickness']

            # 从 df 中找到匹配的病人编号和组织编号的行
            matched_row = df[(df['patient_number'] == patient_id) & (df['tissue_id'] == tissue_id)]
            # print(patient_id, tissue_id)
            if(len(matched_row) == 0):
                if(patient_id == 'P008' and tissue_id == 'T02'):
                    gender, age, brain_region = 'Female', 47, 'PL.L'
                elif(patient_id == 'P020' and tissue_id == 'T02'):
                    gender, age, brain_region = 'Female', 39, 'OL.R'
                else:
                    print(f"{id} {patient_id} {tissue_id} not found")
                    continue
            else:
                gender = matched_row['gender'].values[0]
                age = int(matched_row['patient_age'].values[0])
                brain_region = matched_row['english_abbr_nj'].values[0]
                if(gender == '男'):
                    gender = 'Male'
                elif(gender == '女'):
                    gender = 'Female'

            # 将结果存储到结果列表中
            results.append([id, patient_id, tissue_id, gender, age, brain_region, slice_thickness])

        # 将结果转化为 DataFrame
        final_df = pd.DataFrame(results, columns=['id', 'patient_id', 'tissue_id', 'gender', 'age', 'brain_region', 'slice_thickness'])
        return final_df

    ids = get_ids_from_csv()
    patient_tissue_df = get_patient_and_tissue_info(ids)
    final_df = get_additional_info_from_excel(patient_tissue_df)

    # sort
    final_df = final_df.sort_values(by=['id'])
    final_df_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    if(os.path.exists(final_df_file)):
        os.remove(final_df_file)
    final_df.to_csv(final_df_file, index=False)
    # print(final_df)
    unique_slice_thickness = final_df['slice_thickness'].unique()
    print(unique_slice_thickness)
    print(len(unique_slice_thickness))
    # 统计各个切片厚度的数量和百分比
    slice_thickness_count = final_df['slice_thickness'].value_counts()
    percentage = slice_thickness_count / slice_thickness_count.sum()
    # slice_thickness_count = final_df['slice_thickness'].value_counts()
    print(slice_thickness_count, percentage)

    # sort by id
    final_df = final_df.sort_values(by=['id'])

    plt.figure(figsize=(10, 6))
    # show 切片厚度和编号的关系
    # sns.scatterplot(data=final_df, x='id', y='slice_thickness')
    # 折线图
    sns.lineplot(data=final_df, x='id', y='slice_thickness')
    plt.show()
    plt.close()

def get_total_length(final_df_file):
    final_df = pd.read_csv(final_df_file)

    l_measure_swc_file = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv"
    l_measure_df = pd.read_csv(l_measure_swc_file)

    final_df = final_df.merge(l_measure_df[['ID', 'Total Length']], left_on='id', right_on='ID', how='left')
    final_df = final_df.drop(columns=['ID'])

    final_df.to_csv(final_df_file, index=False)

def check_gender(final_df_file):
    final_df = pd.read_csv(final_df_file)
    print(final_df.shape)
    patient_id = final_df['patient_id'].tolist()
    patient_id = list(set(patient_id))
    print(len(patient_id))

    patient_info_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/sample_info10302024.xlsx"
    df = pd.read_excel(patient_info_file)
    male_patient, female_patient = [], []

    for i in patient_id:
        matched_row = df[df['patient_number'] == i]
        if(len(matched_row) == 0):
            print(f"{i} not found")

        if (len(matched_row) == 0):
            if (patient_id == 'P008'):
                gender = 'Female'
            elif (patient_id == 'P020'):
                gender = 'Female'
            else:
                print(f"{i} {patient_id} not found")
                continue
        else:
            gender = matched_row['gender'].values[0]

        if (gender == '男'):
            gender = 'Male'
        elif (gender == '女'):
            gender = 'Female'

        if (gender == 'Male'):
            male_patient.append(i)
        else:
            female_patient.append(i)

    print(len(male_patient))
    print(len(female_patient))




if __name__ == "__main__":

    # get_final_id_list()
    # get_unlabeled_list()
    # check_final_list()

    final_df_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    check_gender(final_df_file)
    exit()
    if(not os.path.exists(final_df_file)):
        get_new_neuron_info()

    get_total_length(final_df_file)





