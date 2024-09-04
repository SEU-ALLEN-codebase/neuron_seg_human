import os.path

def eswc2swc(eswc_file, swc_file):
    if(os.path.exists(swc_file)):
        os.remove(swc_file)
    result_lines = []
    with open(eswc_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if(line.startswith("#")):
                result_lines.append(line)
            else:
                line = line.strip().split()
                line = line[:7]
                result_lines.append(" ".join(line) + "\n")

    with open(swc_file, 'w') as f:
        f.writelines(result_lines)

if __name__ == '__main__':
    eswc_dir = "/data/kfchen/New Folder/to_Kaifeng/eswc/oneChecked_annotation"
    swc_dir = "/data/kfchen/New Folder/to_Kaifeng/oneChecked_annotationn"

    if(not os.path.exists(swc_dir)):
        os.makedirs(swc_dir)

    eswc_files = os.listdir(eswc_dir)

    for eswc_file in eswc_files:
        if(eswc_file.endswith(".eswc")):
            swc_file = eswc_file.replace(".eswc", ".swc")
            eswc2swc(os.path.join(eswc_dir, eswc_file), os.path.join(swc_dir, swc_file))

