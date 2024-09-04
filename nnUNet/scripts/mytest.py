import pandas as pd
from collections import Counter

csv_path = r"C:\Users\12626\Documents\WXWork\1688855276369019\Cache\File\2024-07\data.csv"
df = pd.read_csv(csv_path)
brain_regions = df.iloc[:, 1].tolist()
# for i in range(len(brain_regions)):
#     print(brain_regions[i])

element_counts = df.iloc[:, 1].value_counts()
print(element_counts)

brain_region_map = {
    'FL': ['FL.R', 'FL.L', 'MFG', 'FP.R', 'MFG.R', 'SFG.R', 'M(I)FG.L', 'SFG.L', 'S(M)FG.R', 'FP.L', 'IFG.R', 'SFG', 'IFG'],
    'TL': ['TL.R', 'TL.L', 'MTG.R', 'S(M)TG.R', 'MTG.L', 'FL_TL.L', 'MTG', 'S(M)TG.L', 'TP.L', 'TP.R', 'STG.R'],
    'PL': ['PL.L_OL.L', 'PL.L', 'IPL-near-AG', 'PL', 'IPL.L'],
    'OL': ['OL.R', 'OL.L']
}

brain_region_map = {
    "superior frontal gyrus": ["SFG.R", "SFG.L"],
    "middle frontal gyrus": ["MFG.R", "MFG.L"],
    "inferior frontal gyrus": ["IFG.R", "IFG.L"],


    "superior temporal gyrus": ["STG.R", "STG.L"],
    "middle temporal gyrus": ["MTG.R", "MTG.L"],

    "parietal lobe": ["PL.L_OL.L", "PL.L", "IPL-near-AG", "PL", "IPL.L"],
    "inferior parietal lobe": ["IPL-near-AG", "IPL.L"],

    "temporal pole": ["TP.R", "TP.L"],

    "s(m)fg": ["S(M)FG.R", "S(M)FG.L"],

    "s(m,i)fg": ["S(M)FG.R", "M(I)FG.L"],

}

grouped_counts = {group: sum (element_counts[region] for region in regions) for group, regions in brain_region_map.items()}
print(grouped_counts)
print(f"other: {len(brain_regions) - sum(grouped_counts.values())}")
# other elements
other_list = []
for region in brain_regions:
    if region not in grouped_counts.keys():
        other_list.append(region)

print(Counter(other_list))