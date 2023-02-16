import pandas as pd

base_dir = '_pipeline_runs/1676387276'
clusters = pd.read_excel(f'./{base_dir}/_general_results.xlsx')
clusters['Paper_counts'].describe(include='all')
# count    698.000000
# mean      26.891117
# std       31.496526
# min        1.000000
# 25%        8.000000
# 50%       13.000000
# 75%       27.000000
# max      101.000000

#
clusters.hist(column='Paper_counts')
clusters.plot(kind='hist')
clusters.plot.hist()

# convert data
clusters['Arxive_tag'] = pd.Categorical(clusters['Arxive_tag'])
#rings_category = pd.cut(clusters['Paper_counts'], bins=[0,8,17,65,120], labels=[3, 2, 1, 0])
rings_category = pd.cut(clusters['Paper_counts'], bins=[0,12,27,90,120], labels=[0, 1, 2, 3])
rings_category = pd.cut(clusters['Paper_counts'], bins=[0,6,12,25,120], labels=[3, 2, 1, 0])
#clusters = clusters.drop('Radar_ring', axis=1)
clusters.insert(5, 'Radar_ring', rings_category)

# save new data frame
f_clusters = pd.DataFrame(columns = ['quadrant', 'ring', 'label', 'active', 'moved'])
f_clusters['quadrant'] = clusters['Arxive_tag'].cat.codes
f_clusters['ring'] = clusters['Radar_ring']
f_clusters['label'] = clusters['Topic_label'].str.capitalize()
f_clusters['active'] = True
f_clusters['moved'] = 0
f_clusters = f_clusters.drop_duplicates(subset='label', keep="first")
f_clusters.to_json(f'./{base_dir}/_pipeline_output_records.json', orient="records")

# EXAMPLE
# entries: [
#       {
#         "quadrant": 3,
#         "ring": 2,
#         "label": "AWS Athena",
#         "active": true,
#         "moved": 0
#       },
#       {
#         "quadrant": 3,
#         "ring": 1,
#         "label": "Flink",
#         "link": "https://engineering.zalando.com/tags/apache-flink.html",
#         "active": true,
#         "moved": 0
#       }
#     ]