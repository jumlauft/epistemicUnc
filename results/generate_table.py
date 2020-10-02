import pandas as pd

data_sets, models = [], []
data_sets.append("1D_centered")
data_sets.append("1D_split")
data_sets.append("2D_square")
data_sets.append("2D_gaussian")
data_sets.append("pmsm_temperature")
data_sets.append("sarcos")
all_data = dict()
for data_set in data_sets:
    all_data[data_set] = pd.read_csv(data_set + '.csv', delimiter=',', header=0, index_col=0)


performance_measures = list(all_data[data_sets[0]].columns)
model_names = list(all_data[data_sets[0]].index)
model_names = [n if n != 'Negsep' else 'EpiOut' for n in model_names]
for pm in performance_measures:
    newdfs = []
    for data_set in data_sets:
        newdfs.append(pd.DataFrame(index=model_names, columns = [data_set],
                                   data=all_data[data_set][pm].values))

    with open(pm+".tex", "w") as f:
        f.write(pd.concat(newdfs,axis=1).to_latex())