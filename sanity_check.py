import code
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

data_frame = pd.read_csv("mnist/fashion_precomp_encoderBIV_test.csv")


def sanity_check_encoding(data_frame,key_str):
    labels = data_frame['label']
    encodings = data_frame.drop(columns=['label'])

    pca = PCA(n_components=2)

    pri_comp = pca.fit_transform(encodings)
    pri_comp_labels = ['comp_1', 'comp_2']
    pri_comp_df = pd.DataFrame(data=pri_comp, columns=pri_comp_labels)

    pri_comp_df.plot(kind="scatter", x='comp_1', y="comp_2")
    plt.title("plotting PCA shape={}".format(pri_comp_df.shape))
    plt.savefig(f"./visualizations/{key_str}_PCA_plot.png")
