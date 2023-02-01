import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


input_dir = "outputs/myModel/20ng_50_4396_1_500_gcn"
# input_dir = "outputs/myModel/imdb_50_2022_2_200_gcn"
# input_dir = "outputs/clntm_bert/20ng_50_2048_1_500"
# input_dir = "outputs/clntm_bert/imdb_50_2022_2_200"
with np.load(input_dir + "/theta.train.npz") as train_theta:
    train = train_theta['theta']
with np.load(input_dir + "/theta.dev.npz") as dev_theta:
    dev = dev_theta['theta']
with np.load(input_dir + "/theta.test.npz") as test_theta:
    test = test_theta['theta']

train_labels = np.load(input_dir + "/train_labels.npy")
dev_labels = np.load(input_dir + "/dev_labels.npy")
test_labels = np.load(input_dir + "/test_labels.npy")

# train_dev = np.concatenate((train, dev))
# train_dev_labels = np.concatenate((train_labels, dev_labels))

all_data = np.concatenate((train, dev, test))
all_labels = np.concatenate((train_labels, dev_labels, test_labels))

test_size = 0.4 if "20ng" in input_dir else 0.25

train, test, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=test_size, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=8)

# scores = cross_val_score(clf, train_dev, train_dev_labels, cv=5)  # 交叉验证
# print(scores.mean(), scores.std())

clf.fit(train, train_labels)
print(clf.score(test, test_labels))

