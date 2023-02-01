import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

# input_dir = "outputs/myModel/20ng_50_4396_1_500_gcn"
input_dir = "outputs/myModel/imdb_50_4396_1_300_gcn"
with np.load(input_dir + "/theta.train.npz") as train_theta:
    train = train_theta['theta']
with np.load(input_dir + "/theta.dev.npz") as dev_theta:
    dev = dev_theta['theta']
with np.load(input_dir + "/theta.test.npz") as test_theta:
    test = test_theta['theta']

train_labels = np.load(input_dir + "/train_labels.npy")
dev_labels = np.load(input_dir + "/dev_labels.npy")
test_labels = np.load(input_dir + "/test_labels.npy")

train_dev = np.concatenate((train, dev))
train_dev_labels = np.concatenate((train_labels, dev_labels))

# print(type(train), type(train_labels))
# print(train.shape, train_labels.shape)
# print(dev.shape, dev_labels.shape)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=8)

# scores = cross_val_score(clf, train_dev, train_dev_labels, cv=5)  # 交叉验证
# print(scores.mean(), scores.std())

clf.fit(train_dev, train_dev_labels)
print(clf.score(test, test_labels))


# score_lt = []
# # 每隔10步建立一个随机森林，获得不同n_estimators的得分
# for i in range(10, 201, 10):
#     clf = RandomForestClassifier(n_estimators=i, random_state=42)
#     score = cross_val_score(clf, train_dev, train_dev_labels, cv=5, n_jobs=8).mean()
#     score_lt.append(score)
# score_max = max(score_lt)
# print('最大得分：{}'.format(score_max), '子树数量为：{}'.format(score_lt.index(score_max) * 10 + 10))
#
# # 绘制学习曲线
# x = np.arange(10, 201, 10)
# plt.subplot(111)
# plt.plot(x, score_lt, 'r-')
# plt.show()