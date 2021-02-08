import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs
from race import RACE
from lsh import ORHash
from scipy.stats import entropy
from skmultiflow.data import DataStream


np.random.seed(114514)
sgd_lr = 0.005
kd_beta = 1.0


def sync_data(dim, num, groups):
    y_tr = np.empty(shape=[0])
    y_val = np.empty(shape=[0])
    x_tr = np.empty(shape=[0, dim])
    x_val = np.empty(shape=[0, dim])
    for g in range(groups):
        class_num = int(num/2)
        mean_pos = np.zeros(shape=[dim])
        mean_pos[0] = g + 1
        mean_neg = np.zeros(shape=[dim])
        mean_neg[0] = - g - 1
        cov = np.identity(dim)
        x_pos = np.random.multivariate_normal(mean_pos, cov, size=class_num+100)
        x_neg = np.random.multivariate_normal(mean_neg, cov, size=class_num+100)
        x = np.vstack([x_pos, x_neg])
        y = np.concatenate([np.ones(class_num+100, int), np.zeros(class_num+100, int)])
        perm = np.random.permutation(np.arange(y.shape[0], dtype=int))
        x = x[perm, :]
        y = y[perm]
        # x, y = make_blobs(n_samples=num+200, centers=5, n_features=dim, center_box=(3*g, 3*g+1), random_state=g)
        x_tr = np.vstack([x_tr, x[:-200, :]])
        y_tr = np.concatenate([y_tr, y[:-200]])
        x_val = np.vstack([x_val, x[-200:, :]])
        y_val = np.concatenate([y_val, y[-200:]])
    return x_tr, y_tr, x_val, y_val


class Client:
    def __init__(self, data_dim, sketch_rep, hash_func, eps=0.0, name="default client"):
        self.name = name
        self.data_dim = data_dim
        self.sketch_rep = sketch_rep
        self.classifier = SGDClassifier('log', learning_rate='constant', eta0=sgd_lr)
        # self.classifier = SGDClassifier('log')
        self.hash_func = hash_func
        self.local_sketch = RACE(sketch_rep, 2)
        self.global_sketch = RACE(sketch_rep, 2)
        self.data_stream = None
        self.class_list = []
        self.labeled_count = 0
        self.stream_count = 0
        self.clf_param = None
        self.grad = None
        self.eps = eps
        self.labeled_x = np.empty(shape=[0, data_dim])
        self.labeled_y = np.empty(shape=[0])

    def set_stream(self, stream, class_list):
        self.data_stream = stream
        self.class_list = class_list

    def local_train_iter(self, batch_size, select_size):
        if self.data_stream.n_remaining_samples() < batch_size:
            self.data_stream.restart()
        x_batch, y_batch = self.data_stream.next_sample(batch_size)
        h = self.hash_func.batch_hash(x_batch)
        self.local_sketch.batch_add(h)
        self.global_sketch.batch_add(h)
        kd = self.global_sketch.batch_query(h)
        if self.labeled_count == 0:
            ent = np.ones(batch_size)
        else:
            pred_prob = self.classifier.predict_proba(x_batch)
            ent = entropy(pred_prob, axis=1)
        select_scores = ent * (kd ** kd_beta)
        select_index = np.argsort(select_scores)[-select_size:]
        if self.clf_param is None:
            self.classifier.partial_fit(x_batch[select_index, :], y_batch[select_index], classes=self.class_list)
            self.classifier.coef_ = np.zeros_like(self.classifier.coef_)
            self.classifier.intercept_ = np.zeros_like(self.classifier.intercept_)
        self.clf_param = self.get_param()
        self.classifier.partial_fit(x_batch[select_index, :], y_batch[select_index])
        self.grad = self.get_param() - self.clf_param
        self.stream_count += batch_size
        self.labeled_count += select_size

    def local_train_iter_pool(self, batch_size, select_size):
        if self.data_stream.n_remaining_samples() < batch_size:
            self.data_stream.restart()
        x_batch, y_batch = self.data_stream.next_sample(batch_size)
        h = self.hash_func.batch_hash(x_batch)
        self.local_sketch.batch_add(h)
        self.global_sketch.batch_add(h)
        kd = self.global_sketch.batch_query(h)
        if self.labeled_count == 0:
            ent = np.ones(batch_size)
        else:
            pred_prob = self.classifier.predict_proba(x_batch)
            ent = entropy(pred_prob, axis=1)
        select_scores = ent * (kd ** kd_beta)
        select_index = np.argsort(select_scores)[-select_size:]
        self.labeled_x = np.vstack([self.labeled_x, x_batch[select_index, :]])
        self.labeled_y = np.concatenate([self.labeled_y, y_batch[select_index]])
        if self.clf_param is None:
            self.classifier.partial_fit(self.labeled_x, self.labeled_y, classes=self.class_list)
            self.classifier.coef_ = np.zeros_like(self.classifier.coef_)
            self.classifier.intercept_ = np.zeros_like(self.classifier.intercept_)
        self.clf_param = self.get_param()
        self.classifier.partial_fit(x_batch[select_index, :], y_batch[select_index])
        self.grad = self.get_param() - self.clf_param
        self.stream_count += batch_size
        self.labeled_count += select_size

    def local_test(self, x_test, y_test):
        acc = self.classifier.score(x_test, y_test)
        print(self.name, self.labeled_count, acc)
        return acc

    def local_train(self, batch_size, select_size, budget, x_test=None, y_test=None):
        while self.labeled_count < budget:
            self.local_train_iter(batch_size, select_size)
            if x_test is not None:
                acc = self.local_test(x_test, y_test)
                print(self.name, self.labeled_count, acc)

    def get_param(self):
        coef = self.classifier.coef_
        intercept = self.classifier.intercept_
        return np.hstack([coef, intercept[:, np.newaxis]])

    def set_param(self, param_mat):
        coef = param_mat[:, :-1]
        intercept = param_mat[:, -1]
        self.classifier.coef_ = coef
        self.classifier.intercept_ = intercept

    def get_grad(self):
        return self.grad

    def get_sketch(self):
        return self.local_sketch.counts_lap(self.eps)

    def set_sketch(self, counts):
        self.global_sketch.counts = counts


class CenterNode:
    def __init__(self):
        self.total_count = None
        self.classifier = SGDClassifier('log', learning_rate='constant', eta0=sgd_lr)
        self.clf_param = None
        self.grad = None

    def init_param(self, data_dim, class_list):
        x = np.ones([1, data_dim])
        y = np.array([class_list[0]])
        self.classifier.partial_fit(x, y, classes=class_list)
        self.classifier.coef_ = np.zeros_like(self.classifier.coef_)
        self.classifier.intercept_ = np.zeros_like(self.classifier.intercept_)

    def sum_sketches(self, sketch_list):
        self.total_count = np.sum(sketch_list, axis=0)

    def get_sketch(self):
        return self.total_count

    def update_py_param(self, param_list):
        self.clf_param = self.get_param()
        mean_param = np.sum(param_list, axis=0)/param_list.shape[0]
        coef = mean_param[:, :-1]
        intercept = mean_param[:, -1]
        self.classifier.coef_ = coef
        self.classifier.intercept_ = intercept
        self.grad = self.get_param()-self.clf_param

    def get_param(self):
        coef = self.classifier.coef_
        intercept = self.classifier.intercept_
        return np.hstack([coef, intercept[:, np.newaxis]])

    def update_by_grad(self, grad_list):
        self.clf_param = self.get_param()
        mean_grad = np.sum(grad_list, axis=0)/grad_list.shape[0]
        self.classifier.coef_ += mean_grad[:, :-1]
        self.classifier.intercept_ = mean_grad[:, -1]
        self.grad = mean_grad

    def get_grad(self):
        return self.grad

    def test(self, x_test, y_test):
        acc = self.classifier.score(x_test, y_test)
        print("CENTER", acc)
        return acc


class DistributeModel:
    def __init__(self, data_dim, sketch_rep, client_num, eps=0.0):
        self.center = CenterNode()
        self.clients = []
        self.client_num = client_num
        self.data_dim = data_dim
        self.sketch_rep = sketch_rep
        self.hash_func = ORHash(sketch_rep, data_dim, 1)
        for i in range(client_num):
            self.clients.append(Client(data_dim, sketch_rep, self.hash_func, eps=eps, name="CLIENT"+str(i+1)))
        self.class_list = None
        self.x_test = None
        self.y_test = None

    def set_data(self, x_train, y_train, x_test, y_test):
        self.class_list = np.array(list(set(np.concatenate([y_train, y_test]))))
        self.center.init_param(self.data_dim, self.class_list)
        train_num = x_train.shape[0]
        stream_size = int(np.floor(train_num/self.client_num))
        for i in range(self.client_num):
            stream = DataStream(x_train[i*stream_size:(i+1)*stream_size, :], y_train[i*stream_size:(i+1)*stream_size])
            self.clients[i].set_stream(stream, self.class_list)
        self.x_test = x_test
        self.y_test = y_test

    def train_param(self, batch_size, select_size, budget):
        acc_list = []
        while self.clients[0].labeled_count < budget:
            param_list = []
            sketch_list = []
            for client in self.clients:
                client.local_train_iter_pool(batch_size, select_size)
                param_list.append(client.get_param())
                sketch_list.append(client.get_sketch())
                client.local_test(self.x_test, self.y_test)
            self.center.update_py_param(np.array(param_list))
            self.center.sum_sketches(np.array(sketch_list))
            for client in self.clients:
                client.set_param(self.center.get_param())
                client.set_sketch(self.center.total_count)
            center_acc = self.center.test(self.x_test, self.y_test)
            acc_list.append(center_acc)
        return acc_list

    def train_grad(self, batch_size, select_size, budget):
        while self.clients[0].labeled_count < budget:
            grad_list = []
            sketch_list = []
            for client in self.clients:
                client.local_train_iter(batch_size, select_size)
                grad_list.append(client.get_grad())
                sketch_list.append(client.get_sketch())
                client.local_test(self.x_test, self.y_test)
            self.center.update_by_grad(np.array(grad_list))
            self.center.sum_sketches(np.array(sketch_list))
            for client in self.clients:
                client.set_param(self.center.get_param())
                client.set_sketch(self.center.total_count)
            self.center.test(self.x_test, self.y_test)


if __name__ == '__main__':
    np.random.seed(114514)
    rep = 200
    p = 1
    # data_set_list = ['SensIT Vehicle', 'shuttle', 'satimage', 'usps', "sync"]
    data_set_list = ["sync"]
    for data_set_name in data_set_list:
        print("=================")
        if data_set_name is not "sync":
            data_file = np.load(data_set_name + '.npz')
            x = data_file['X_train']
            y = data_file['y_train']
            x_test = data_file['X_test']
            y_test = data_file['y_test']
        else:
            x, y, x_test, y_test = sync_data(1000, 3000, 5)
        data_dim = x.shape[-1]
        base_model = SGDClassifier(loss="log")
        rnd_idx = np.random.choice(x.shape[0], 300, False)
        base_model.fit(x[rnd_idx, :], y[rnd_idx])
        base_acc = base_model.score(x_test, y_test)
        print("base:", base_acc)
        model = DistributeModel(data_dim, rep, 5, eps=0)
        model.set_data(x, y, x_test, y_test)
        acc_list = model.train_param(50, 2, 60)
        print(data_set_name, max(acc_list))
        print("=================")
