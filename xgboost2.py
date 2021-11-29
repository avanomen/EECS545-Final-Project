import numpy as np

class Node:
    def __init__(self, x, samples, hessian, gradient, feature_sel=0.8, gamma=1, reg=1, min_leaf_samples=5, min_child_weight = 1, max_depth=5, depth=0):
        self.x = x
        self.samples = samples
        self.n_samples = len(samples)
        self.gamma = gamma
        self.reg = reg  # lambda
        self.hessian = hessian
        self.gradient = gradient
        self.min_leaf_samples = min_leaf_samples
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.depth = depth
        self.feature_sel = feature_sel

        self.score = np.NINF
        
        # randomly selected subsample of columns/features
        n_features = x.shape[1]
        self.sampled_features = np.random.choice(n_features, size=int(np.floor(self.feature_sel * n_features)), replace=False)

        self.leaf_val = self.get_leaf_val(self.gradient[samples], self.hessian[samples])

        # find best split over all subsampled features
        self.find_best_split()

    def get_leaf_val(self, gradient, hessian):
        return -np.sum(gradient)/(np.sum(hessian) + self.reg)


    def find_best_split(self):
        for feature in self.sampled_features:
            self.find_greedy_split(feature)

            # check is node is a leaf
            if self.depth == self.max_depth or self.score == np.NINF:
                pass

            else:
                # create 2 new nodes at best split already found
                x_tmp = self.x[self.samples, self.split_feature]
                lhs = np.nonzero(x_tmp <= self.split_val)[0]
                rhs = np.nonzero(x_tmp > self.split_val)[0]

                self.lhs = Node(self.x, self.samples[lhs], 
                                self.hessian, 
                                self.gradient, 
                                self.feature_sel, 
                                self.gamma, 
                                self.reg, 
                                self.max_depth, 
                                self.min_child_weight,
                                self.max_depth,
                                self.depth + 1)

                self.rhs = Node(self.x, self.samples[rhs], 
                                self.hessian, 
                                self.gradient, 
                                self.feature_sel, 
                                self.gamma, 
                                self.reg, 
                                self.max_depth, 
                                self.min_child_weight,
                                self.max_depth,
                                self.depth + 1)

    def find_greedy_split(self, feature_idx):
        x_tmp = self.x[self.samples, feature_idx]

        for sample_idx in range(len(self.samples)):
            lhs_ind = np.nonzero(x_tmp <= x_tmp[sample_idx])[0]
            rhs_ind = np.nonzero(x_tmp > x_tmp[sample_idx])[0]

            # check that not less than min samples per leaf
            if len(lhs_ind) < self.min_leaf_samples or len(rhs_ind) < self.min_leaf_samples:
                continue
            # check purity score
            if self.hessian[lhs_ind].sum() < self.min_child_weight or self.hessian[rhs_ind].sum() < self.min_child_weight:
                continue

            # find split score
            score = self.find_gain(lhs_ind, rhs_ind)

            if score > self.score:
                self.score = score
                self.split_feature = feature_idx
                self.split_val = x_tmp[sample_idx]

    def predict(self, x):
        n_samples = x.shape[0]
        pred = np.zeros(n_samples)
        for i in range(n_samples):
            pred[i] = self.predict_sample(x[i, :])
        return pred

    def predict_sample(self, sample):
        # if a leaf return the val
        if self.score == np.NINF or self.depth == self.max_depth:
            return self.leaf_val

        # otherwise predict on next nodes
        else:
            if sample[self.split_feature] <= self.split_val:
                next_node = self.lhs
            else:
                next_node = self.rhs

            return next_node.predict_sample(sample)

    def find_gain(self, lhs, rhs):
        gradient = self.gradient[self.samples]
        hessian  = self.hessian[self.samples]
        
        gradl = gradient[lhs].sum()
        hessl  = hessian[lhs].sum()
        
        gradr = gradient[rhs].sum()
        hessr  = hessian[rhs].sum()

        return 0.5 * (gradl**2/(hessl + self.reg) + gradr**2/(hessr + self.reg) - (gradl + gradr)**2/(hessr + hessl + self.reg)) - self.gamma
   

class XGBtree:
    def fit(self, x, hessian, gradient, feature_sel=0.8, gamma=1, reg=1, min_leaf_samples=5, min_child_weight=1, max_depth=5):
        self.dec_tree = Node(x, np.array(np.arange(x.shape[0])), hessian, gradient, feature_sel, gamma, reg, min_leaf_samples, min_child_weight, max_depth, depth=0)
        return self

    def predict(self, x):
        return self.dec_tree.predict(x)


class XGBclassifier:
    def __init__(self):
        self.dec_trees = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def grad(self, pred, true):
        preds = self.sigmoid(pred)
        return(preds - true)

    # second order gradient logLoss
    def hess(self, pred):
        preds = self.sigmoid(pred)
        return(pred * (1 - preds))

    def fit(self, x, y, boosting_rounds=5, lr = 0.3, feature_sel=0.8, gamma=1, reg=1, min_leaf_samples=5, min_child_weight=1, max_depth=5):
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.init_pred = np.full((self.n_samples, 1), 0.0).flatten()

        self.x = x
        self.y = y
        self.boosting_rounds = boosting_rounds
        self.lr = lr
        self.feature_sel = feature_sel
        self.gamma = gamma
        self.reg = reg
        self.min_leaf_samples = min_leaf_samples
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth

        for boosting_round in range(self.boosting_rounds):
            print('boosting round {}'.format(boosting_round))
            gradient = self.grad(self.init_pred, self.y)
            hessian = self.hess(self.init_pred)
            tree = XGBtree().fit(x, hessian, gradient, self.feature_sel, self.gamma, self.reg, self.min_leaf_samples, self.min_child_weight, self.max_depth)
            self.init_pred += self.lr * tree.predict(self.x)
            self.dec_trees.append(tree)

    def predict(self, x):
        out = np.zeros(x.shape[0])

        for dec_tree in self.dec_trees:
            out += self.lr * dec_tree.predict(x)

        pred_prob = self.sigmoid(np.full((x.shape[0], 1), np.mean(self.y)).flatten().astype('float64') + out)
        return np.where(pred_prob > np.mean(pred_prob), 1, 0)






