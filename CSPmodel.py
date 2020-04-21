import numpy as np
import scipy.linalg as la

from mne import cov as co

class CSP():
    def __init__(self, n_components=10, reg=None, log=None, cov_est="concat",
                transform_into='average_power', norm_trace=False,
                cov_method_params=None, rank=None):
        """Init of CSP."""
        #componentはint型じゃなきゃやーよ
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        self.rank = rank

        self.reg = reg

        # Init default cov_est
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        # Init default transform_into ここ消したよ
        #_check_option('transform_into', transform_lkinto,
        #            ['average_power', 'csp_space'])
        self.transform_into = transform_into

        # Init default log
        if transform_into == 'average_power':
            #average_powerを使うときにはlog==Trueでなければならない
            if log is not None and not isinstance(log, bool):
                raise ValueError('log must be a boolean if transform_into == '
                                '"average_power".')
        else:
            if log is not None:
                #csp_spaceを使うときはlogはNoneでなければならない
                raise ValueError('log must be a None if transform_into == '
                                '"csp_space".')
        self.log = log

        if not isinstance(norm_trace, bool):
            raise ValueError('norm_trace must be a bool.')
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params

    #Xとyのトライアル数が一緒じゃないとだめ
    #Xは三次元以上のデータでないとダメ
    def _check_Xy(self, X, y=None):
        """Aux. function to check input data."""
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError('X and y must have the same length.')
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        #numpy配列でないとエラー
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        self._check_Xy(X, y)
        #チャンネル数
        n_channels = X.shape[1]

        #クラス数
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        #クラス数＊チャンネル数＊チャンネル数の0行列
        covs = np.zeros((n_classes, n_channels, n_channels))
        sample_weights = list()
        #クラス数分
        for class_idx, this_class in enumerate(self._classes):
            if self.cov_est == "concat":  # concatenate epochs
                #class_は各クラスごとのデータに分割する
                class_ = np.transpose(X[y == this_class], [1, 0, 2])
                class_ = class_.reshape(n_channels, -1) #エポック数×時間
                #正則化共分散行列？
                cov = co._regularized_covariance(
                    class_, reg=self.reg, method_params=self.cov_method_params,
                    rank=self.rank)
                #各エポック数
                weight = sum(y == this_class)
            elif self.cov_est == "epoch":
                class_ = X[y == this_class]
                cov = np.zeros((n_channels, n_channels))
                for this_X in class_:
                    cov += _regularized_covariance(
                        this_X, reg=self.reg,
                        method_params=self.cov_method_params,
                        rank=self.rank)
                cov /= len(class_)
                weight = len(class_)
            #2*10*10
            covs[class_idx] = cov
            if self.norm_trace:
                # Append covariance matrix and weight. Prior to version 0.15,
                # trace normalization was applied, but was breaking results for
                # some usecases by changing the apparent ranking of patterns.
                # Trace normalization of the covariance matrix was removed
                # without signigificant effect on patterns or performances.
                # If the user interested in this feature, we suggest trace
                # normalization of the epochs prior to the CSP.
                covs[class_idx] /= np.trace(cov)

            sample_weights.append(weight)

        if n_classes == 2:
            #固有値と固有ベクトルを計算
            eigen_values, eigen_vectors = la.eigh(covs[0], covs.sum(0))
            # sort eigenvectors
            self.ev = np.abs(eigen_values - 0.5)
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = _ajd_pham(covs)

            # Here we apply an euclidean mean. See pyRiemann for other metrics
            mean_cov = np.average(covs, axis=0, weights=sample_weights)
            eigen_vectors = eigen_vectors.T

            # normalize
            for ii in range(eigen_vectors.shape[1]):
                tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov), eigen_vectors[:, ii])
                eigen_vectors[:, ii] /= np.sqrt(tmp)

            # class probability
            class_probas = [np.mean(y == _class) for _class in self._classes]

            # mutual information
            mutual_info = []
            for jj in range(eigen_vectors.shape[1]):
                aa, bb = 0, 0
                for (cov, prob) in zip(covs, class_probas):
                    tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov), eigen_vectors[:, jj])
                    aa += prob * np.log(np.sqrt(tmp))
                    bb += prob * (tmp ** 2 - 1)
                mi = - (aa + (3.0 / 16) * (bb ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]

        # sort eigenvectors
        #eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        #ムーアペンローズの疑似逆行列
        self.patterns_ = la.pinv2(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        #エポックごとに計算する
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        # compute features (mean band power)
        X = (X ** 2).mean(axis=2)


        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        #----------
        #X : array, shape (n_epochs, n_channels, n_times)
        #Returns
        #-------
        #X : ndarray (n_epochs, n_sources, n_times)
        
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP ' 'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        
        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        

        return X, self.ev