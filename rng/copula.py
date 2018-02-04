from .rng import *
from scipy.stats import norm, t
import seaborn as sns


class BaseCopula(BaseRNG):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, cov_matrix):
        self.cov = cov_matrix
        self.d = len(self.cov)
        self.std = np.sqrt(np.diag(self.cov))
        self.A = np.linalg.cholesky(self.cov).T

    def reset_cov(self, cov_matrix):
        assert len(cov_matrix) == self.d
        self.cov = cov_matrix
        self.std = np.sqrt(np.diag(self.cov))
        self.A = np.linalg.cholesky(self.cov).T

    def bivariate_plot(self, n_sample, d1=0, d2=1):
        X = self.draw(n_sample)
        g = sns.jointplot(X[:, d1], X[:, d2], kind="reg")
        return g


class GaussianCopula(BaseCopula):
    def __init__(self, cov_matrix, marginal_inv_cdfs=None):
        super(GaussianCopula, self).__init__(cov_matrix)
        self.inv_cdfs = marginal_inv_cdfs

    def draw(self, n=1):
        Z = np.random.normal(size=(n, self.d))
        Y = Z.dot(self.A)
        if not self.inv_cdfs:
            return Y
        U = norm.cdf(Y / self.std)
        return np.apply_along_axis(self.inv_cdfs, 1, U)


class StudentTCopula(GaussianCopula):
    def __init__(self, df, cov_matrix, marginal_inv_cdfs=None):
        super(StudentTCopula, self).__init__(
            cov_matrix, marginal_inv_cdfs)
        self.df = df

    def draw(self, n=1):
        Z = np.random.normal(size=(n, self.d))
        Y = Z.dot(self.A)
        S = np.random.chisquare(df=self.df, size=n)
        T = Y * np.sqrt(self.df / S).reshape(n, 1)
        if not self.inv_cdfs:
            return T
        U = t.cdf(T / self.std, df=self.df)
        return np.apply_along_axis(self.inv_cdfs, 1, U)