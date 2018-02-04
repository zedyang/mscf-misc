from .rng import *
from progressbar import ProgressBar


class MarkovAliasTable(object):
    def __init__(self, transition_matrix):
        self.n = len(transition_matrix)
        self.transition_matrix = transition_matrix
        table_dim = (self.n, self.n - 1)
        self.prob = np.zeros(table_dim)
        self.values = np.zeros(table_dim, dtype=np.int32)
        self.alias = np.zeros(table_dim, dtype=np.int32)
        self.__make_table()

    def __make_table(self):
        for i, row in enumerate(self.transition_matrix):
            pmf = sorted([list(t) for t in zip(range(self.n), (
                self.n - 1) * row)], key=lambda t: -t[1])
            for j in range(self.n - 1):
                (state, p), alias = pmf[-1], pmf[0][0]
                self.prob[i, j] = p
                self.values[i, j] = state
                self.alias[i, j] = alias
                pmf[0][1] -= 1 - p
                pmf.pop()
                pmf = sorted(pmf, key=lambda t: -t[1])

    def draw(self, init_state):
        u = np.random.uniform()
        v = (self.n - 1) * u
        i = int(np.ceil(v))
        w = i - v
        if w <= self.prob[init_state, i - 1]:
            return self.values[init_state, i - 1]
        else:
            return self.alias[init_state, i - 1]

    def draw_path(self, clock_model, T, init_state):
        clock, state = 0, init_state
        while clock <= T:
            dt = clock_model(state)
            clock += dt
            if clock >= T: break
            state = self.draw(state)
        return state

    def estimate_prob(self, clock_model, T, init_state, n_paths):
        est = np.zeros(self.n)
        for _ in range(n_paths):
            final = self.draw_path(clock_model, T, init_state)
            est[final] += 1
        est /= n_paths
        return est

    def estimate_trans_matrix(self, clock_model, T, n_paths):
        bar = ProgressBar()
        est = np.zeros((self.n, self.n))
        for i in bar(range(self.n)):
            est[i, :] = self.estimate_prob(
                clock_model, T, i, n_paths)
        return est


# for testing purpose
def exponential_holding_time(state):
    exp_rates = [.1154, .1043, .1172, .1711, .2530, .1929, .4318, .0001]
    return np.random.exponential(1/exp_rates[state])


def gamma_holding_time(state):
    gamma_rates = [.1154, .1043, .1172, .1711, .2530, .1929, .4318, .0001]
    return 0.5*np.random.gamma(2,1/gamma_rates[state])