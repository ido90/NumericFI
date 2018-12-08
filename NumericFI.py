
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from prettytable import PrettyTable as prt
from time import time

# Numerically calculate Fisher Information for test statistics of
# t-test, Wilcoxon signed-rank test, and rank-sum test,
# based on normal iid paired data samples.
# FI of a statistic u with relation to parameter mu:
# Iu(mu) = E((d(logP(u))/dmu)^2)  (computed at mu=0 in our case)
def compare_tests_FI(n_exp=10000, n_samp=100, sigma=1,
       seed=1, dm=0.1, nbins=100, smooth=0.0, # numerical parameters
       t_method='calculate', w_method='zero_mean', r_method='calculate',
       do_plot=True, id=1):

    # generate data
    if seed is not None: np.random.seed(seed)
    x1, x2 = generate_data(n_exp,n_samp,n_sets=2,sigma=sigma)

    # calculate statistics
    tm, t0, tp = t_test(x1, x2, x2_shifts=(-dm,0,dm),
                        normalization = np.sqrt(2*sigma**2/n_samp),
                        method = t_method)
    wm, w0, wp = wilcox(x1, x2, x2_shifts=(-dm,0,dm), normalize=True, method=w_method)
    rm, r0, rp = rank_sum(x1, x2, x2_shifts=(-dm,0,dm), normalize=True, method=r_method)

    # visualize distinctions
    if do_plot:
        plt.subplot(3,1,1)
        plot_densities((tm,tp), tit='T-test - Statistic Histogram')
        plt.subplot(3,1,2)
        plot_densities((wm,wp), tit='Signed-rank test - Statistic Histogram')
        plt.subplot(3,1,3)
        plot_densities((rm,rp), tit='Rank-sum test - Statistic Histogram')

    # Fisher Information - theoretical value for t-statistic:
    FI = {'theoretical': n_samp * (1/(2*sigma**2))} # note: var(x2-x1)=2*sigma^2.

    # Fisher Information - method 1:
    # All 3 statistics are roughly u~N(0,var), so
    # they all have I(u)=1/var^2 wrt their own expectation m.
    # to find I(u) wrt the expectation m0 of the underlying data,
    # we just need to multiply by (dm/dm0)^2, where
    # dm is given by u(x+dm)-u(x-dm), and dm0 is given by 2*dm.
    FI['approx'] = {'t': ((np.mean(tp)-np.mean(tm))/(2*dm))**2 / np.var(t0),
                    'wilcox': ((np.mean(wp)-np.mean(wm))/(2*dm))**2 / np.var(w0),
                    'rank-sum': ((np.mean(rp)-np.mean(rm))/(2*dm))**2 / np.var(r0) }

    # Fisher Information - method 2:
    # Derive FI explicitly from definition, using the empirical pdf of each statistic.
    pt0, ptm, ptp = dist((t0,tm,tp), nbins, smooth)
    pw0, pwm, pwp = dist((t0,tm,tp), nbins, smooth)
    pr0, prm, prp = dist((r0,rm,rp), nbins, smooth)
    FI['explicit'] = {'t': derive_FI(pt0,ptm,ptp,dm),
                      'wilcox': derive_FI(pw0,pwm,pwp,dm),
                      'rank-sum': derive_FI(pr0,prm,prp,dm) }

    print(print_exp_results(FI, id=id, n=n_samp, sigma=sigma))
    return FI


def generate_data(n_exps=1000, n_samps=1000, n_sets=1, mu=0, sigma=1):
    return tuple(np.random.normal(mu,sigma,(n_exps,n_samps)) for _ in range(n_sets))

def t_test(x1, x2, x2_shifts=(0,), normalization=1, method='calculate'):
    if method == 'calculate':
        return tuple(np.mean((x2+shift)-x1, axis=1) / normalization
                     for shift in x2_shifts)
    elif method == 'builtin':
        return tuple(stats.ttest_ind(x2+shift, x1, axis=1)[0] / normalization
                     for shift in x2_shifts)
    else:
        raise ValueError('Invalid method', method)

def wilcox(x1, x2, x2_shifts=(0,), normalize=True, method='zero_mean'):
    w = []
    if method in ('classic','zero_mean'):
        for shift in x2_shifts:
            x = (x2+shift) - x1
            ids = np.argsort(np.abs(x))
            x = np.array([xx[i] for (xx, i) in zip(x, ids)])
            if method == 'classic':
                w.append(np.sum((x>0)      * (1+np.arange(len(x[0]))), axis=1))
            else:
                w.append(np.sum(np.sign(x) * (1+np.arange(len(x[0]))), axis=1))
    elif method == 'builtin':
        raise NotImplementedError("Builtin Wilcoxon statistic does not "
                                  "distinguish between positive and negative.")
        #for shift in x2_shifts:
        #    w.append([stats.wilcoxon(xx1, xx2)[0] for (xx1, xx2) in zip(x1, x2+shift)])
    else:
        raise ValueError('Invalid method', method)

    if normalize:
        n = x1.shape[1]
        e = 0 if method=='zero_mean' else n*(n+1)/4
        s = np.sqrt(n*(n+1)*(2*n+1)/(6 if method=='zero_mean' else 24))
        w = ((ww-e)/s for ww in w)

    return tuple(w)

def rank_sum(x1, x2, x2_shifts=(0,), normalize=True, method='calculate'):
    w = []
    if method == 'calculate':
        for shift in x2_shifts:
            x2s = x2 + shift
            x = np.sort(np.concatenate((x1, x2s), axis=1))
            r = [np.searchsorted(x[i], x2s[i], sorter=np.argsort(x[i]))
                 for i in range(x2s.shape[0])]
            w.append(np.squeeze(np.sum(r,1)))
    elif method == 'builtin':
        for shift in x2_shifts:
            r = np.array([stats.ranksums(xx2, xx1)[0]
                          for (xx1, xx2) in zip(x1, x2+shift)])
            w.append(r)
    else:
        raise ValueError('Invalid method', method)

    if normalize:
        n = x1.shape[1]
        e = 0 if method=='builtin' else n*(2*n+1)/2
        s = 1 if method=='builtin' else np.sqrt(n**2 * (2*n+1) / 12)
        w = ((ww-e)/s for ww in w)

    return tuple(w)

def plot_densities(x, nbins=25, tit=''):
    for xx in x:
        density = stats.gaussian_kde(xx)
        h = np.histogram(xx, bins=get_bins(xx, nbins)[1:-1])[1]
        plt.plot(h, density(h))
    plt.title(tit)

def dist(x, nbins=100, smooth=0.0):
    bins = get_bins(x[0], nbins)
    d = []
    for xx in x:
        h = np.histogram(xx, bins=bins)[0] + smooth
        d.append( h / (np.sum(h) + smooth*len(h)) )
    return tuple(d)

def get_bins(x, n):
    b = np.percentile(x, 100 * np.array(range(n+1)) / n)
    b[0] -= 1
    b[-1] += 1
    fix = np.array([0] + [(d <= 0) * (-d+np.finfo(float).eps) for d in np.diff(b)])
    b += fix
    return b

def derive_FI(p0,pm,pp,dm):
    return np.sum(p0 * (((np.log((pp)) - np.log((pm))) / (2 * dm)) ** 2))

def print_exp_results(FI, id=None, n=None, sigma=None):
    tb = prt()
    tb.field_names = ['Experiment', 'Statistic',
                      'FI (explicit)', 'FI (approx)']
    tb.add_row([f'{id:d} (n={n:d},s={sigma:.0f})',
                f'T (theory={FI["theoretical"]:.1f})',
                f'{FI["explicit"]["t"]:.1f}',
                f'{FI["approx"]["t"]:.1f}'])
    tb.add_row([id, f'Wilcoxon', f'{FI["explicit"]["wilcox"]:.1f}',
                f'{FI["approx"]["wilcox"]:.1f}'])
    tb.add_row([id, f'Rank-sum', f'{FI["explicit"]["rank-sum"]:.1f}',
                f'{FI["approx"]["rank-sum"]:.1f}'])
    return tb


if __name__=='__main__':
    t0 = time()
    for i in range(3):
        compare_tests_FI(n_exp=10000, n_samp=100, sigma=i+1,
                         seed=1, dm=0.1, nbins=100, smooth=0.0,
                         t_method='calculate', w_method='zero_mean', r_method='calculate',
                         do_plot=(i==0), id=i+1)
    print(f'Running time: {time()-t0:.1f}s')
    plt.show()
