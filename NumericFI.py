
'''
This module compares between the efficacy of
t-test, signed-rank test and rank-sum test,
in terms of Fisher Information wrt iid Normal data.
___________________________________________________

Background:

Given the results of two algorithms on n various inputs
(or more generally, given two data sets derived from two
distributions with paired samples), there are several statistical
tests which ask the question:
"Assuming the two algorithms (distributions) are identical,
how likely is it to have at least such extreme data?"
If that probability is very low (typically <1-5%),
then the assumption of identicality is rejected.

One such test is the classic T-test for the differences
between the paired samples. This test assumes the data to be
iid and Normally-distributed. Since it assumes that
the data is derived from parametric family of distributions
(N(mu,sigma^2)), it is classified as a "parametric test".

Rank-sum test manages to avoid the normality assumption by
replacing the data samples with their ranks (i.e. 1 for
the smallest, 2n for the largest), and calculating the mean
rank of one of the algorithms (with expectation (2n+1)/2).
However, such test ignores much of the information in the data,
since it takes into account only the order of the values
and not the magnitude of the differences between them.

Note that the rank-sum test does not assume paired data samples.
Wilcoxon Signed-rank test tries to take advantage of this
and apply a variant of the rank test on the differences
between each two paired samples. This way large differences
between the datasets get larger weights in the test, hence more
information is exploited (though still less info than in t-test).

In case of not-normal data, the t-test is wrong.
However, many practical datasets are close to normal
(depending on one's tolerance to rough approximations),
hence rank-based tests unnecessarily throw information away.
___________________________________________________

Goal: quantify the amount of information which is lost in such cases.
___________________________________________________

Methodology:

One possible approach is to analyze the significance-power
tradeoff in each of the tests in various setups.
This work takes another approach and calculates the
Fisher Information of the statistic of each of the tests.

Statistical inference is meant to study the underlying
distribution of some data, often by assuming family
of distributions and estimating its parameters, and in
some cases just asking whether a specific parameter
satisfies a specific condition (in our case - "is mu1=mu2?").
The key to the task is the fact that various values of the
parameters result in (statistically) different values
of the observed data.
In particular, the estimation of a parameter is easier as
the distribution of the data is more sensitive to that parameter.
Fisher Information (FI) formalizes this idea.

To measure the information lost in rank test for normal data,
this work calculates the FI of the statistic of each of the tests.
The FI of the T-statistic wrt the expectation is known to be
1/sigma^2 (per data sample).
After failing to calculate FI analytically for the rank-based
tests (due to lack of clean expression for the ranks statistics
in absence of the null-hypothesis), a numeric approach was taken.
The numeric calculation of FI is based on randomized normal data,
along with two different calculations:
1. Apply normal approximation to the distribution of
the (empirical) statistic, and conclude its FI accordingly.
2. Calculate FI explicitly from its definition wrt to the
empirical distribution of the statistic.
___________________________________________________

Results:

1. The code takes few seconds on a standard laptop to run
and analyze the 3 tests on apparently-sufficient 100K datasets
of 100 samples each.

2. Both methods of numeric calculation managed to restore
the theoretic FI of t-test quite accurately in such run.

3. As expected, Wilcoxon signed-rank test had smaller FI -
only ~50% of the FI of t-test. This means that twice the
data is needed to restore the same significance level,
which is a small cost to pay in many applications for getting
rid of the normality assumption.

4. Surprisingly, the rank-sum test had only slightly-smaller FI
compared to t-test, and in particular significantly larger than
the signed-rank test.
It is unknown yet whether this is the result of a bug, of the
chosen setup, or of total uselessness of the signed-rank test.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from prettytable import PrettyTable as prt
from time import time

# Numerically calculate Fisher Information according to:
# Iu(theta) = E((d(logP(u))/dtheta)^2)
# where theta = mu = 0
# for the statistics of the tests t-test, Wilcoxon signed-rank, and rank-sum.
def FI(n_exp=10000, n_samp=100, dm=0.1, sigma=1, seed=1, nbins=100, smooth=0.0,
       alternative_wilcox=False, id=1, do_plot=True):
    # generate data
    if seed is not None: np.random.seed(seed)
    x1 = generate_data(n_exp,n_samp,sigma=sigma)
    x2 = generate_data(n_exp,n_samp,sigma=sigma)
    x = x2 - x1
    # generate shifted data
    xm = x - dm
    xp = x + dm
    # calculate normal statistics
    t0 = np.mean(x,  axis=1)/(np.sqrt(2)*sigma/np.sqrt(n_samp))
    tm = np.mean(xm, axis=1)/(np.sqrt(2)*sigma/np.sqrt(n_samp))
    tp = np.mean(xp, axis=1)/(np.sqrt(2)*sigma/np.sqrt(n_samp))
    # calculate wilcox statistics
    if alternative_wilcox:
        ew = n_samp*(n_samp+1)
        sw = np.sqrt(n_samp*(n_samp+1)*(2*n_samp+1)/24)
        w0 = (wilcox_alt(x )-ew)/sw
        wm = (wilcox_alt(xm)-ew)/sw
        wp = (wilcox_alt(xp)-ew)/sw
    else:
        ew = 0
        sw = np.sqrt(n_samp*(n_samp+1)*(2*n_samp+1)/6)
        w0 = (wilcox(x )-ew)/sw
        wm = (wilcox(xm)-ew)/sw
        wp = (wilcox(xp)-ew)/sw
    # calculate rank sum statistics
    er = n_samp*(2*n_samp+1)/2
    sr = np.sqrt(n_samp**2 * (2*n_samp+1) / 12)
    r0 = (rank_sum(x1,x2   )-er)/sr
    rm = (rank_sum(x1,x2-dm)-er)/sr
    rp = (rank_sum(x1,x2+dm)-er)/sr
    # visualize distinctions
    if do_plot:
        plt.subplot(3,1,1)
        plot_density(tm)
        plot_density(tp, 'T-test - Statistic Histogram')
        plt.subplot(3,1,2)
        plot_density(wm)
        plot_density(wp, 'Signed-rank test - Statistic Histogram')
        plt.subplot(3,1,3)
        plot_density(rm)
        plot_density(rp, 'Rank-sum test - Statistic Histogram')
    # get pdfs
    # please don't judge this code, it's 1:12AM
    bins_t = get_bins(t0,nbins)
    bins_w = get_bins(w0,nbins)
    bins_r = get_bins(r0,nbins)
    pt0 = dist(t0, bins_t, smooth)
    ptm = dist(tm, bins_t, smooth)
    ptp = dist(tp, bins_t, smooth)
    pw0 = dist(w0, bins_w, smooth)
    pwm = dist(wm, bins_w, smooth)
    pwp = dist(wp, bins_w, smooth)
    pr0 = dist(r0, bins_r, smooth)
    prm = dist(rm, bins_r, smooth)
    prp = dist(rp, bins_r, smooth)

    # Estimate FI:
    # theoretical FI of t-statistic:
    FIT = n_samp * (1/(2*sigma**2)) # note: var(x2-x1)=2*sigma^2.
    # estimate FI by explicit derivation according to FI definition:
    FIN = derive(pt0,ptm,ptp,dm)
    FIW = derive(pw0,pwm,pwp,dm)
    FIR = derive(pr0,prm,prp,dm)
    # estimate FI from normal approximation of the statistic:
    # all statistics were normalized to u~N(0,var), so
    # they all have I(u)=1/var^2 wrt their own expectation m.
    # to find I(u) wrt the expectation m0 of the original data,
    # we just need to multiply by (dm/dm0)^2, where
    # dm is given by u(x+dm)-u(x-dm), and dm0 is given by 2dm.
    FINX = ((np.mean(tp)-np.mean(tm))/(2*dm))**2 / np.var(t0)
    FIWX = ((np.mean(wp)-np.mean(wm))/(2*dm))**2 / np.var(w0)
    FIRX = ((np.mean(rp)-np.mean(rm))/(2*dm))**2 / np.var(r0)

    # print as a nice table and return
    tb = prt()
    tb.field_names = ['Experiment', 'Statistic',
                      'FI (explicit derivation)', 'FI (normal approx)']
    tb.add_row([f'{id:d} (n={n_samp:d},s={sigma:.0f})',
                f'T (theory={FIT:.1f})', f'{FIN:.1f}', f'{FINX:.1f}'])
    tb.add_row([id, f'W', f'{FIW:.1f}', f'{FIWX:.1f}'])
    tb.add_row([id, f'R', f'{FIR:.1f}', f'{FIRX:.1f}'])
    print(tb)
    return(FIT,FIN,FINX,FIW,FIWX,FIR,FIRX)


def generate_data(n_exps=1000, n_samps=1000, mu=0, sigma=1):
    return np.random.normal(mu,sigma,(n_exps,n_samps))

def wilcox(x):
    return np.sum(np.sign(x) * np.argsort(x), axis=1)

def wilcox_alt(x):
    return np.sum((1+np.sign(x))/2 * np.argsort(x), axis=1)

def rank_sum(x1, x2):
    x = np.sort(np.concatenate((x1,x2),axis=1))
    r2 = [np.searchsorted(x[i],x2[i],sorter=np.argsort(x[i]))
          for i in range(x2.shape[0])]
    rs = np.sum(r2, 1) + x2.shape[1]
    return np.squeeze(rs)

def get_bins(x, n):
    b = np.percentile(x, 100 * np.array(range(n+1)) / n)
    b[0] -= 1
    b[-1] += 1
    fix = np.array([0] + [(d <= 0) * (-d+np.finfo(float).eps) for d in np.diff(b)])
    b += fix
    return b

def plot_density(x, tit=''):
    density = stats.gaussian_kde(x)
    h = np.histogram(x, bins=get_bins(x,25)[1:-1])[1]
    plt.plot(h, density(h))
    plt.title(tit)

def dist(x, bins, smooth=0.1):
    h = np.histogram(x, bins=bins)[0] + smooth
    h /= (np.sum(h) + smooth*len(h))
    return h

def derive(p0,pm,pp,dm):
    return np.sum(p0 * (((np.log((pp)) - np.log((pm))) / (2 * dm)) ** 2))


if __name__=='__main__':
    t0 = time()
    for i in range(3):
        FI(n_exp=10000, n_samp=100, sigma=i+1,
           seed=1, id=i+1, do_plot=(i==0))
    print(f'Running time: {time()-t0:.1f}s')
    plt.show()
