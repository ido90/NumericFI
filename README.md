
# Numeric Calculation of Fisher Information for Quantification of Information Loss in Non-Parametric Tests in Two-Populations-Comparison

## Abstract

Non-parametric tests can avoid the data-normality assumption
in the cost of losing some of the information of the data.

NumericFI module quantifies the loss of information in
non-parametric tests (signed-rank & rank-sum, both often named
_Wilcoxon test_) compared to parametric tests (t-test) in the case of
two paired datasets of normal iid data, using numeric calculation of
Fisher Information of each test statistic.

The calculations show that while the t-test does have some advantage,
it is quite minor (less than 10% in terms of Fisher Information,
which roughly means it can be compensated by using 10% more data).
In addition, the signed-rank test looks slightly better than the rank-sum
test.
These results are consistent with literature that studied such tests
efficiency using different approaches (see examples in the reference below).

For a demonstration just run NumericFI.py.

## Background

Given the results of two algorithms on n various inputs
(or more generally, given two data sets derived from two
distributions with paired samples), there are several statistical
tests which ask the question:
"Assuming the two algorithms (distributions) are identical,
how likely is it to have at least such extreme data?".
If that probability is very low (typically <1-5%),
then the assumption of identicality is rejected.

One such test is the classic T-test for the differences
between the paired samples. This test assumes the data to be
iid and Normally-distributed. Since it assumes that
the data is derived from parametric family of distributions
(N(mu,sigma^2)), it is classified as a "parametric test".

Rank-sum test manages to avoid the normality assumption by
replacing the data samples with their ranks (i.e. 1 for
the smallest, 2*n for the largest), and calculating the mean
rank of one of the algorithms (with expectation (2n+1)/2).
However, such test ignores some of the information in the data,
as it takes into account only the order of the values
and not the magnitude of the differences between them.

Note that the rank-sum test does not assume paired data samples.
Wilcoxon Signed-rank test tries to take advantage of this
and apply a variant of the rank test on the differences
between each two paired samples. This way large differences
between the datasets get larger weights in the test, hence more
information is exploited (though still less info than in t-test).

In case of not-normal data, the t-test is "wrong".
However, many practical datasets are close to normal
(depending on one's tolerance to rough approximations),
hence rank-based tests unnecessarily throw information away.

### Goal

Quantify the amount of information which is lost in such cases.

## Methodology

One possible approach is to analyze the significance-power
tradeoff in each of the tests in various setups.
This work takes another approach and calculates the
Fisher Information of the statistic of each of the tests.

	Fisher Information:
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
	Note: one important property of FI is being linear with the
	number of (independent) data samples.

To measure the information lost in rank test for normal data,
this work calculates the FI of the statistic of each of the tests.
The FI of the T-statistic with relation to the expectation (assuming
known variance) is known to be 1/sigma^2 (per data sample).
After failing to calculate FI analytically for the rank-based
tests (due to lack of clean expression for the ranks statistics
in absence of the null-hypothesis), a numeric approach was taken.

The numeric calculation of FI is based on randomized normal paired data,
along with two different calculations:
1. Apply normal approximation to the empirical distribution of
the statistic, and conclude its FI accordingly.
2. Calculate FI explicitly from its definition with relation to the
empirical distribution of the statistic.

## Results

1. The code takes few seconds on a standard laptop to run
and analyze the 3 tests on (apparently-sufficient) 10K datasets
of 100 samples each.

2. Both methods of numeric calculation managed to restore
the theoretic FI of t-test quite accurately.

3. As expected (for normal paired data), t-test looks better
than signed-rank test which looks better than rank-sum test,
for several different configurations of the data.

4. However, the differences between the information in the test statistics
were quite small - few percents only, which is equivalent to change
of few percents in the amount of the data.
It turns out that this is consistent with the known literature, which
studied similar questions in terms of Relative Asymptotic Efficiency
rather than FI. See for example:
https://projecteuclid.org/download/pdf_1/euclid.lnms/1215089756

In summary, it seems that using non-parametric tests by default may be
a good practice, which can return the statistical interpretation
to the p-values that are calculated every day around the world.
