
# NUMERIC CALCULATION OF FISHER INFORMATION
# FOR QUANTIFICATION OF INFORMATION LOSS IN NON-PARAMETRIC TESTS
# IN TWO-POPULATIONS-COMPARISON
___________________________________________________

Abstract

	Non-parametric tests can avoid the data-normality assumption
	in the cost of losing some of the information of the data.
	
	NumericFI module quantifies the loss of information in
	non-parametric tests (signed-rank & rank-sum) compared to
	parametric tests (t-test) in the case of two paired datasets
	of normal iid data, using numeric calculation of Fisher Information
	of each test statistic.

	The default settings (just run NumericFI.py...) demonstrate
	the advantage of t-test, though the rank-sum test is
	surprisingly close and in particular more informative than
	the signed-rank test.

	This is surprising since the signed-rank test is dedicatedly
	designed for paired data, and widely considered superior to
	the rank-sum test for such data - see for example:
	https://stats.stackexchange.com/questions/91034/difference-between-the-wilcoxon-rank-sum-test-and-the-wilcoxon-signed-rank-test
	It is unknown yet whether this is the result of a bug, of the
	chosen setup, of problems with the FI measure, or of total
	uselessness of the signed-rank test.
	Update: here is a reference claiming that neither of the tests should be much inferior to the t-test.
	https://projecteuclid.org/download/pdf_1/euclid.lnms/1215089756

	Warning: the code of the project is mostly poorly-designed,
	just as the Tel-Avivean nights in which it was written.

___________________________________________________

Background

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

Goal

	Quantify the amount of information which is lost in such cases.
___________________________________________________

Methodology

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

Results

	1. The code takes few seconds on a standard laptop to run
	and analyze the 3 tests on apparently-sufficient 10K datasets
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