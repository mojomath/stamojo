# ===----------------------------------------------------------------------=== #
# StaMojo — Usage examples (Part I)
# ===----------------------------------------------------------------------=== #
"""Run with:  mojo run -I src examples/examples.mojo"""

from stamojo.special import gammainc, gammaincc, beta, lbeta, betainc, erfinv, ndtri
from stamojo.distributions import Normal, StudentT, ChiSquared, FDist
from stamojo.stats import (
    mean, variance, std, median, quantile, skewness, kurtosis,
    pearsonr, spearmanr, kendalltau,
    ttest_1samp, ttest_ind, ttest_rel,
    chi2_gof, chi2_ind, ks_1samp, f_oneway,
)


fn main() raises:
    # --- Special functions ---------------------------------------------------
    print("gammainc(1, 2) =", gammainc(1.0, 2.0))           # 0.8646647167628346
    print("gammaincc(1, 2) =", gammaincc(1.0, 2.0))         # 0.13533528323716537
    print("beta(2, 3)     =", beta(2.0, 3.0))               # 0.08333333333323925
    print("betainc(2, 3, 0.5) =", betainc(2.0, 3.0, 0.5))   # 0.6875000000000885
    print("erfinv(0.5)    =", erfinv(0.5))                  # 0.4769362762044701
    print("ndtri(0.975)   =", ndtri(0.975))                 # 1.9599639845400543
    print()

    # --- Distributions -------------------------------------------------------
    var n = Normal(0.0, 1.0)
    print("Normal(0,1).pdf(0)   =", n.pdf(0.0))             # 0.3989422804014327
    print("Normal(0,1).cdf(1.96)=", n.cdf(1.96))            # 0.9750021048517795
    print("Normal(0,1).ppf(0.975)=", n.ppf(0.975))          # 1.9599639845400543
    print("Normal(0,1).sf(1.96) =", n.sf(1.96))             # 0.02499789514822043

    var t = StudentT(10.0)
    print("StudentT(10).cdf(2.0)=", t.cdf(2.0))             # 0.9633059826444078
    print("StudentT(10).ppf(0.975)=", t.ppf(0.975))         # 2.2281388540534057

    var c = ChiSquared(5.0)
    print("ChiSquared(5).cdf(11.07)=", c.cdf(11.07))        # 0.9499903814759155

    var f = FDist(5.0, 10.0)
    print("FDist(5,10).cdf(3.33)=", f.cdf(3.33))            # 0.9501687242532277
    print()

    # --- Descriptive statistics ----------------------------------------------
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print("mean    =", mean(data))              # 5.0
    print("variance=", variance(data, ddof=0))  # 4.0
    print("std     =", std(data, ddof=0))       # 2.0
    print("median  =", median(data))            # 4.5
    print("Q(0.25) =", quantile(data, 0.25))    # 4.0
    print("skewness=", skewness(data))          # 0.8184875533567997
    print("kurtosis=", kurtosis(data))          # 0.940625
    print()

    # --- Correlation ---------------------------------------------------------
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var y: List[Float64] = [2.1, 3.8, 6.0, 7.9, 10.1]
    var pr = pearsonr(x, y)
    print("pearsonr  r=", pr[0], " p=", pr[1])    # 0.9991718425080479, 2.8605484175113625e-05
    var sr = spearmanr(x, y)
    print("spearmanr ρ=", sr[0], " p=", sr[1])    # 1.0, 0.0
    var kt = kendalltau(x, y)
    print("kendalltau τ=", kt[0], " p=", kt[1])   # 1.0, 0.014305878435429659
    print()

    # --- Hypothesis tests ----------------------------------------------------
    var sample: List[Float64] = [5.1, 4.8, 5.3, 5.0, 4.9, 5.2]
    var res = ttest_1samp(sample, 5.0)
    print("ttest_1samp  t=", res[0], " p=", res[1])   # 0.654653670707975, 0.5416045608507769

    var a: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var b: List[Float64] = [4.0, 5.0, 6.0, 7.0, 8.0]
    var res2 = ttest_ind(a, b)
    print("ttest_ind    t=", res2[0], " p=", res2[1])  # -3.0, 0.0170716812337895

    var obs: List[Float64] = [16.0, 18.0, 16.0, 14.0, 12.0, 14.0]
    var exp: List[Float64] = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    var res3 = chi2_gof(obs, exp)
    print("chi2_gof   χ²=", res3[0], " p=", res3[1])  # 1.4666666666666666, 0.9168841203537823

    var g1: List[Float64] = [3.0, 4.0, 5.0]
    var g2: List[Float64] = [6.0, 7.0, 8.0]
    var g3: List[Float64] = [9.0, 10.0, 11.0]
    var groups = List[List[Float64]]()
    groups.append(g1^)
    groups.append(g2^)
    groups.append(g3^)
    var res4 = f_oneway(groups)
    print("f_oneway   F=", res4[0], " p=", res4[1])   # 27.0, 0.0010000000005315757
