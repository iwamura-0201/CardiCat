#########################
#### CardiCat ###########
#########################


import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
import scipy.stats as stats
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap
from umap import UMAP

sns.set_theme(style="ticks", color_codes=True)


def cardinality_table(X, Xhat, catCols):
    """Description.

    Parameters
    ----------
    par1 : type
            The first categorical variable
    par2 : type
            The second categorical variable

    """
    return pd.merge(
        X[catCols].nunique().reset_index(name="cardinality"),
        Xhat[catCols].nunique().reset_index(name="cardinality"),
        on="index",
        suffixes=["_original", "_generated"],
    )


def flatten_list(list_of_lists):
    """Description.

    Parameters
    ----------
    par1 : type
            The first categorical variable
    par2 : type
            The second categorical variable

    """
    return [item for sublist in list_of_lists for item in sublist]


def plot_marginals(X, Xhat, layer_sizes):
    """Description.

    Parameters
    ----------
    par1 : type
            The first categorical variable
    par2 : type
            The second categorical variable

    """
    ncols = 4
    nrows = int(np.ceil((len(Xhat.columns)) / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    plt.subplots_adjust(hspace=0.5)
    # fig.suptitle("Marginal Distributions:\n{} and {}".
    #              format(txt,txt_loss,plt_loss[0],plt_loss[1]), fontsize=18, y=0.95)

    # loop through tickers and axes
    for lsize, col, ax in zip(flatten_list(layer_sizes), Xhat.columns, axs.ravel()):
        # filter df for ticker and plot on specified axes\
        if lsize > 1:
            try:
                pd.concat(
                    [
                        X[col].value_counts().rename("original"),
                        Xhat[col].value_counts().rename("generated"),
                    ],
                    axis=1,
                ).iloc[:20].plot.bar(ax=ax, logy=True)
            except:
                print("REACHED EXCEPT IN plot_marginals ")
                pass
        else:
            try:
                pd.concat(
                    [X[col].rename("original"), Xhat[col].rename("generated")], axis=1
                ).plot.density(ax=ax)
            except:
                print("REACHED EXCEPT IN plot_marginals ")
                pass
        ax.set_title(col.upper())

    return fig


def cramersV(cat1, cat2):
    """Calculates the Cramer's V  which is a measure of correlation that follows
    from the Chi square test (for two categorical features).

    Parameters
    ----------
    cat1 : ndarray
            The first categorical variable
    cat2 : ndarray
            The second categorical variable

    See also: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    """
    dataset = np.array(pd.crosstab(cat1, cat2))
    X2 = stats.chi2_contingency(dataset, correction=False)[0]
    N = np.sum(dataset)
    minimum_dimension = min(dataset.shape) - 1
    # Calculate Cramer's V
    if minimum_dimension != 0:
        cramersV = np.sqrt((X2 / N) / minimum_dimension)
    else:
        cramersV = 0
    return cramersV


def get_catCols_cramersV(catCols, df):
    """_summary_

    Args:
        catCols (_type_): _description_
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    cramersV_df = pd.DataFrame(index=catCols, columns=catCols)
    couples = {comb for comb in combinations(catCols, r=2)}
    for en, coup in enumerate(couples):
        cramersV_df.loc[coup[0], coup[1]] = cramersV(df[coup[0]], df[coup[1]])
        cramersV_df.loc[coup[1], coup[0]] = cramersV(df[coup[1]], df[coup[0]])
    cramersV_df.fillna(0, inplace=True)
    return cramersV_df


# Anova test to check that the groups are not just random:\
def get_mixed_anova(catCols, intCols, floatCols, df):
    """_summary_

    Args:
        catCols (_type_): _description_
        intCols (_type_): _description_
        floatCols (_type_): _description_
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    anova_df = pd.DataFrame(index=catCols, columns=intCols + floatCols)
    for cat in catCols:
        for num in intCols + floatCols:
            try:
                anova_df.loc[cat, num] = float(
                    round(
                        sp.stats.f_oneway(
                            *[i for i in df.groupby(cat).agg(list)[num]],
                        )[1],
                        4,
                    )
                )
            except:
                anova_df.loc[cat, num] = np.NaN
    return anova_df.astype(float)


def get_qScoreMixed(originalDF, generatedDF):
    """Calculate the feature mean of the MSE of the anova values of mixed types"""
    assert originalDF.shape == originalDF.shape
    return 1 - (np.abs(originalDF - generatedDF)).mean().mean(), np.abs(
        originalDF - generatedDF
    )


def df_marginals(CardiCat_report, tVAE_report, tGAN_report):
    """_summary_

    Args:
        CardiCat_report (_type_): _description_
        tVAE_report (_type_): _description_
        tGAN_report (_type_): _description_
    """
    df = CardiCat_report["marginals"]
    df["model"] = "CardiCat"
    tVAE_report["marginals"]["model"] = "tVAE"
    tGAN_report["marginals"]["model"] = "tGAN"
    df = pd.concat([df, tVAE_report["marginals"], tGAN_report["marginals"]])
    df = df.rename(columns={"Column": "Feature"})
    return df


def df_pairs_corr(CardiCat_report, tVAE_report, tGAN_report):
    """_summary_

    Args:
        CardiCat_report (_type_): _description_
        tVAE_report (_type_): _description_
        tGAN_report (_type_): _description_
    """
    df = CardiCat_report["pairs"]
    df["model"] = "CardiCat"
    tVAE_report["pairs"]["model"] = "tVAE"
    tGAN_report["pairs"]["model"] = "tGAN"
    df = pd.concat([df, tVAE_report["pairs"], tGAN_report["pairs"]])
    df["pair"] = df["Column 1"] + "-" + df["Column 2"]
    df = df.loc[df["Metric"] == "CorrelationSimilarity"]
    # df = df.rename(columns={"Column":"Feature"})
    return df


def df_pairs_cat(CardiCat_report, tVAE_report, tGAN_report):
    """_summary_

    Args:
        CardiCat_report (_type_): _description_
        tVAE_report (_type_): _description_
        tGAN_report (_type_): _description_
    """
    df = CardiCat_report["pairs"]
    df["model"] = "CardiCat"
    tVAE_report["pairs"]["model"] = "tVAE"
    tGAN_report["pairs"]["model"] = "tGAN"
    df = pd.concat([df, tVAE_report["pairs"], tGAN_report["pairs"]])
    df["pair"] = df["Column 1"] + "-" + df["Column 2"]
    df = df.loc[df["Metric"] == "ContingencySimilarity"]
    # df = df.rename(columns={"Column":"Feature"})
    return df


# def ks_test(array_test, array_true):
#     """Applies a two sample (test,true) Kolmogorov–Smirnov test using the `stats` packages.

#     Args:
#         array_test (numpy.array/pandas.series): The test array (distribution)
#         array_true (numpy.array/pandas.series): _description_

#     Returns:
#         KstestResult: A KstestResult list object with the statistic value, p-value, location
#     """
#     return stats.kstest(array_test, array_true)


def ks_two_sample_mixed_test(cat_feature_name, num_feat_name, df_test, df_true):
    """Calculates the weighted mean of KS test statistic between two numerical arrays,
        given different values of a categorical feature.

    Args:
        cat_feature_name (str): The categorical feature name (should appear in both dfs)
        num_feat_name (str): The numerical feature name (should appear in both dfs)
        df_test (pandas.DataFrame): The dataframe of the test/generated data
        df_true (pandas.DataFrame): The dataframe of the true/real data


    Returns:
        list,list,list: The list of statistics for each cond value,
                        a list of indices that are "empty",
                        the categorical marginal probabilities
    """
    cat_values = df_true[cat_feature_name].unique()
    cat_probs = (
        df_true[cat_feature_name].value_counts() / df_true[cat_feature_name].count()
    )
    cond_stat_list = []
    empty_values_indx = []
    for en, cat in enumerate(cat_values):
        array_test = df_test[df_test[cat_feature_name] == cat][num_feat_name]
        array_true = df_true[df_true[cat_feature_name] == cat][num_feat_name]
        try:
            cond_stat_list.append(stats.kstest(array_test, array_true)[0])
        except ValueError:
            # print("array_test",array_test)
            cond_stat_list.append("empty")
            empty_values_indx.append(en)
    return cond_stat_list, empty_values_indx, cat_probs


def ks_mixed_statistic(
    cond_stat_list, empty_values_indx, cat_probs, weighted=True, empty_set_f_value=0
):
    """Using the vector of cond_stat_list (the F values of the conditional
        probs of each value of a categorical variable), applies a weighted mean
        and returns the complement of that.

    Args:
        cond_stat_list (_type_): _description_
        empty_values_indx (_type_): _description_
        weighted (bool, optional): If true, will weight each categorical value by the marginal
                                    categorical probability (pi). Defaults to True.
        empty_set_f_value (int, optional): _description_. Defaults to 0.

    Returns:
        float: the complement of the weighted mean KS mixed divergent.
    """

    cond_stat_list = [F if F != "empty" else empty_set_f_value for F in cond_stat_list]

    cat_probs_reweighted = cat_probs.copy()
    # for en in empty_values_indx:
    #     if cond_stat_list[en] == 0:  # if missing value is set to F=0
    #         cat_probs_reweighted[en] = 0  # readjust cat_probs
    # # making sure all prob (not equal to zero) sum up to 1:
    # cat_probs_reweighted = np.array(cat_probs_reweighted) / sum(cat_probs_reweighted)
    if weighted:
        mean_stat = sum((cond_stat_list) * cat_probs_reweighted) #/ len( cat_probs_reweighted )
    else:
        mean_stat = sum((cond_stat_list)) / len(cat_probs_reweighted)

    return 1 - mean_stat


def get_mean_ks_mixed_stats(df_test, df_true, catCols, intCols, floatCols):
    """Returns a dictionary of the mean_ks_mixed statistic all the possible
        combination of mixed (categorical+numerical) features.

    Args:
        df_test (pandas.DataFrame): The dataframe of the test/generated data
        df_true (pandas.DataFrame): The dataframe of the true/real data
        weighted (bool, optional): If true, will weight each categorical value by the marginal
                                    categorical probability (pi). Defaults to True.

    Returns:
        float,float,float,fict: ks_mixed_0_weighted, ks_mixed_1_weighted, ks_mixed_0 scores
                                A dictionary of cat_feature:num_feature
                                keys and their mean ks statistic.
    """
    catCols = catCols
    numCols = intCols + floatCols
    mean_ks_mixed_0_weighted = {}
    mean_ks_mixed_1_weighted = {}
    mean_ks_mixed_0 = {}
    mean_ks_mixed_1 = {}
    ks_mixed_raw_stats = {}
    for cat in catCols:
        for num in numCols:
            cond_stat_list, empty_values_indx, cat_probs = ks_two_sample_mixed_test(
                cat, num, df_test, df_true
            )
            ks_mixed_0_weighted = ks_mixed_statistic(
                cond_stat_list,
                empty_values_indx,
                cat_probs,
                weighted=True,
                empty_set_f_value=0,
            )
            ks_mixed_1_weighted = ks_mixed_statistic(
                cond_stat_list,
                empty_values_indx,
                cat_probs,
                weighted=True,
                empty_set_f_value=1,
            )
            ks_mixed_0 = ks_mixed_statistic(
                cond_stat_list,
                empty_values_indx,
                cat_probs,
                weighted=False,
                empty_set_f_value=0,
            )

            ks_mixed_1 = ks_mixed_statistic(
                cond_stat_list,
                empty_values_indx,
                cat_probs,
                weighted=False,
                empty_set_f_value=1,
            )

            mean_ks_mixed_0_weighted["{}:{}".format(cat, num)] = ks_mixed_0_weighted
            mean_ks_mixed_1_weighted["{}:{}".format(cat, num)] = ks_mixed_1_weighted
            mean_ks_mixed_0["{}:{}".format(cat, num)] = ks_mixed_0
            mean_ks_mixed_1["{}:{}".format(cat, num)] = ks_mixed_1
            ks_mixed_raw_stats["{}:{}".format(cat, num)] = cond_stat_list

    return (
        mean_ks_mixed_0_weighted,
        mean_ks_mixed_1_weighted,
        mean_ks_mixed_0,
        mean_ks_mixed_1,
        ks_mixed_raw_stats,
    )


# def get_emb_spaces(
#     embCols, label_encoder, emb_weights, n_components=2, method="PCA", n_neighbors=None
# ):
#     """Description.

#     Parameters
#     ----------
#     par1 : type
#             The first categorical variable
#     par2 : type
#             The second categorical variable

#     """
#     emb_spaces_list = {}
#     for emb in embCols:
#         emb_wts = emb_weights["emb_" + emb]
#         emb_labels = label_encoder[emb].inverse_transform(range(len(emb_wts)))
#         if not n_neighbors:
#             if len(emb_labels) < 10:
#                 n_neighbors = 2
#             elif len(emb_labels) < 50:
#                 n_neighbors = 4
#             else:
#                 n_neighbors = 6
#         if method == "UMAP":
#             umap_2d = UMAP(
#                 n_neighbors=n_neighbors, n_components=n_components, init="spectral"
#             )
#             X_dimensions = umap_2d.fit_transform(emb_wts.copy())
#         elif method == "PCA":
#             PCA_2d = PCA(n_components=n_components)
#             X_dimensions = PCA_2d.fit_transform(emb_wts.copy())
#         elif method == "KernelPCA":
#             KernelPCA_2d = KernelPCA(n_components=n_components, kernel="rbf")
#             X_dimensions = KernelPCA_2d.fit_transform(emb_wts.copy())
#         elif method == "Isomap":
#             Isomap_2d = Isomap(n_neighbors=n_neighbors, n_components=n_components)
#             X_dimensions = Isomap_2d.fit_transform(emb_wts.copy())
#         elif method == "MDS":
#             MDS_2d = MDS(n_components=n_components, metric=True)
#             X_dimensions = MDS_2d.fit_transform(emb_wts.copy())

#         # create a dataframe from the dataset
#         df = pd.DataFrame(
#             data=X_dimensions, columns=["Dimension 1", "Dimension 2"]
#         )  # merge this with the NBA data
#         df["labels"] = emb_labels
#         emb_spaces_list[emb] = df
#     return emb_spaces_list

# def label_point(x, y, val, ax, font_sizes):
#     """Description.

#     Parameters
#     ----------
#     par1 : type
#             The first categorical variable
#     par2 : type
#             The second categorical variable

#     """
#     a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
#     if len(val) > 25:
#         alpha = 0.8
#         fontsize = font_sizes[0]
#     else:
#         alpha = 1
#         fontsize = font_sizes[1]
#     for i, point in a.iterrows():
#         ax.text(
#             point["x"] + 0.02,
#             point["y"],
#             str(point["val"]),
#             alpha=alpha,
#             fontsize=fontsize,
#         )


# def plot_emb_spaces(embCols, emb_spaces, main_title="", font_sizes=(10, 16)):
#     """Description.

#     Parameters
#     ----------
#     par1 : type
#             The first categorical variable
#     par2 : type
#             The second categorical variable

#     """
#     fig, axs = plt.subplots(1, len(embCols), figsize=(15 * len(embCols), 15))
#     fig.suptitle(main_title, fontsize=40)
#     # plt.subplots_adjust(hspace=10)
#     for ax, emb in zip(axs, embCols):
#         ax = sns.scatterplot(
#             x="Dimension 1", y="Dimension 2", data=emb_spaces[emb], ax=ax
#         )
#         ax.set_title(emb)
#         label_point(
#             emb_spaces[emb]["Dimension 1"],
#             emb_spaces[emb]["Dimension 2"],
#             emb_spaces[emb].labels,
#             ax,
#             font_sizes,
#         )
#     plt.close()
#     return fig

# def plot_marginals(
#     dataframe, lst_benchmark_dfs, layer_sizes, reports, outlogs, col_types
# ):
#     """_summary_

#     Args:
#         dataframe (_type_): _description_
#         lst_benchmark_dfs (_type_): _description_
#         layer_sizes (_type_): _description_
#         reports (_type_): _description_

#     Returns:
#         _type_: _description_
#     """

#     catCols, intCols, floatCols = col_types[0], col_types[1], col_types[2]
#     # palette_tri = ['#0173B2','#0173B2','#0173B2','#DE8F05','#DE8F05','#DE8F05',
#     #                '#D55E00','#D55E00','#D55E00']
#     palette_tri = ["#0173B2", "#DE8F05", "#D55E00"]
#     # palette_tri = {'CardiCat':'#0173B2',
#     #               'tVAE':'#DE8F05',
#     #               'tGAN':'#D55E00'}
#     Xhat = lst_benchmark_dfs[0]
#     # nrows=len(flatten_list([catCols,intCols,floatCols]))+4
#     nrows = len(dataframe.columns) + 5

#     fig, axs = plt.subplots(nrows=nrows, ncols=4, figsize=(22, nrows * 5))
#     plt.subplots_adjust(hspace=0.5)
#     # loop through tickers and axes
#     sns.set(font_scale=1)

#     # High level Stats:
#     sns.barplot(
#         pd.concat([outlogs[0], outlogs[1], outlogs[2]]),
#         x="model",
#         y="score_rand",
#         ax=axs[0][0],
#         palette=["#0173B2", "#DE8F05", "#D55E00"],
#     ).set(title="Average of all Quality-Scores")
#     sns.barplot(
#         pd.concat([outlogs[0], outlogs[1], outlogs[2]]).melt(
#             id_vars="model",
#             value_vars=[
#                 "score_rand_marginals",
#                 "score_rand_marginals_KS",
#                 "score_rand_marginals_TV",
#             ],
#         ),
#         x="model",
#         y="value",
#         hue="variable",
#         ax=axs[0][1],
#         palette=palette_tri,
#     ).set(title="Average of all Quality-Scores")
#     sns.move_legend(axs[0][1], "lower left", frameon=False, fontsize="small", title="")

#     sns.barplot(
#         pd.concat([outlogs[0], outlogs[1], outlogs[2]]).melt(
#             id_vars="model",
#             value_vars=[
#                 "score_rand_pairs",
#                 "score_rand_pairs_corr",
#                 "score_rand_pairs_cont",
#             ],
#         ),
#         x="model",
#         y="value",
#         hue="variable",
#         ax=axs[0][2],
#     ).set(title="Average of Pairs Quality-Scores")
#     sns.move_legend(axs[0][2], "lower left", frameon=False, fontsize="small", title="")

#     for en, (lsize, col) in enumerate(zip(flatten_list(layer_sizes), Xhat.columns)):
#         en = en + 1
#         # print(en)
#         # filter df for ticker and plot on specified axes\
#         if lsize > 1:
#             # try:
#             # print("{}".format(round(reports[0][reports[0].Column=='age']['Quality Score'][0],3)))
#             dataframe[col].value_counts().rename("original").iloc[:20].plot.bar(
#                 ax=axs[en][0], logy=False, title="{}: original".format(col)
#             )

#             pd.concat(
#                 [
#                     dataframe[col].value_counts().rename("original"),
#                     lst_benchmark_dfs[0][col].value_counts().rename("CardiCat"),
#                 ],
#                 axis=1,
#             ).iloc[:20].plot.bar(
#                 ax=axs[en][1],
#                 logy=False,
#                 title="{}: CardiCat \nKSComplement: {}".format(
#                     col,
#                     round(
#                         reports[0]["marginals"][reports[0]["marginals"].Column == col][
#                             "Quality Score"
#                         ].to_numpy()[0],
#                         3,
#                     ),
#                 ),
#             )
#             axs[en][1].legend(prop={"size": 7})

#             pd.concat(
#                 [
#                     dataframe[col].value_counts().rename("original"),
#                     lst_benchmark_dfs[1][col].value_counts().rename("tVAE"),
#                 ],
#                 axis=1,
#             ).iloc[:20].plot.bar(
#                 ax=axs[en][2],
#                 logy=False,
#                 title="{}: tVAE \nKSComplement: {}".format(
#                     col,
#                     round(
#                         reports[1]["marginals"][reports[1]["marginals"].Column == col][
#                             "Quality Score"
#                         ].to_numpy()[0],
#                         3,
#                     ),
#                 ),
#             )
#             axs[en][2].legend(prop={"size": 7})

#             pd.concat(
#                 [
#                     dataframe[col].value_counts().rename("original"),
#                     lst_benchmark_dfs[2][col].value_counts().rename("tGAN"),
#                 ],
#                 axis=1,
#             ).iloc[:20].plot.bar(
#                 ax=axs[en][3],
#                 logy=False,
#                 title="{}: tGAN \nKSComplement: {}".format(
#                     col,
#                     round(
#                         reports[2]["marginals"][reports[2]["marginals"].Column == col][
#                             "Quality Score"
#                         ].to_numpy()[0],
#                         3,
#                     ),
#                 ),
#             )
#             axs[en][3].legend(prop={"size": 7})

#             # except:
#             #     pass
#         else:
#             try:
#                 dataframe[col].rename("original").plot.density(
#                     ax=axs[en][0], title="{}: original".format(col)
#                 )

#                 pd.concat(
#                     [
#                         dataframe[col].rename("original"),
#                         lst_benchmark_dfs[0][col].rename("CardiCat"),
#                     ],
#                     axis=1,
#                 ).plot.density(
#                     ax=axs[en][1],
#                     color=["b", "c"],
#                     title="{}: CardiCat \nTVComplement: {}".format(
#                         col,
#                         round(
#                             reports[0]["marginals"][
#                                 reports[0]["marginals"].Column == col
#                             ]["Quality Score"].to_numpy()[0],
#                             3,
#                         ),
#                     ),
#                 )
#                 axs[en][1].legend(prop={"size": 7})

#                 pd.concat(
#                     [
#                         dataframe[col].rename("original"),
#                         lst_benchmark_dfs[1][col].rename("tVAE"),
#                     ],
#                     axis=1,
#                 ).plot.density(
#                     ax=axs[en][2],
#                     color=["b", "y"],
#                     title="{}: tVAE \nTVComplement: {}".format(
#                         col,
#                         round(
#                             reports[1]["marginals"][
#                                 reports[1]["marginals"].Column == col
#                             ]["Quality Score"].to_numpy()[0],
#                             3,
#                         ),
#                     ),
#                 )
#                 axs[en][2].legend(prop={"size": 7})

#                 pd.concat(
#                     [
#                         dataframe[col].rename("original"),
#                         lst_benchmark_dfs[2][col].rename("tGAN"),
#                     ],
#                     axis=1,
#                 ).plot.density(
#                     ax=axs[en][3],
#                     color=["b", "g"],
#                     title="{}: tGAN \nTVComplement: {}".format(
#                         col,
#                         round(
#                             reports[2]["marginals"][
#                                 reports[2]["marginals"].Column == col
#                             ]["Quality Score"].to_numpy()[0],
#                             3,
#                         ),
#                     ),
#                 )
#                 axs[en][3].legend(prop={"size": 7})
#             except:
#                 pass

#     sns.set(font_scale=0.6)

#     rw_general = nrows - 4
#     out_fig = sns.barplot(
#         df_marginals(reports[0], reports[1], reports[2]),
#         x="Feature",
#         y="Quality Score",
#         hue="model",
#         width=0.5,
#         ax=axs[rw_general][0],
#     )
#     out_fig.set(title="Marginals quality score")
#     out_fig.set_xticklabels(out_fig.get_xticklabels(), rotation=45)
#     out_fig = sns.scatterplot(
#         df_pairs_corr(reports[0], reports[1], reports[2]),
#         x="pair",
#         y="Quality Score",
#         hue="model",
#         ax=axs[rw_general][1],
#     )  # width=0.3,
#     out_fig.set(title="Correlation Similarity scores")
#     out_fig.set_xticklabels(out_fig.get_xticklabels(), rotation=45)
#     out_fig = sns.scatterplot(
#         df_pairs_cat(reports[0], reports[1], reports[2]),
#         x="pair",
#         y="Quality Score",
#         hue="model",
#         ax=axs[rw_general][2],
#     )
#     out_fig.set(title="Contingency Similarity scores")
#     out_fig.set_xticklabels(out_fig.get_xticklabels(), rotation=45)
#     rw_corr = nrows - 3
#     corr_matrix_original = dataframe[intCols + floatCols].corr().round(2)
#     corr_mask_original = np.triu(np.ones_like(corr_matrix_original, dtype=bool))
#     sns.heatmap(
#         corr_matrix_original,
#         center=0,
#         annot=True,
#         ax=axs[rw_corr][0],
#         mask=corr_mask_original,
#         vmax=1,
#     )
#     corr_matrix_cardicat = lst_benchmark_dfs[0][intCols + floatCols].corr().round(2)
#     corr_mask_cardicat = np.triu(np.ones_like(corr_matrix_cardicat, dtype=bool))
#     sns.heatmap(
#         corr_matrix_cardicat,
#         center=0,
#         annot=True,
#         ax=axs[rw_corr][1],
#         mask=corr_mask_cardicat,
#         vmax=1,
#     )
#     corr_matrix_tvae = lst_benchmark_dfs[1][intCols + floatCols].corr().round(2)
#     corr_mask_tave = np.triu(np.ones_like(corr_matrix_tvae, dtype=bool))
#     sns.heatmap(
#         corr_matrix_tvae,
#         center=0,
#         annot=True,
#         ax=axs[rw_corr][2],
#         mask=corr_mask_tave,
#         vmax=1,
#     )
#     corr_matrix_tgan = lst_benchmark_dfs[2][intCols + floatCols].corr().round(2)
#     corr_mask_tgan = np.triu(np.ones_like(corr_matrix_tgan, dtype=bool))
#     sns.heatmap(
#         corr_matrix_tgan,
#         center=0,
#         annot=True,
#         ax=axs[rw_corr][3],
#         mask=corr_mask_tgan,
#         vmax=1,
#     )

#     rw_cramer = nrows - 2
#     cramersV_original = get_catCols_cramersV(catCols, dataframe)
#     cramersV_mask_original = np.triu(np.ones_like(cramersV_original, dtype=bool))
#     sns.heatmap(
#         cramersV_original,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="coolwarm",
#         ax=axs[rw_cramer][0],
#         mask=cramersV_mask_original,
#         vmax=1,
#     )
#     cramersV_cardicat = get_catCols_cramersV(catCols, lst_benchmark_dfs[0])
#     cramersV_mask_cardicat = np.triu(np.ones_like(cramersV_cardicat, dtype=bool))
#     sns.heatmap(
#         cramersV_cardicat,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="coolwarm",
#         ax=axs[rw_cramer][1],
#         mask=cramersV_mask_cardicat,
#         vmax=1,
#     )
#     cramersV_tvae = get_catCols_cramersV(catCols, lst_benchmark_dfs[1])
#     cramersV_mask_tvae = np.triu(np.ones_like(cramersV_tvae, dtype=bool))
#     sns.heatmap(
#         cramersV_tvae,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="coolwarm",
#         ax=axs[rw_cramer][2],
#         mask=cramersV_mask_tvae,
#         vmax=1,
#     )
#     cramersV_tgan = get_catCols_cramersV(catCols, lst_benchmark_dfs[2])
#     cramersV_mask_tgan = np.triu(np.ones_like(cramersV_tgan, dtype=bool))
#     sns.heatmap(
#         cramersV_tgan,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="coolwarm",
#         ax=axs[rw_cramer][3],
#         mask=cramersV_mask_tgan,
#         vmax=1,
#     )

#     rw_mixed = nrows - 1
#     anove_mixed_original = get_mixed_anova(catCols, intCols, floatCols, dataframe)
#     sns.heatmap(
#         anove_mixed_original,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="rocket_r",
#         ax=axs[rw_mixed][0],
#         vmax=1,
#     )
#     anove_mixed_cardicat = get_mixed_anova(
#         catCols, intCols, floatCols, lst_benchmark_dfs[0]
#     )
#     sns.heatmap(
#         anove_mixed_cardicat,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="rocket_r",
#         ax=axs[rw_mixed][1],
#         vmax=1,
#     )
#     anove_mixed_tvae = get_mixed_anova(
#         catCols, intCols, floatCols, lst_benchmark_dfs[1]
#     )
#     sns.heatmap(
#         anove_mixed_tvae,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="rocket_r",
#         ax=axs[rw_mixed][2],
#         vmax=1,
#     )
#     anove_mixed_tgan = get_mixed_anova(
#         catCols, intCols, floatCols, lst_benchmark_dfs[2]
#     )

#     sns.heatmap(
#         anove_mixed_tgan,
#         annot=True,
#         fmt=".2f",
#         linewidth=0.8,
#         cmap="rocket_r",
#         ax=axs[rw_mixed][3],
#         vmax=1,
#     )
#     # axs[en].set_title(col.upper())
#     # axs[en][0].legend(size=8)

#     qScore_mixed_cardicat = get_qScoreMixed(anove_mixed_original, anove_mixed_cardicat)
#     qScore_mixed_tvae = get_qScoreMixed(anove_mixed_original, anove_mixed_tvae)
#     qScore_mixed_tgan = get_qScoreMixed(anove_mixed_original, anove_mixed_tgan)
#     pd.Series(
#         {
#             "CardiCat": qScore_mixed_cardicat,
#             "tVAE": qScore_mixed_tvae,
#             "tGAN": qScore_mixed_tgan,
#         }
#     ).plot.bar(ax=axs[0][3]).set(title="Average of the Anova scores MSE")

#     return fig
