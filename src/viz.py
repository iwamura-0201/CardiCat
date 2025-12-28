#########################
#### CardiCat ###########
#########################


import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from plotly.subplots import make_subplots

from src import postprocessing as postprocessing
from src import preprocessing as preprocessing
from src import reporting as reporting

pio.templates.default = "plotly_white"
sns.set(style="ticks", color_codes=True)

comp_name = ""
lib_path = "/Users/{}/Dropbox/School/gitRepos/CardiCat".format(comp_name)
# Loading the CardiCat module:
sys.path.insert(1, lib_path)


class cardicat_viz_general:
    def __init__(self, reports_path, models_order, datasets_order,drop_metrics,mycols):
        self.reports_path = reports_path
        self.reports_files_list = []
        self.reports_df = pd.DataFrame()
        self.models_order = models_order
        self.datasets_order = datasets_order
        self.drop_metrics = drop_metrics
        self.mycols = mycols

    def get_report_files_list(self):
        self.reports_files_list = [
            self.reports_path + f
            for f in os.listdir(self.reports_path)
            if f != ".DS_Store" and f != "CardiCat_Old"
        ]

    def load_reports(self):
        if not self.reports_files_list:
            self.get_report_files_list()
        self.reports_df = pd.concat(
            (pd.read_csv(f, index_col=0) for f in self.reports_files_list)
        )
        self.reports_df["scores"] = self.reports_df.scores.apply(ast.literal_eval)
        self.reports_df["marginals"] = self.reports_df.marginals.apply(ast.literal_eval)
        self.reports_df["pairs"] = self.reports_df.pairs.apply(
            lambda x: ast.literal_eval(x.replace("nan", "0"))
        )
        self.reports_df["cramersV"] = self.reports_df.cramersV.apply(
            lambda x: ast.literal_eval(x.replace("nan", "0"))
        )
        self.reports_df["mixed"] = self.reports_df.mixed.apply(
            lambda x: ast.literal_eval(x.replace("nan", "0"))
        )
        self.reports_df = self.reports_df.reset_index()

    def evaluation_summary_viz(self):
        if self.reports_df.empty:
            self.load_reports()

        tmp_fig_summary = (
            self.reports_df[["model", "dataset_log_name"]]
            .join(pd.json_normalize(self.reports_df["scores"]))
            .set_index("model")
        )

        datasets = tmp_fig_summary.dataset_log_name.unique()
        datasets = [y for x in self.datasets_order for y in datasets if y == x]
        models = self.reports_df.model.unique()
        models = [y for x in self.models_order for y in models if y == x]
        dataset_num = datasets.__len__()

        # tmp_fig_summary = tmp_fig_summary.rename(
        #     {
        #         "marginals_KS": "marginals-numerical",
        #         "marginals_TV": "marginals-categorical",
        #         "pairs_corr": "pairs-correlation",
        #         "pairs_cont": "pairs-contigency",
        #         "pairs_mixed": "pairs-mixed",
        #         "pairs_cat": "pairs-categorical",
        #     },
        #     axis=1,
        # )
        
        tmp_fig_summary = tmp_fig_summary.rename(
            {
                "marginals_KS": "marginals-numerical",
                "marginals_TV": "marginals-categorical",
                "pairs_corr": "pairs-correlation",
                "pairs_cont": "pairs-contigency",
                "pairs_mixed": "pairs-mixed-old",
                "pairs_cat": "pairs-categorical-anova",
                "pairs_cont_fix":"pairs-categorical",
                "pairs_ks_mixed_1_weighted":"pairs-mixed",
            },
            axis=1,
        )
        
        # tmp_fig_summary = tmp_fig_summary[[self.metrics]]
        tmp_fig_summary = tmp_fig_summary.drop(
            # ["marginals_all", "pairs_all", "pairs-contigency"], axis=1
            self.drop_metrics,
            axis=1,
        )
        self.evaluation_summary_df = tmp_fig_summary

        # self.tmp_fig_summary = tmp_fig_summary
        fig = make_subplots(rows=dataset_num, cols=1, subplot_titles=datasets)

        for i, set in enumerate(datasets):
            tmp_final = tmp_fig_summary[tmp_fig_summary.dataset_log_name == set].drop(
                "dataset_log_name", axis=1
            )
            for j, mod in enumerate(models):
                showLegend = [True if i == dataset_num - 1 else False]
                stds = tmp_final.filter(regex="^{}".format(mod), axis=0).std(axis=0)
                means = tmp_final.filter(regex="^{}".format(mod), axis=0).mean(axis=0)
                fig.add_trace(
                    go.Bar(
                        name=mod,
                        x=means.keys(),
                        y=means.values,
                        legendgroup=set,  # text_auto='0.2f' ,
                        marker_color=self.mycols[j],
                        error_y=dict(
                            type="data",
                            array=stds,
                        ),
                        showlegend=showLegend[0],
                        # xaxis = showLegend[0]
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(
            barmode="group",
            height=1200,
            width=1000,
            title_text="Evaluation metrics summary",
        )
        # fig.update_xaxes(
        #     tickmode="array",
        #     # categoryorder="total ascending",
        #     tickvals=models,
        #     ticktext=models,
        #     ticklabelposition="inside",
        #     tickfont=dict(color="white"),
        # )

        # for i in range(dataset_num-1):
        #     fig.update_xaxes(showticklabels=False,row=1+i)
        return fig

    def evaluation_overall(self):
        if self.reports_df.empty:
            self.load_reports()

        tmp_fig_summary = (
            self.reports_df[["model", "dataset_log_name"]]
            .join(pd.json_normalize(self.reports_df["scores"]))
            .set_index("model")
        )

        # datasets = tmp_fig_summary.dataset_log_name.unique()
        models = self.reports_df.model.unique()
        models = [y for x in self.models_order for y in models if y == x]
        # dataset_num = datasets.__len__()

        tmp_fig_summary = tmp_fig_summary.rename(
            {
                "marginals_KS": "marginals-numerical",
                "marginals_TV": "marginals-categorical",
                "pairs_corr": "pairs-correlation",
                "pairs_cont": "pairs-contigency",
                "pairs_mixed": "pairs-mixed-old",
                "pairs_cat": "pairs-categorical-anova",
                "pairs_cont_fix":"pairs-categorical",
                "pairs_ks_mixed_1_weighted":"pairs-mixed",
            },
            axis=1,
        )

        tmp_fig_summary = tmp_fig_summary.drop(
            # ["marginals_all", "pairs_all", "pairs-contigency"], axis=1
            self.drop_metrics,
            axis=1,
        )
        tmp_fig_summary = tmp_fig_summary[tmp_fig_summary.index.isin(models)]
        # tmp_fig_summary.reset_index()#.set_index("model")
        # tt = tmp_fig_summary.reset_index()
        # aa = tt[tt.model=="VAE"]
        # aa["model"]="space"
        # aa[['marginals-numerical','marginals-categorical', 'pairs-correlation', 'pairs_cont_fix',
        #     'pairs_ks_mixed_0_weighted', 'pairs_ks_mixed_1_weighted']]=0
        # bb = aa.groupby(['model','dataset_log_name']).mean().reset_index()
        # tmp_fig_summary = pd.concat([tmp_fig_summary.reset_index(),bb]).set_index("model")

        self.fig_summary = tmp_fig_summary[['marginals-categorical','marginals-numerical', 
        'pairs-categorical', 'pairs-mixed' ,'pairs-correlation',]]
        
        fig = px.bar(
            self.fig_summary.groupby(["model"])
            .mean(numeric_only=True)
            .transpose()[models],
            barmode="group",
            # width = 2,
            color_discrete_sequence=self.mycols,
            text_auto=".2f",
            labels={"index": "evaluation metric", "value": "score"},
            width=2000,
            height=500,
            # category_orders = 
            # pattern_shape_sequence=[["x","x", "x", "-", "-"]]*3+[["x","x", "x", "x", "x"]]*3,
            # title="Overall evaluation metrics",
            # error_y=dict(type='data', array=tmp_fig_summary.groupby(['model']).std(numeric_only=True).transpose().values.tolist(), )
        )
        return fig


class cardicat_viz:
    def __init__(self, dataset, data_path, synthetics_path):
        self.dataset = dataset
        self.data_path = data_path
        self.synthetics_path = synthetics_path
        self.df_original_train = pd.DataFrame()
        self.df_original_test = pd.DataFrame()

    def load_original_data(self):
        df_original, self.is_target = preprocessing.load_dataset(
            self.dataset, self.data_path
        )
        self.df_original_train, self.df_original_test = np.split(
            df_original.sample(frac=1), [int(0.7 * len(df_original))]
        )
        print(
            "Train data size: ",
            self.df_original_train.shape,
            "Test data size: ",
            self.df_original_test.shape,
        )
        self.catCols, self.intCols, self.floatCols = preprocessing.get_col_types(
            self.df_original_test, is_y=self.is_target
        )
        self.col_tokens_all = preprocessing.get_cat_tokens(df_original, self.catCols)

    def load_synthetics(self):
        if self.df_original_train.empty:
            self.load_original_data()
        synthetics_file_list = [
            self.synthetics_path + f
            for f in os.listdir(self.synthetics_path)
            if f != ".DS_Store"
        ]
        self.synths_dict = {}
        for f in synthetics_file_list:
            f_split = f.split("synthetics_pkl_")[1]

            if (("_cCardiCatMask") in f_split) and (self.dataset in f_split):
                self.synths_dict["cCardiCat"] = pd.read_pickle(f)

            if (("_CardiCatAttention") in f_split) and (self.dataset in f_split):
                self.synths_dict["CardiCatAttention"] = pd.read_pickle(f)

            if (("_CardiCat") in f_split) and (self.dataset in f_split):
                self.synths_dict["CardiCat"] = pd.read_pickle(f)

            if (("_VAE") in f_split) and (self.dataset in f_split):
                self.synths_dict["VAE"] = pd.read_pickle(f)

            if (("_tVAE") in f_split) and (self.dataset in f_split):
                self.synths_dict["tVAE"] = pd.read_pickle(f)

            if (("_tGAN") in f_split) and (self.dataset in f_split):
                self.synths_dict["tGAN"] = pd.read_pickle(f)

    def get_report(self, df_real, df_synth):
        # if df_synth.empty:
        #     self.load_synthetics()
        assert not df_real.empty
        assert not df_synth.empty

        param_dict = {"is_target": self.is_target}
        report = postprocessing.get_report(df_real, df_synth, param_dict, full=True)

        (
            ks_mixed_0_weighted,
            ks_mixed_1_weighted,
            ks_mixed_0,
            ks_mixed_1,
            ks_mixed_raw_stats,
        ) = reporting.get_mean_ks_mixed_stats(
            df_synth,
            df_real,
            self.catCols,
            self.intCols,
            self.floatCols,
        )

        anove_mixed_real = reporting.get_mixed_anova(
            self.catCols, self.intCols, self.floatCols, df_real
        )
        cramersV_real = reporting.get_catCols_cramersV(self.catCols, df_real)

        anove_mixed_synth = reporting.get_mixed_anova(
            self.catCols, self.intCols, self.floatCols, df_synth
        )
        qScore_mixed_synth, mixed_synth = reporting.get_qScoreMixed(
            anove_mixed_real, anove_mixed_synth
        )
        cramersV_synth = reporting.get_catCols_cramersV(self.catCols, df_synth)
        
        score_cat_pairs, pairs_synth = reporting.get_qScoreMixed(
            cramersV_real, cramersV_synth
        )

        scores = {
            "marginals": round(report["summary"].iloc[0, 1], 2),
            "pairs": round(report["summary"].iloc[1, 1], 2),
            "marginals_KS": round(
                report["marginals"][report["marginals"].Metric == "KSComplement"][
                    "Quality Score"
                ].mean(),
                2,
            ),
            "marginals_TV": round(
                report["marginals"][report["marginals"].Metric == "TVComplement"][
                    "Quality Score"
                ].mean(),
                2,
            ),
            "pairs_corr": round(
                report["pairs"][report["pairs"].Metric == "CorrelationSimilarity"][
                    "Quality Score"
                ].mean(),
                2,
            ),
            "pairs_cont": round(
                report["pairs"][report["pairs"].Metric == "ContingencySimilarity"][
                    "Quality Score"
                ].mean(),
                2,
            ),
            "pairs_cont_fix": round(
                report["pairs"][
                    (report["pairs"]["Column 1"].isin(self.catCols))
                    & (report["pairs"]["Column 2"].isin(self.catCols))
                ][report["pairs"].Metric == "ContingencySimilarity"][
                    "Quality Score"
                ].mean(),
                2,
            ),
            "pairs_mixed": round(qScore_mixed_synth, 2),
            "pairs_cat": round(score_cat_pairs, 2),
            "marginal_all": report["marginals"],
            "pairs_all": report["pairs"],
            "pairs_ks_mixed_0_weighted": round(
                np.mean(list(ks_mixed_0_weighted.values())), 3
            ),
            "pairs_ks_mixed_1_weighted": round(
                np.mean(list(ks_mixed_1_weighted.values())), 3
            ),
            "pairs_ks_mixed_0": round(np.mean(list(ks_mixed_0.values())), 3),
            "pairs_ks_mixed_1": round(np.mean(list(ks_mixed_1.values())), 3),
            "pairs_ks_mixed_all":ks_mixed_raw_stats,
            "pairs_synth":pairs_synth,
            "mixed_synth":mixed_synth,
        }

        return scores

    def get_all_reports(self, df_real):
        self.scores_dict = {}
        for synth_name, synth in self.synths_dict.items():
            print(f"Getting {synth_name} scores\n")
            self.scores_dict[synth_name] = self.get_report(df_real, synth)

    def plot_report_summary(self, name_order_list):
        summary_figs_dict = {}
        all_reports_df = pd.DataFrame(
            [self.scores_dict[mod] for mod in name_order_list],
            columns=self.scores_dict["CardiCat"].keys(),
            index=name_order_list,
        )

        ### summary figure ###
        summary_df = all_reports_df[
            [
                "marginals_KS",
                "marginals_TV",
                "pairs_corr",
                # "pairs_cont",
                "pairs_cont_fix",
                # "pairs_mixed",
                # "pairs_cat",
                "pairs_ks_mixed_0_weighted",
                "pairs_ks_mixed_1_weighted",
                # "pairs_ks_mixed_0",
                # "pairs_ks_mixed_1",
                # 'marginal_all', 'pairs_all'
            ]
        ]
        summary_df = summary_df.transpose()
        summary_df = summary_df.rename(
            {
                "marginals_KS": "marginals-numerical",
                "marginals_TV": "marginals-categorical",
                "pairs_corr": "pairs-correlation",
                "pairs_cont": "pairs-contigency",
                "pairs_mixed": "pairs-mixed",
                "KS_mixed_all": "pairs-mixed-ks",
                "pairs_cat": "pairs-categorical",
            },
            axis=0,
        )

        fig_summary = px.bar(
            summary_df,
            barmode="group",
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="{}: evaluation metrics".format(self.dataset),
            labels={
                "value": "score",
                "index": "reconstruction metrics",
            },
        )
        fig_summary.update_layout(legend_title="model")
        summary_figs_dict["eval_summary"] = fig_summary

        ### marginals figure ###
        marginals_dict = all_reports_df["marginal_all"].to_dict()
        marginals_df = []
        for mod, dat in marginals_dict.items():
            dat["model"] = mod
            marginals_df.append(dat)

        fig_marginals = px.bar(
            pd.concat(marginals_df).drop(["Metric"], axis=1),
            x="Column",
            y="Quality Score",
            color="model",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="{}: marginal reconstruction".format(self.dataset),
            labels={
                "Quality Score": "Reconstruction Score",
                "Column": "Feature",
            },
        )
        fig_marginals.update_layout(legend_title="model")

        for mod in range(len(name_order_list)):
            for i, k in enumerate(fig_marginals.data[mod]["x"]):
                if k in self.col_tokens_all:
                    fig_marginals.data[mod]["x"][i] = "{}:{}".format(
                        k, self.col_tokens_all[k]
                    )
        summary_figs_dict["marginals"] = fig_marginals

        ### pairs figures ###
        numCols = self.floatCols + self.intCols
        pairs_dict = all_reports_df["pairs_all"].to_dict()
        pairs_df = []
        for mod, dat in pairs_dict.items():
            dat["model"] = mod
            pairs_df.append(dat)
        pairs_df = pd.concat(pairs_df)
        pairs_df["pair"] = pairs_df["Column 1"] + "-" + pairs_df["Column 2"]
        mixed = pairs_df[
            ((pairs_df["Column 1"].isin(numCols)) & ((pairs_df["Column 2"].isin(self.catCols)) ))
              | 
              ((pairs_df["Column 2"].isin(numCols)) & ((pairs_df["Column 1"].isin(self.catCols)) ))
        ].drop(
            ["Real Correlation", "Synthetic Correlation", "Column 1", "Column 2"],
            axis=1,
        )
        cats = pairs_df[
            (pairs_df["Column 1"].isin(self.catCols))
            & (pairs_df["Column 2"]).isin(self.catCols)
        ].drop(
            ["Real Correlation", "Synthetic Correlation", "Column 1", "Column 2"],
            axis=1,
        )
        CorrS = pairs_df[pairs_df.Metric == "CorrelationSimilarity"].drop(
            ["Metric"], axis=1
        )
        ContS = cats[cats.Metric == "ContingencySimilarity"].drop(["Metric"], axis=1)
        # Mixed = mixed[mixed.Metric == "ContingencySimilarity"].drop(["Metric"], axis=1)

        ### pairs cont figure ###
        fig_pairs_cont = px.bar(
            ContS,
            x="pair",
            y="Quality Score",
            color="model",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="{}: Pairs Contigency Similarity Scores".format(self.dataset),
            labels={
                "Quality Score": "Contigency Similarity Score",
                "Column": "Pair",
            },
        )
        fig_pairs_cont.update_layout(legend_title="model")
        summary_figs_dict["pairs_cont"] = fig_pairs_cont

        ### pairs mixed figure ###
        fig_pairs_mixed = px.bar(
            mixed,
            x="pair",
            y="Quality Score",
            color="model",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="{}: Mixed Contigency Similarity Scores".format(self.dataset),
            labels={
                "Quality Score": "Contigency Similarity Score",
                "Column": "Pair",
            },
        )
        fig_pairs_mixed.update_layout(legend_title="model")

        summary_figs_dict["pairs_mixed"] = fig_pairs_mixed

        ### pairs corr figure ###
        fig_pairs_corr = px.bar(
            CorrS,
            x="pair",
            y="Quality Score",
            color="model",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="{}: Pairs Correlation Similarity Scores".format(self.dataset),
            labels={
                "Quality Score": "Correlation Similarity Score",
                "Column": "Pair",
            },
        )
        fig_pairs_corr.update_layout(legend_title="model")

        for mod in range(len(name_order_list)):
            for i, k in enumerate(fig_pairs_corr.data[mod]["x"]):
                if k in self.col_tokens_all:
                    fig_pairs_corr.data[mod]["x"][i] = "{}:{}".format(
                        k, self.col_tokens_all[k]
                    )

        summary_figs_dict["pairs_corr"] = fig_pairs_corr
        
        
        ks_mixed_df = pd.DataFrame(columns = ['pair','score','model'])
        for mod in name_order_list:
            tmp_dict = {}
            for k in self.scores_dict[mod]['pairs_ks_mixed_all'].keys():
                tmp = self.scores_dict[mod]['pairs_ks_mixed_all'][k]
                tmp_dict[k] = 1-np.mean([t if t!='empty' else 1 for t in tmp])
            tmp_df = pd.DataFrame(tmp_dict.items(),columns=['pair','score'])
            tmp_df['model'] = mod
            ks_mixed_df = pd.concat([ks_mixed_df,tmp_df])
        ### pairs ks mixed figure ###
        fig_pairs_mixed_ks = px.bar(
            ks_mixed_df,
            x="pair",
            y="score",
            color="model",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="{}: Pairs mixed KS scores".format(self.dataset),
            labels={
                "score": "Complement of mean KS deviations",
                "Column": "Pair",
            },
        )
        summary_figs_dict["pairs_mixed_ks"] = fig_pairs_mixed_ks
        
        return summary_figs_dict

    def plot_marginals(self, df_real, synths_dict, name_order_list):
        all_reports_df = pd.DataFrame(
            [self.scores_dict[mod] for mod in name_order_list],
            columns=self.scores_dict["CardiCat"].keys(),
            index=name_order_list,
        )
        models = self.synths_dict.keys()
        models = [y for x in name_order_list for y in models if y == x]
        # palette_tri = ["#b2011b", "#0173B2", "#DE8F05", "#D55E00"]
        # Xhat = lst_benchmark_dfs[0]

        nrows = len(df_real.columns)

        max_disp_cols = 20

        marginal_scores_df = []
        for mod in models:  # ["cCardiCat", "CardiCat", "tVAE", "tGAN"]:
            tmp = all_reports_df["marginal_all"][mod]
            tmp["model"] = mod
            marginal_scores_df.append(tmp)
        marginal_scores_df = pd.concat(marginal_scores_df)

        fig, axs = plt.subplots(
            nrows=nrows, ncols=len(name_order_list) + 1, figsize=(25, nrows * 5)
        )
        plt.subplots_adjust(hspace=0.5)
        # loop through tickers and axes
        sns.set(font_scale=1)

        for en, col in enumerate(synths_dict["CardiCat"].columns):
            # filter df for ticker and plot on specified axes\
            if col in self.catCols:
                df_real[col].value_counts().rename("original").iloc[
                    :max_disp_cols
                ].plot.bar(ax=axs[en][0], logy=False, title="{}: original".format(col))
                for en_col, synth_name in enumerate(name_order_list):
                    score = marginal_scores_df[
                        (marginal_scores_df["Metric"] == "TVComplement")
                        & (marginal_scores_df["model"] == synth_name)
                        & (marginal_scores_df["Column"] == col)
                    ]["Quality Score"].to_numpy()[0]

                    pd.concat(
                        [
                            df_real[col].value_counts().rename("original"),
                            synths_dict[synth_name][col]
                            .value_counts()
                            .rename(synth_name),
                        ],
                        axis=1,
                    ).iloc[:max_disp_cols].plot.bar(
                        ax=axs[en][en_col + 1],
                        logy=False,
                        title="{} \nKSComplement: {} ".format(
                            synth_name, round(score, 2)
                        ),
                    )
                    axs[en][en_col + 1].legend(prop={"size": 7})

            else:
                try:
                    df_real[col].rename("original").plot.density(
                        ax=axs[en][0], title="{}: original".format(col)
                    )
                    for en_col, synth_name in enumerate(name_order_list):
                        score = marginal_scores_df[
                            (marginal_scores_df["Metric"] == "KSComplement")
                            & (marginal_scores_df["model"] == synth_name)
                            & (marginal_scores_df["Column"] == col)
                        ]["Quality Score"].to_numpy()[0]
                        pd.concat(
                            [
                                df_real[col].rename("original"),
                                synths_dict[synth_name][col].rename("synth_name"),
                            ],
                            axis=1,
                        ).plot.density(
                            ax=axs[en][en_col + 1],
                            color=["b", "c"],
                            title="{} \nTVComplement: {}".format(
                                synth_name, round(score, 2)
                            ),
                        )
                        axs[en][en_col + 1].legend(prop={"size": 7})

                except np.linalg.LinAlgError:
                    print(
                        "col: {}, 1-th leading minor of the array is not positive definite".format(
                            col
                        )
                    )
                    pass
        return fig

    def plot_marginal_probs(self, df_real, synths_dict, name_order_list):
        # palette_tri = ["#b2011b", "#0173B2", "#DE8F05", "#D55E00"]
        # Xhat = lst_benchmark_dfs[0]

        all_reports_df = pd.DataFrame(
            [self.scores_dict[mod] for mod in name_order_list],
            columns=self.scores_dict["CardiCat"].keys(),
            index=name_order_list,
        )

        nrows = len(self.catCols)
        max_disp_cols = 20

        marginal_scores_df = []
        for mod in name_order_list:
            tmp = all_reports_df["marginal_all"][mod]
            tmp["model"] = mod
            marginal_scores_df.append(tmp)
        marginal_scores_df = pd.concat(marginal_scores_df)

        fig, axs = plt.subplots(
            nrows=nrows, ncols=len(name_order_list) + 1, figsize=(20, nrows * 5)
        )
        plt.subplots_adjust(hspace=0.5)
        # loop through tickers and axes
        sns.set(font_scale=1)

        for en, col in enumerate(synths_dict["CardiCat"].columns):
            # filter df for ticker and plot on specified axes\
            if col in self.catCols:
                df_real[col].value_counts().rename("original").iloc[
                    :max_disp_cols
                ].plot.bar(ax=axs[en][0], logy=False, title="{}: original".format(col))
                for en_col, synth_name in enumerate(name_order_list):
                    score = marginal_scores_df[
                        (marginal_scores_df["Metric"] == "TVComplement")
                        & (marginal_scores_df["model"] == synth_name)
                        & (marginal_scores_df["Column"] == col)
                    ]["Quality Score"].to_numpy()[0]

                    pd.concat(
                        [
                            np.sqrt(
                                df_real[col].value_counts() / df_real[col].shape
                            ).rename("original"),
                            np.sqrt(
                                synths_dict[synth_name][col].value_counts()
                                / df_real[col].shape
                            ).rename(synth_name),
                        ],
                        axis=1,
                    ).plot.scatter(
                        x="original",
                        y=synth_name,
                        ax=axs[en][en_col + 1],
                        logy=False,
                        title="{} \nKSComplement: {} ".format(
                            synth_name, round(score, 2)
                        ),
                    )
                    axs[en][en_col + 1].legend(prop={"size": 7})
                    axs[en][en_col + 1].axline([0, 0], [1, 1], color="k")
        # fig.legend("")
        return fig
