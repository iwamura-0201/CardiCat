import os
import sys
from datetime import datetime
from time import process_time
import json
import yaml
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
import joblib
from types import SimpleNamespace
from pathlib import Path
import tensorflow as tf

# CardiCatパス設定
LIB_PATH = "/home/ubuntu/CardiCat/"
sys.path.insert(1, LIB_PATH)

from src import network as network
from src import postprocessing as postprocessing
from src import preprocessing as preprocessing
from src import reporting as reporting
from src import vae_model as vae

print(f"TensorFlow バージョン: {tf.__version__}")
print(f"GPU 利用可能: {tf.config.list_physical_devices('GPU')}")

DATA_DIR = Path("../eventid_content/concat/train")

class DataLoader:
    def __init__(
        self,
        event_id: str,
        model: str = "CardiCat", 
    ):
        config_filepath = Path(f"./config/{event_id}.yaml")
        self.event_id = event_id
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
            self.params = config.get("params", {}) # dictとしてロード
            self.drop_cols = config.get("drop_cols", []) # listとしてロード

        # モデルタイプに応じた設定
        if model == "VAE":
            self.params["embed_th"] = 5000
            self.params["emb_loss"] = False
        elif model == "eVAE":
            self.params["emb_loss"] = False
        elif model == "CardiCat":
            self.params["emb_loss"] = True

        # モデル設定
        self.params["model"] = model

        # ログ用コピー
        logs = self.params.copy()
        logs["model"] = model
        logs["event_id"] = self.event_id
        #logs["run_id"] = run_id
        logs["date_time"] = datetime.now().strftime("%Y%m%d_%H%M")

        print(f"モデル: {self.params['model']}")
        print(f"データセット: event_{self.event_id}")
        
        return

    def data_loader(self) -> None:

        # CSVファイル読み込み
        csv_path = DATA_DIR / f"event_{self.event_id}.csv"
        print(f"読み込み完了: {csv_path}")

        dataframe_full = pd.read_csv(csv_path, low_memory=False)
        print(f"データフレーム形状: {dataframe_full.shape}")
        print(f"カラム: {list(dataframe_full.columns)}")

        # 学習に使用しないカラムを除外
        DROP_COLUMNS = self.drop_cols
        existing_drop_cols = [col for col in DROP_COLUMNS if col in dataframe_full.columns]
        dataframe = dataframe_full.drop(columns=existing_drop_cols)

        # カーディナリティが1以下のカラムを除外
        low_card_cols = [col for col in dataframe.columns if dataframe[col].nunique() <= 1]
        print(f"低カーディナリティカラムを除外: {low_card_cols}")
        dataframe = dataframe.drop(columns=low_card_cols)

        print(f"学習用データフレーム形状: {dataframe.shape}")
        print(f"除外したカラム: {existing_drop_cols + low_card_cols}")

        # ターゲットカラムなし
        self.params["is_target"] = False

        # 訓練/テスト分割（評価用に元データを保持）
        dataframe_shuffled = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(self.params["train_ratio"] * len(dataframe_shuffled))

        dataframe_train = dataframe_shuffled.iloc[:split_idx].copy()
        dataframe_test = dataframe_shuffled.iloc[split_idx:].copy()

        print(f"訓練データサイズ: {dataframe_train.shape}")
        print(f"テストデータサイズ: {dataframe_test.shape}")

        # カーディナリティ確認
        preprocessing.get_df_cardinality(dataframe_train)  

        self.dataframe_train = dataframe_train
        self.dataframe_test = dataframe_test

        return

    def preprocess(self) -> None:
        # カラムタイプの検出
        catCols, intCols, floatCols = preprocessing.get_col_types(
            self.dataframe_train, is_y=self.params["is_target"], verbose=True
        )

        # 
        col_tokens_all_tmp = {}
        for i in catCols:
            col_tokens_all_tmp[i] = len(dataframe_train[i].unique())

        if param_dict["cVAE"]:
            tmpEmb = [
                key
                for key, value in col_tokens_all_tmp.items()
                if value >= param_dict["embed_th"]
            ]
        else:
            tmpEmb = []

        if param_dict["cVAE"]:
            # marg_sample = lambda e: np.random.choice(
            #     list(catMarginals[e].keys()), p=list(catMarginals[e].values())
            #
            alpha = 1  # (if <= alpha then x, else marg sample)

            if param_dict["cVAE_mask"]:
                for col in tmpEmb:
                    dataframe_train["cond_" + col] = "mask"
                for index, row in dataframe_train.iterrows():
                    # print(index,row)
                    col_sampled = np.random.choice(tmpEmb)
                    # df.loc[index, "cond_" + col_sampled] = preprocessing.marg_sample(
                    #     col_sampled, catMarginals
                    # )

                    # df.loc[index, "cond_" + col_sampled] = df.loc[index, col_sampled]
                    dataframe_train.loc[index, "cond_" + col_sampled] = (
                        dataframe_train.loc[index, col_sampled]
                        if np.random.rand() <= alpha
                        else preprocessing.marg_sample(col_sampled, catMarginals)
                    )
            else:
                for col in tmpEmb:
                    dataframe_train["cond_" + col] = dataframe_train[col].apply(
                        lambda x: x
                        if np.random.rand() <= alpha
                        else preprocessing.marg_sample(col, catMarginals)
                    )  # df[col].copy()
        print(dataframe_train.head())

        # ラベルエンコーディング（重要なステップ！）
        df_encoded, label_encoder = preprocessing.labelEncoding(
            dataframe_train.copy(),
            catCols,
        )

        print(f"エンコード後データフレーム形状: {df_encoded.shape}")
        print(f"ラベルエンコーダ作成対象: {list(label_encoder.keys())}")

        # TensorFlowデータセットの作成
        train, test = np.split(
                df_encoded.sample(frac=1), [int(param_dict["val_ratio"] * len(df_encoded))]
            )
        print("Total length: {:,}".format(len(df_encoded)))
        print(
            "length of train: {:,} which is {:.1%}".format(
                len(train), param_dict["train_ratio"]
            )
        )
        print(
            "length of test: {:,} which is {:.1%}".format(
                len(test), 1 - param_dict["train_ratio"]
            )
        )
        train_ds = preprocessing.df_to_dset(
            train, is_y=param_dict["is_target"], batch_size=param_dict["batch_size"]
        )
        test_ds = preprocessing.df_to_dset(
            test, is_y=param_dict["is_target"], batch_size=param_dict["batch_size"]
        )
        train_ds.element_spec

        print("訓練データセット作成完了")
        print(f"要素仕様: {train_ds.element_spec}")

        self.train_ds = train_ds
        self.test_ds = test_ds
        return


    def define_network(self) -> SomeType:
        # データエンコーダ作成
        self.data_encoder = preprocessing.data_encoder(
            original_df=self.dataframe_train,
            train_ds=self.train_ds,
            param_dict=self.params
        )

        #print(f"カテゴリカラム: {self.data_encoder.catCols}")
        #print(f"カテゴリトークン数: {self.data_encoder.col_tokens_all}")
        #print(f"総カーディナリティ: {sum(self.data_encoder.col_tokens_all.values()):,}")
        #print(f"エンベディングサイズ: {self.data_encoder.emb_sizes_all}")
        #print(f"埋め込みカラム: {self.data_encoder.embCols}")
        #print(f"One-Hotカラム: {self.data_encoder.ohCols}")

        # エンコーディングレイヤーの取得
        (
            all_inputs,
            all_features,
            all_inputs_1hot,
            all_features_1hot,
            condInputs,
            condFeatures,
            numFeatures,
            numLookup,
            all_features_cod,
        ) = self.data_encoder.get_encoding_layers()

        # data_encoderに保存（後で使用）
        self.data_encoder.all_inputs = all_inputs
        self.data_encoder.all_features = all_features
        self.data_encoder.all_inputs_1hot = all_inputs_1hot
        self.data_encoder.all_features_1hot = all_features_1hot
        self.data_encoder.condInputs = condInputs
        self.data_encoder.condFeatures = condFeatures
        self.data_encoder.numFeatures = numFeatures
        self.data_encoder.numLookup = numLookup
        self.data_encoder.all_features_cod = all_features_cod

        #print(f"入力レイヤー数: {len(all_inputs)}")
        #print("エンコーディングレイヤー構築成功")

        # オプティマイザ
        optimizer = tf.keras.optimizers.Adam(learning_rate=param_dict["learn_rate"])

        # サンプリングレイヤー
        final = vae.sampling(param_dict["latent_dim"], param_dict["latent_dim"])
        input_decoder = param_dict["latent_dim"]

        # エンコーダ構築
        dec_inputs = tf.keras.Input(shape=(input_decoder,), name="dec_latent_input")
        enc = network.encoder(
            data_encoder.all_inputs, 
            data_encoder.all_features, 
            param_dict["latent_dim"]
        )

        # デコーダ構築
        dec = network.decoder(dec_inputs, dec_inputs, data_encoder.layer_sizes)

        # コーデックス構築（エンベディング損失用）
        if param_dict["emb_loss"]:
            cod = vae.codex(data_encoder.all_inputs, data_encoder.all_features_cod)
        else:
            cod = vae.codex(data_encoder.all_inputs_1hot, data_encoder.all_features_1hot)

        #print("ネットワーク構築成功")

        # 学習可能パラメータ数の取得
        trainableParamsEnc = np.sum([np.prod(v.shape) for v in enc.trainable_weights])
        trainableParamsDec = np.sum([np.prod(v.shape) for v in dec.trainable_weights])
        logs["trainable_params"] = f"{int(trainableParamsEnc + trainableParamsDec):,}"

        # エンコーダのサマリーを返す
        return enc.summary()


    def train(self) -> None:
        # エンベディング重みの初期分散を取得
        emb_weights = {}
        for layer in self.enc.layers:
            if layer.name.startswith("emb_") and hasattr(layer, "trainable_weights") and layer.trainable_weights:
                emb_weights[layer.name] = layer.trainable_weights[0].numpy()

        emb_init_var = [
            tf.math.reduce_mean(tf.math.reduce_variance(emb_weights[emb], axis=0))
            for emb in emb_weights.keys()
        ] if emb_weights else []

        # cVAE用（今回は未使用）
        # tmpEmb = []

        # 学習実行！
        print(f"\n### {model} の学習開始 ###")
        CardiCat_start = process_time()

        tf.config.run_functions_eagerly(True)

        output_loss, loss_logs_all, emb_weights = vae.train_high_vae(
            train_ds=train_ds,
            enc=enc,
            dec=dec,
            cod=cod,
            final=final,
            optimizer=optimizer,
            layer_sizes=data_encoder.layer_sizes,
            param_dict=param_dict,
            weights=data_encoder.weights,
            emb_init_var=emb_init_var,
            tmpEmb=tmpEmb,
        )

        CardiCat_stop = process_time()
        CardiCat_time = round(CardiCat_stop - CardiCat_start, 2)
        print(f"\n学習完了: {CardiCat_time} 秒")
        logs["losses"] = round(output_loss.iloc[-1, 1:], 2)

        # 損失のプロット
        df_melt = output_loss[
            ["epoch", "total_loss", "mse_loss_factor", "kl_loss", "emb_reg_loss"]
        ].melt(id_vars=["epoch"], var_name="loss_type", value_name="loss")

        fig, ax = plt.subplots(figsize=(12, 6))
        df_melt.set_index(["loss_type", "epoch"]).unstack("loss_type")["loss"].plot(ax=ax)
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.set_title("CardiCat loss transition")
        plt.show()

        # ----------------------------モデルの保存--------------------------- #
        model_dir = os.path.join(LIB_PATH, "output/models", dataset)
        os.makedirs(model_dir, exist_ok=True)

        # エンコーダとデコーダを保存
        enc.save(os.path.join(model_dir, f"encoder.keras"))
        dec.save(os.path.join(model_dir, f"decoder.keras"))
        cod.save(os.path.join(model_dir, f"codex.keras"))

        # ラベルエンコーダを保存（推論時に必要）
        with open(os.path.join(model_dir, f"label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

        # data_encoderの情報も保存
        encoder_info = {
            "catCols": data_encoder.catCols,
            "intCols": data_encoder.intCols,
            "floatCols": data_encoder.floatCols,
            "embCols": data_encoder.embCols,
            "ohCols": data_encoder.ohCols,
            "numCols": data_encoder.numCols,
            "col_tokens_all": data_encoder.col_tokens_all,
            "layer_sizes": data_encoder.layer_sizes,
            "emb_sizes_all": data_encoder.emb_sizes_all,  
            "oh_tokens_all": getattr(data_encoder, 'oh_tokens_all', {}), 
        }
        with open(os.path.join(model_dir, f"encoder_info.pkl"), "wb") as f:
            pickle.dump(encoder_info, f)

        # パラメータも保存
        with open(os.path.join(model_dir, f"params.pkl"), "wb") as f:
            pickle.dump(param_dict, f)

        print(f"モデル保存完了: {model_dir}")
        print(f"保存ファイル:")
        for f in os.listdir(model_dir):
            print(f"  - {f}")

        return

    def test(self) -> None:
        # ---------------------------- モデルの読み込み --------------------------- #
        # 1. 読み込むモデルディレクトリを指定 (最新のものを自動取得する場合)
        model_dir = os.path.join(LIB_PATH, param_dict["output_path"], "models", f"{self.event_id}")

        # 2. Keras モデルの読み込み
        enc = tf.keras.models.load_model(os.path.join(model_dir, "encoder.keras"))
        dec = tf.keras.models.load_model(os.path.join(model_dir, "decoder.keras"))
        cod = tf.keras.models.load_model(os.path.join(model_dir, "codex.keras"))

        # 3. LabelEncoder の読み込み
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

        # 4. パラメータの読み込みと復元 (param_dict, logs)
        saved_params = joblib.load(os.path.join(model_dir, "params.pkl"))
        param_dict.update(saved_params)
        logs.update(saved_params)

        # 5. DataEncoder 情報の復元
        de_info = joblib.load(os.path.join(model_dir, "encoder_info.pkl"))
        # data_encoder オブジェクトに必要な属性を流し込む
        class Dummy:
            pass
        data_encoder = Dummy()
        for k, v in de_info.items():
            setattr(data_encoder, k, v)
        # 特殊なレイヤーはモデルから再取得
        data_encoder.all_inputs = enc.inputs
        data_encoder.numLookup = [
            layer for layer in enc.layers
            if "Normalization" in layer.__class__.__name__
        ]
        data_encoder.numFeatures = [
            layer.output for layer in data_encoder.numLookup
        ]

        # その他必要な変数の復元
        catCols = de_info.get("catCols", [])
        numCols = de_info.get("numCols", [])
        intCols = de_info.get("intCols", [])
        embCols = de_info.get("embCols", [])
        emb_sizes = list(de_info.get("emb_sizes_all", {}).values())
        oh_tokens = list(de_info.get("oh_tokens_all", {}).values())
        col_tokens_all = de_info.get("col_tokens_all", {})

        print("モデルと設定の読み込みが正常に完了しました。")

        # ---------------------------- 予測データの生成 --------------------------- #
        # 訓練セットから合成データを生成
        mean, log_var = enc.predict(train_ds)
        latents = final([mean, log_var])

        df_y_dict = latents

        # 予測データの取得
        gen_df_train, emb_weights = postprocessing.get_pred(
            enc,
            dec,
            df_y_dict,
            param_dict,
            oh_tokens,
            emb_sizes,
            embCols,
            data_encoder.numFeatures,
            catCols,
            numCols,
            data_encoder.numLookup,
            intCols,
            data_encoder.all_inputs,
            list(col_tokens_all.values()),
            label_encoder,
        )

        print(f"生成された合成データ形状: {gen_df_train.shape}")
        gen_df_train.head()

        # ---------------------------- SDVレポート --------------------------- #
        report = postprocessing.get_report(
            dataframe_test,
            gen_df_train,
            param_dict,
            full=True
        )

        print("=== 品質レポートサマリー ===")
        print(report["summary"])

        # ---------------------------- 最終サマリー --------------------------- #
        summary = {
            "モデル": model,
            "データセット": dataset,
            "学習時間(秒)": CardiCat_time,
            "エポック数": param_dict["epochs"],
            "訓練サンプル数": len(dataframe_train),
            "テストサンプル数": len(dataframe_test),
            "学習可能パラメータ": logs["trainable_params"],
            "総合スコア": round(report["summary"].Score.mean(), 2),
        }

        print("\n=== 学習サマリー ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")