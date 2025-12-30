"""
CardiCat 推論モジュール
学習済みモデルを使って新しいデータの潜在表現を抽出
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


class CardiCatEncoder:
    """学習済みCardiCatモデルを使った潜在表現抽出クラス"""
    
    def __init__(self, model_dir: str, dataset_name: str):
        """
        Args:
            model_dir: モデル保存ディレクトリのパス
            dataset_name: データセット名（例: "event_5158"）
        """
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        
        # モデルとメタデータを読み込み
        self._load_models()
        self._load_metadata()
        
    def _load_models(self):
        """学習済みモデルを読み込み"""
        self.encoder = tf.keras.models.load_model(
            os.path.join(self.model_dir, f"encoder_{self.dataset_name}.keras")
        )
        print(f"エンコーダ読み込み完了")
        
    def _load_metadata(self):
        """メタデータを読み込み"""
        # ラベルエンコーダ
        with open(os.path.join(self.model_dir, f"label_encoder_{self.dataset_name}.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # エンコーダ情報
        with open(os.path.join(self.model_dir, f"encoder_info_{self.dataset_name}.pkl"), "rb") as f:
            self.encoder_info = pickle.load(f)
        
        # パラメータ
        with open(os.path.join(self.model_dir, f"param_dict_{self.dataset_name}.pkl"), "rb") as f:
            self.param_dict = pickle.load(f)
        
        self.catCols = self.encoder_info["catCols"]
        self.numCols = self.encoder_info.get("numCols", [])
        self.intCols = self.encoder_info.get("intCols", [])
        
        print(f"メタデータ読み込み完了")
        print(f"  カテゴリカラム: {self.catCols}")
        print(f"  数値カラム: {self.numCols}")
    
    def preprocess(self, df: pd.DataFrame) -> dict:
        """
        新しいデータを前処理してモデル入力形式に変換
        
        Args:
            df: 入力データフレーム
            
        Returns:
            モデル入力用の辞書
        """
        df_processed = df.copy()
        
        # カテゴリカラムをラベルエンコード
        for col in self.catCols:
            if col in df_processed.columns:
                # 未知のラベルは最頻値で置換
                known_labels = set(self.label_encoder[col].classes_)
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in known_labels else self.label_encoder[col].classes_[0]
                )
                df_processed[col] = self.label_encoder[col].transform(df_processed[col])
        
        # 入力辞書を作成
        input_dict = {}
        for col in df_processed.columns:
            input_dict[col] = df_processed[col].values
        
        return input_dict
    
    def encode(self, df: pd.DataFrame, return_mean: bool = True) -> np.ndarray:
        """
        データを潜在表現に変換
        
        Args:
            df: 入力データフレーム
            return_mean: Trueなら平均のみ、Falseなら(mean, log_var)を返す
            
        Returns:
            潜在表現 (N, latent_dim) または (mean, log_var)のタプル
        """
        # 前処理
        input_dict = self.preprocess(df)
        
        # エンコーダで潜在表現を取得
        mean, log_var = self.encoder.predict(input_dict, verbose=0)
        
        if return_mean:
            return mean
        else:
            return mean, log_var
    
    def encode_with_sampling(self, df: pd.DataFrame) -> np.ndarray:
        """
        サンプリングを含む潜在表現を取得（VAEの標準的な方法）
        
        Args:
            df: 入力データフレーム
            
        Returns:
            サンプリングされた潜在表現 (N, latent_dim)
        """
        mean, log_var = self.encode(df, return_mean=False)
        
        # 再パラメータ化トリック
        epsilon = np.random.normal(size=mean.shape)
        z = mean + np.exp(0.5 * log_var) * epsilon
        
        return z
    
    def encode_to_dataframe(
        self, 
        df: pd.DataFrame, 
        prefix: str = "z",
        include_original_index: bool = True
    ) -> pd.DataFrame:
        """
        潜在表現をDataFrame形式で取得
        
        Args:
            df: 入力データフレーム
            prefix: カラム名のプレフィックス
            include_original_index: 元のインデックスを含めるか
            
        Returns:
            潜在表現のDataFrame
        """
        latents = self.encode(df)
        latent_dim = latents.shape[1]
        
        latent_df = pd.DataFrame(
            latents,
            columns=[f"{prefix}_{i}" for i in range(latent_dim)]
        )
        
        if include_original_index:
            latent_df["original_index"] = df.index.values
        
        return latent_df


def extract_latent_representations(
    model_dir: str,
    dataset_name: str,
    input_csv: str,
    output_csv: str,
    drop_columns: list = None
) -> pd.DataFrame:
    """
    CSVファイルから潜在表現を抽出して保存するユーティリティ関数
    
    Args:
        model_dir: モデル保存ディレクトリ
        dataset_name: データセット名
        input_csv: 入力CSVファイルパス
        output_csv: 出力CSVファイルパス
        drop_columns: 除外するカラムのリスト
        
    Returns:
        潜在表現のDataFrame
    """
    # モデル読み込み
    encoder = CardiCatEncoder(model_dir, dataset_name)
    
    # データ読み込み
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"入力データ: {df.shape}")
    
    # カラム除外
    if drop_columns:
        existing_cols = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=existing_cols)
        print(f"除外したカラム: {existing_cols}")
    
    # 潜在表現抽出
    latent_df = encoder.encode_to_dataframe(df)
    print(f"潜在表現: {latent_df.shape}")
    
    # 保存
    latent_df.to_csv(output_csv, index=False)
    print(f"保存完了: {output_csv}")
    
    return latent_df


# 使用例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CardiCat潜在表現抽出")
    parser.add_argument("--model_dir", required=True, help="モデルディレクトリ")
    parser.add_argument("--dataset", required=True, help="データセット名")
    parser.add_argument("--input", required=True, help="入力CSVファイル")
    parser.add_argument("--output", required=True, help="出力CSVファイル")
    parser.add_argument("--drop", nargs="*", default=[], help="除外カラム")
    
    args = parser.parse_args()
    
    extract_latent_representations(
        model_dir=args.model_dir,
        dataset_name=args.dataset,
        input_csv=args.input,
        output_csv=args.output,
        drop_columns=args.drop
    )
