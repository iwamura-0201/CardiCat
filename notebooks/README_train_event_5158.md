# CardiCat モデル学習手順書 (event_5158.csv)

## 概要

`concat/train/event_5158.csv` に対してCardiCatモデルの学習を行う手順です。
参考: `notebooks/CardiCat_ICLR25.ipynb`

---

## 実行すべき動作の流れ

### 1. ライブラリのインポート
```python
import tensorflow as tf
from src import network, postprocessing, preprocessing, reporting, vae_model
```

主要な依存ライブラリ:
- TensorFlow
- pandas, numpy
- sdv (Synthetic Data Vault) - 評価用

---

### 2. パラメータ設定

```python
param_dict = {
    "embed_th": 3,           # エンベディング閾値
    "epochs": 100,           # エポック数
    "batch_size": 512,       # バッチサイズ
    "learn_rate": 0.0005,    # 学習率
    "latent_dim": 15,        # 潜在次元数
    "recon_factor": 5,       # 再構成損失の重み
    "kl_factor": 1,          # KLダイバージェンスの重み
    "emb_loss": True,        # CardiCatの場合True
    ...
}
```

**モデルタイプ:**
- `CardiCat`: エンベディング損失あり（推奨）
- `VAE`: 通常のVAE（エンベディングなし）
- `eVAE`: エンコーダーのみエンベディング

---

### 3. データの読み込みと前処理

```python
# CSVファイル読み込み
dataframe = pd.read_csv("eventid_content/concat/train/event_5158.csv")

# 学習に不要なカラムを除外
DROP_COLUMNS = ["EventID", "Timestamp", "Label"]
dataframe = dataframe.drop(columns=DROP_COLUMNS)

# 訓練/テスト分割（7:3）
dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.3)

# カラムタイプの検出
catCols, intCols, floatCols = preprocessing.get_col_types(dataframe_train)
```

---

### 4. データエンコーダの作成

```python
# TensorFlow Datasetに変換
train_ds = tf.data.Dataset.from_tensor_slices(dict(dataframe_train)).batch(batch_size)

# データエンコーダ作成
data_encoder = preprocessing.data_encoder(dataframe_train, train_ds, param_dict)

# エンコーディングレイヤーの取得
encoding_layers = data_encoder.get_encoding_layers()
```

**data_encoderが行うこと:**
- カテゴリ変数・数値変数の検出
- カーディナリティに応じたエンベディング対象の決定
- 各変数のエンコーディングレイヤー（One-Hot、エンベディング、正規化）の作成

---

### 5. VAEモデルの構築と学習

```python
# エンコーダーネットワーク構築
encoder = network.get_encoder(all_inputs, all_features, latent_dim)

# VAEモデル構築
vae_model = vae.VAE(encoder=encoder, data_encoder=data_encoder, param_dict=param_dict)

# コンパイル
vae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

# 学習
history = vae_model.fit(train_ds, epochs=epochs)
```

---

### 6. 合成データの生成

```python
# サンプリング
synthetic_data = vae_model.sample(n_samples)

# 後処理（デコード）
synthetic_df = postprocessing.decode_synthetic(synthetic_data, data_encoder, dataframe_train)
```

---

### 7. 評価レポートの作成

```python
# SDVレポート生成
report = postprocessing.get_report(dataframe_test, synthetic_df, param_dict, full=True)

# 主要な評価指標
# - marginals: 各変数の周辺分布の品質
# - pairs: 変数ペアの相関の品質
# - KSComplement, TVComplement: 分布の類似度
```

---

## テストノートブック

`notebooks/train_event_5158.ipynb` に上記の手順をすべて含むテストスクリプトを用意しました。

---

## 注意事項

1. **GPUメモリ**: 大きなデータセットの場合、batch_sizeを調整
2. **カーディナリティ**: `embed_th` を調整してエンベディング対象を制御
3. **エポック数**: 損失が収束するまで調整（early stoppingも検討）
4. **除外カラム**: EventID, Timestamp, Labelは学習から除外

---

## ファイル構造

```
CardiCat/
├── eventid_content/
│   └── concat/
│       └── train/
│           └── event_5158.csv  # 入力データ
├── notebooks/
│   ├── CardiCat_ICLR25.ipynb   # 参考ノートブック
│   └── train_event_5158.ipynb  # テストスクリプト
├── src/
│   ├── preprocessing.py        # 前処理
│   ├── network.py              # ネットワーク定義
│   ├── vae_model.py            # VAEモデル
│   ├── postprocessing.py       # 後処理・評価
│   └── reporting.py            # レポート生成
└── params_dict.txt             # パラメータ設定ファイル
```
