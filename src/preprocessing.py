#########################
#### CardiCat ###########
#########################


import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Attention, Flatten, Dense  # noqa: F401
from keras.layers import CategoryEncoding, IntegerLookup, Normalization, StringLookup
from sklearn.preprocessing import LabelEncoder

from src.utils import flatten_list


def get_catMarginals(dataframe, catCols):
    """Returns a dict of the marginal distribution (probabilities) for each of
        the categorical columns.

    Args:
        dataframe (pandas.DataFrame): A dataframe containing categorical columns
        catCols (list): a list containing the names of the categorical columns

    Returns:
        dict(): _description_
    """
    catMarginals = {}
    denom = dataframe.shape[0]
    for cat in catCols:
        catMarginals[cat] = (dataframe[cat].value_counts() / denom).to_dict()
    return catMarginals


class data_encoder:
    """Provides all the objects necessary to encode (preprocessing) the data."""

    def __init__(self, original_df, train_ds, param_dict):
        self.original_df = original_df
        self.train_ds = train_ds
        self.is_y = param_dict["is_target"]
        self.embed_th = param_dict["embed_th"]
        self.attention = param_dict["attention"]
        self.catCols, self.intCols, self.floatCols = self.get_col_types()
        # Getting a list of numerical columns based on data types:
        self.numCols = self.floatCols + self.intCols
        # the cardinality of all categorical features:
        self.col_tokens_all = self.get_cat_tokens(self.original_df, self.catCols)
        # suggested emb sizes of all categorical features:
        self.emb_sizes_all = self.get_emb_sizes(self.col_tokens_all)
        # According to emb_sizes, get embCols and ohCols by threshold:
        self.embCols_list = [
            key for key, value in self.col_tokens_all.items() if value >= self.embed_th
        ]
        self.ohCols_list = [
            key
            for key, value in self.col_tokens_all.items()
            if value < param_dict["embed_th"]
        ]
        # Get dict of emb/oh columns and their emb/one hot sizes:
        self.embCols = {k: self.emb_sizes_all[k] for k in self.embCols_list}
        self.ohCols = {k: self.col_tokens_all[k] for k in self.ohCols_list}
        tmp = self.get_encoding_layers()

        (
            self.all_inputs,
            self.all_features,
            self.all_inputs_1hot,
            self.all_features_1hot,
            self.condInputs,
            self.condFeatures,
            self.numFeatures,
            self.numLookup,
            self.all_features_cod,
        ) = tmp

        if param_dict["emb_loss"]:
            self.layer_sizes = (
                [[*self.ohCols.values()]]
                + [[*self.embCols.values()]]
                + [[1] * len(self.numCols)]
            )
        else:
            self.layer_sizes = (
                [[*self.col_tokens_all.values()]] + [[]] + [[1] * len(self.numCols)]
            )

        self.layer_sizes_dict = {
            k: v
            for k, v in zip(
                [*self.ohCols.keys()] + [*self.embCols.keys()] + self.numCols,
                flatten_list(self.layer_sizes),
            )
        }

        # These can/should be passed by user.... NOTE
        self.weights = flatten_list(
            [
                flatten_list([i * [1.0 / i] for i in layer])
                for layer in self.layer_sizes
                # flatten_list([i * [1.0 / i] for i in self.layer_sizes[0]]),
                # flatten_list([i * [1.0 / i] for i in self.layer_sizes[1]]),
                # flatten_list([i * [1.0 / i] for i in self.layer_sizes[2]]),
            ]
        )

    def get_col_types(self):
        """
        Given the dtypes of a pandas dataframe (original_df), returns two list, one
        with col names of numerical (int/float) columns, and one for
        categorical (object) columns

        Args:
            df (pandas.Dataframe): a pandas dataframe with column names
            is_y (list,list): a list of coloumn names for each type


        Returns:
            (list,list):two list, one
            with col names of numerical (int/float) columns, and one for
            categorical (object) columns
        """

        if self.is_y:
            colTypes = self.original_df.drop("target", axis=1).dtypes.to_dict()
        else:
            colTypes = self.original_df.dtypes.to_dict()
        catCols = []
        floatCols = []
        intCols = []
        for i in colTypes.items():
            if isinstance(i[1], np.dtypes.ObjectDType):
                # if i[1] is object:
                catCols.append(i[0])
            elif isinstance(i[1], (int, np.dtypes.Int64DType)):
                # elif i[1] is int or i[1] is pd.Int64Dtype():
                intCols.append(i[0])
            elif isinstance(i[1], (float, np.dtypes.Float64DType)):
                # elif i[1] == float:
                floatCols.append(i[0])
            else:
                print("Print-unused type: ", i, i[0])
        return catCols, intCols, floatCols

    def get_cat_tokens(self, df, catCols):
        """Returns a dict of {catCol:numberUniques},
           the unique values (cardinality) of all the catCols in df.

        Args:
            df (pandas.DataFrame): the tabluar data
            catCols (list): a list of all categorical columns

        Returns:
            dict: the dictionary of categorical columns and associated cardinality
        """
        col_tokens_all = {}
        for i in catCols:
            col_tokens_all[i] = len(df[i].unique())
        return col_tokens_all

    # def get_emb_sizes(self, col_tokens_all):
    #     """Assigns a recommended embedding size for each categorical column.

    #     Args:
    #         col_tokens_all (dict): the dictionary of categorical columns and
    #                                associated cardinality

    #     Returns:
    #         dic: the suggested embedding size for each categorical column
    #     """
    #     emb_sizes = {}
    #     for key, value in col_tokens_all.items():
    #         emb_sizes[key] = max(3, int(value ** (1 / 4))) if value > 2 else 2
    #     return emb_sizes
    
    def get_emb_sizes(self,col_tokens_all):
        """Assigns a recommended embedding size for each categorical variable. 
        """
        emb_sizes = {}
        for key,value in col_tokens_all.items():
            if value <=1:
                raise ValueError('Categorical token size is less than 2!')
            elif value ==2:
                emb_sizes[key]=2
            elif value <=20:
                emb_sizes[key]=3
            elif value >20 and value<=100:
                emb_sizes[key]=4
            elif value >100 and value<500:
                emb_sizes[key]=5
            elif value >500 and value<1000:
                emb_sizes[key]=6
            elif value >1000 and value<2000:
                emb_sizes[key]=7
            else:
                emb_sizes[key]=8
        return emb_sizes

    def get_encoding_layers(self):
        """Builds the encoding layers primitives (tf), and outputs two versions of
            inputs and features layers of the model : one with embedding layers,
            and one without (replaced by one-hot). Embedded columns are determined
            by the embedding threshold.

        Returns:
            list(KerasTensors): the possible encoded layers (normalization, one-hot,
                                embedding to choose from.)
        """
        ## 1-hot if users do not want emb-loss in decoder:

        # FIRST, one-hot encode all categorical variables no matter their designation:
        ohInputs_all_temp = [
            tf.keras.Input(shape=(1,), name=cat, dtype=tf.int64) for cat in self.catCols
        ]
        ohFeatures_all_temp = [
            get_1hot_encoding_layer(
                name=name,
                # dataset=self.train_ds,
                # dtype="int64",
                # is_y=self.is_y,
                max_tokens=tok,
            )(inpt)
            for name, inpt, tok in zip(
                self.catCols, ohInputs_all_temp, self.col_tokens_all.values()
            )
        ]
        # SECONDLY, set designated one-hot inputs and features
        ohInputs = [
            i
            for i, f in zip(ohInputs_all_temp, ohFeatures_all_temp)
            if f.shape[1] < self.embed_th
        ]
        ohFeatures = [f for f in ohFeatures_all_temp if f.shape[1] < self.embed_th]
        # Thirdly, Override emb columns as one-hot INSTEAD, just in case we don't want to embed:
        ohEmbInputs = [
            i
            for i, f in zip(ohInputs_all_temp, ohFeatures_all_temp)
            if f.shape[1] >= self.embed_th
        ]
        ohEmbFeatures = [f for f in ohFeatures_all_temp if f.shape[1] >= self.embed_th]
        # set embeddings inputs and features
        tmp_emb_out = [
            encode_entity_emb(
                cat_name, self.original_df[cat_name].nunique(), embed_size=emb_size
            )
            for cat_name, emb_size in self.embCols.items()
        ]
        tmp_cond_out = [
            encode_cond_emb(
                cat_name, self.original_df[cat_name].nunique(), embed_size=emb_size
            )
            for cat_name, emb_size in self.embCols.items()
        ]

        ### code to split it into 3 lists
        if tmp_emb_out:
            embInputs, embFeatures, embEmbeddings = map(list, zip(*tmp_emb_out))
            condInputs, condFeatures = map(list, zip(*tmp_cond_out))

        else:
            embInputs, embFeatures, embEmbeddings, condInputs, condFeatures = (
                [],
                [],
                [],
                [],
                [],
            )

        # print(embFeatures)
        # Overriding embedings (if user wants it, outputs regardless):
        ohInputs_all = ohInputs + ohEmbInputs  #
        ohFeatures_all = ohFeatures + ohEmbFeatures  #

        ## Numerical encoding (Normalization):
        numInputs = [tf.keras.Input(shape=(1,), name=num) for num in self.numCols]
        ### creating encoded_feature for categorical string variables:
        if self.numCols:
            numFeatures = [
                encode_numerical_feature(numFeat, num, self.train_ds, is_y=self.is_y)
                for num, numFeat in zip(self.numCols, numInputs)
            ]
            (
                numFeatures,
                numLookup,
            ) = map(list, zip(*numFeatures))
        else:
            numFeatures = []
            numLookup = False

        ## EXPERIMENTAL:
        ## TRYING ATTENTION
        print(embFeatures)
        if self.attention:
            embFeatures_conc = tf.keras.layers.concatenate(embFeatures)
            embFeats_attn = Attention(use_scale=False,dropout=0.0)([embFeatures_conc, embFeatures_conc])
            # embFeats_attn = Flatten()(embFeats_attn) # not needed...?
            embFeats_attn = Dense(units=sum(self.embCols.values()), activation="relu")(
                embFeats_attn
            )
            all_features = tf.keras.layers.concatenate(
                ohFeatures + [embFeats_attn] + numFeatures
            )
        else:
            all_features = tf.keras.layers.concatenate(
                ohFeatures + embFeatures + numFeatures
            )

        ##  Applying Encoders:
        # Entity Embedding Categorical Features
        # combining the two together:

        all_features_cod = tf.keras.layers.concatenate(
            ohFeatures + embFeatures + numFeatures
            # ohFeatures + [embAttenDense] + numFeatures
        )
        all_inputs = ohInputs + embInputs + numInputs

        # If we don't want to use embedding as input layers, only one-hot:
        ## This is called nonetheless, but might not be used.
        all_inputs_1hot = ohInputs_all + numInputs
        all_features_1hot = tf.keras.layers.concatenate(ohFeatures_all + numFeatures)

        return (
            all_inputs,
            all_features,
            all_inputs_1hot,
            all_features_1hot,
            condInputs,
            condFeatures,
            numFeatures,
            numLookup,
            all_features_cod,
        )


def encode_entity_emb(cat_name, vocab_size, embed_size=5):
    """Returns an entity embedding layer per category (cat_name)  of a dataset

    Args:
        cat_name (string): column name of category to be entity-encoded
        vocab_size (int): the size of categorical column unique values
        dataset (tf.dataset): tensorflow dataset
        embed_size (int): size of embedding

    Returns:
        (inpt,embed_rehsaped) (tuple): tf input layer,embedding layer reshaped
    """
    inpt = tf.keras.layers.Input(shape=(1,), name="_".join(cat_name.split(" ")))
    embed = tf.keras.layers.Embedding(
        vocab_size,
        embed_size,
        trainable=True,
        embeddings_initializer=tf.initializers
        # .RandomUniform(minval=-1, maxval=1),\
        .RandomNormal(mean=0.0, stddev=1.0),
        #   embeddings_regularizer=tf.keras.regularizers.L2, # worse...
        # embeddings_regularizer=tf.keras.regularizers.OrthogonalRegularizer(mode='columns'),
        name="emb_{}".format(cat_name),
    )(inpt)
    # embed = tf.keras.layers.BatchNormalization(axis=1 )(embed)
    embed_rehsaped = tf.keras.layers.Reshape(
        target_shape=(embed_size,),
        name="embed_reshape_" + "_".join(cat_name.split(" ")),
    )(embed)
    return inpt, embed_rehsaped, embed


def encode_cond_emb(cat_name, vocab_size, embed_size=5):
    """Returns an entity embedding layer per category (cat_name)  of a dataset

    Args:
        cat_name (string): column name of category to be entity-encoded
        vocab_size (int): the size of categorical column unique values
        dataset (tf.dataset): tensorflow dataset
        embed_size (int): size of embedding

    Returns:
        (inpt,embed_rehsaped) (tuple): tf input layer,embedding layer reshaped
    """
    inpt = tf.keras.layers.Input(
        shape=(1,), name="cond_" + "_".join(cat_name.split(" "))
    )
    dense = tf.keras.layers.Embedding(
        vocab_size, embed_size, trainable=False, name="cond_dense_{}".format(cat_name)
    )(inpt)
    # embed = tf.keras.layers.BatchNormalization(axis=1 )(embed)
    dense_rehsaped = tf.keras.layers.Reshape(
        target_shape=(embed_size,),
        name="cond_dense_reshape_" + "_".join(cat_name.split(" ")),
    )(dense)
    return inpt, dense_rehsaped


def encode_numerical_feature(num_feature, name, dataset, is_y):
    """Encode a numerical feature using a Keras normalization layer

    Args:
        num_feature (keras input layer): the respective Keras input layer
        name (string): column name of numerical features to be normalized
        dataset (tf.dataset): tensorflow dataset
        is_y (bool): if the dataset has a target column (label)

    Returns:
        encoded_feature (Keras layer): keras normazliation layer
    """
    # Create a Normalization layer for our feature
    normalizer = Normalization(name=name + "_norm")

    # Prepare a Dataset that only yields our feature
    if is_y:
        feature_ds = dataset.map(lambda x, y: x[name])
    else:
        feature_ds = dataset.map(lambda x: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(num_feature)
    return encoded_feature, normalizer


def encode_categorical_feature(feature, cat_name, dataset, is_string, is_y=False):
    """Returns a 1-hot encoding layer per category (name)  of a dataset

    Args:
        feature (keras input layer): the respective Keras input layer
        cat_name (string): column name of category to be 1-hot-encoded
        dataset (tf.dataset): tensorflow dataset
        is_string (boolean): if the values are strings (vs. ordinal int)
        is_y (bool, optional):  if the dataset has a target column (label)

    Returns:
        lookup/ lookup(feature) (tuple): tf 1-hot lookup layer/ inverse lookup layer
    """
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="one_hot", num_oov_indices=0)
    lookup_inv = lookup_class(invert=True)

    # Prepare a Dataset that only yields our feature
    if is_y:
        feature_ds = dataset.map(lambda x, y: x[cat_name])
    else:
        feature_ds = dataset.map(lambda x: x[cat_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    # encoded_feature = lookup(feature)
    return lookup_inv, lookup(feature)


def get_1hot_encoding_layer(name, max_tokens=None):  # dataset, dtype, is_y=False,
    """Using CategoryEncoding, retrieves feature from dataset and 1hot encode
       The feature.

    Args:
        name (str): name of layer
        dataset (pandas.DataFrame): the tabluar data
        dtype (pandas data types): e.g. "int64"
        is_y (bool, optional): if the dataset has target label. Defaults to False.
        max_tokens (int, optional): one-hot vector max size. Defaults to None.

    Returns:
        lambda function: encoder(feature)
    """
    # if is_y:
    #     feature_ds = dataset.map(lambda x, y: x[name])
    # else:
    #     feature_ds = dataset.map(lambda x: x[name])

    # Create a Discretization for our integer indices.
    encoder = CategoryEncoding(
        num_tokens=max_tokens, output_mode="one_hot", name=name + "_1hot"
    )
    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(feature)


def get_df_cardinality(df):
    """Returns a nice dataframe of the cardinality/type of each field

    Args:
        df (pandas.DataFrame): a tabular table

    Returns:
        pandas.DataFrame: a nice dataframe of the cardinality/type of each field
    """
    b = df.dtypes.reset_index().rename({0: "type"}, axis=1)
    a = df[df.select_dtypes("object").columns].nunique().reset_index(name="cardinality")
    return pd.merge(a, b, how="outer")


def get_col_types(df, is_y, verbose=False):
    """
    Given the dtypes of a pandas dataframe, returns two list, one
    with col names of numerical (int/float) columns, and one for
    categorical (object) columns

    Args:
        df (pandas.Dataframe): a pandas dataframe with column names
        is_y (list,list): a list of coloumn names for each type


    Returns:
        (list,list):two list, one
        with col names of numerical (int/float) columns, and one for
        categorical (object) columns
    """

    if is_y:
        colTypes = df.drop("target", axis=1).dtypes.to_dict()
    else:
        colTypes = df.dtypes.to_dict()
    catCols = []
    floatCols = []
    intCols = []
    for i in colTypes.items():
        if isinstance(i[1], np.dtypes.ObjectDType):
            # if i[1] is object:
            catCols.append(i[0])
        elif isinstance(i[1], (int, np.dtypes.Int64DType)):
            # elif i[1] is int or i[1] is pd.Int64Dtype():
            intCols.append(i[0])
        elif isinstance(i[1], (float, np.dtypes.Float64DType)):
            # elif i[1] is float:
            floatCols.append(i[0])
        else:
            print("Print-unused type: ", i)
    if verbose:
        print("Columns Types: \n", colTypes)
        print("\n")
        print("Categorical Columns: ", catCols)
        print("\n")
        print("Int Columns: ", intCols)
        print("\n")
        print("Float Columns: ", floatCols)
        print("\n")
    return catCols, intCols, floatCols


def labelEncoding(df, catCols):
    """Label encodes the categorical columns of a dataframe

    Args:
        df (pd.DataFrame): dataframe to label-encode cat columns
        catCols (list): list of categorical column names (string)

    Returns:
        pd.DataFrame: label-encoded dataframe
    """

    label_encoder = {}
    for col in catCols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        label_encoder[col] = encoder
    return df, label_encoder


def df_to_dset(dataframe, is_y, shuffle=False, batch_size=100):
    """Given a pandas datafrae, returns a tf.Dataset

    Args:
        dataframe (pd.DataFrame): pandas dataframe
        is_y (bool): indicating if there is a target label (Y) to be considered
        shuffle (bool, optional): to shuffle or not. Defaults to False.
        batch_size (int, optional): batch size. Defaults to 100.

    Returns:
        tf.data.Dataset: a tf.data.Dataset object from the dataframe.
    """
    dataframe = dataframe.copy()
    if is_y:
        labels = dataframe.pop("target")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def add_mask_labels(label_encoder):
    """Adds a mask value for each label_encoded column

    Args:
        label_encoder (list(LabelEncoder())): a list of LabelEncoder object for each encoded column
    """
    for feat in label_encoder.keys():
        label_encoder[feat].classes_ = np.append(label_encoder[feat].classes_, "mask")


def marg_sample(col_sampled: str, catMarginals: dict[str, dict[str, float]]) -> str:
    """Returns a random value sampled from the feature `col_sampled` using its
    marginal probabilities  (pi).

    Args:
        col_sampled (str): the feature to be sampled from
        catMarginals (dict): a dictionary of each feature with its marginal probs
    Returns:
        str: the sampled random value
    """
    return np.random.choice(
        list(catMarginals[col_sampled].keys()),
        p=list(catMarginals[col_sampled].values()),
    )


def load_dataset(dataset_log_name, data_path):
    """Given a dataset_log_name and data_path, loads dataset.

    Args:
        dataset_log_name (str): the name of the dataset
        data_path (str): the path of the dataset

    Raises:
        Exception: _description_

    Returns:
        pd.DataFrame,bool: loaded dataframe, is_target label.
    """
    if dataset_log_name == "PetFinder":
        # PetFinder Dataset
        petfinder_path = os.path.join(data_path, "Petfinder-mini.csv")
        dataframe = pd.read_csv(petfinder_path)
        # In the original dataset "4" indicates the pet was not adopted.
        dataframe["target"] = np.where(dataframe["AdoptionSpeed"] == 4, 0, 1)
        dataframe = dataframe.astype({"target": str})
        # Drop un-used; columns.
        dataframe = dataframe.drop(columns=["AdoptionSpeed", "Description"])
        print("Dataframe shape: ", dataframe.shape)
        is_target = True
    elif dataset_log_name == "Bank":
        # Bank Marketing Dataset
        bank_path = os.path.join(data_path, "Bankmarketing-mini.csv")
        dataframe = pd.read_csv(bank_path)
        dataframe["target"] = dataframe["y"]
        dataframe = dataframe.drop("y", axis=1)
        dataframe["target"] = np.where(dataframe["target"] == "no", "0", "1")
        print("Dataframe shape: ", dataframe.shape)
        is_target = True
    elif dataset_log_name == "Census":
        ## Adult Census Dataset
        census_path = os.path.join(data_path, "Adult_census.csv")
        adult_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "earnings",
        ]
        dataframe = pd.read_csv(census_path, names=adult_names)
        dataframe["target"] = np.where(dataframe["earnings"] == " >50K", "1", "0")
        dataframe = dataframe.drop(columns=["earnings"])
        is_target = True
    elif dataset_log_name == "Credit":
        ## HOME CREDIT DATASET
        ## Columns with mostly NAN are removed:
        col_list = [
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START",
            "ORGANIZATION_TYPE",
            # 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',  'EMERGENCYSTATE_MODE',
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "target",
        ]
        home_credit_path = os.path.join(data_path, "Credit_application_train.csv")
        dataframe = pd.read_csv(home_credit_path)
        dataframe["target"] = dataframe["TARGET"]
        dataframe = dataframe[col_list]
        ## This dataset has 307k rows. However, we have to drop NAs:
        dataframe = dataframe.dropna()
        is_target = True
    elif dataset_log_name == "Medical":
        ## Medical Charges Dataset
        medcharge_path = os.path.join(data_path, "MedicalCharges.csv")
        dataframe = pd.read_csv(medcharge_path)
        dataframe["DRG_Cd"] = dataframe["DRG_Cd"].astype(str)
        dataframe["Rndrng_Prvdr_Zip5"] = dataframe["Rndrng_Prvdr_Zip5"].astype(str)
        dataframe["Rndrng_Prvdr_State_FIPS"] = dataframe[
            "Rndrng_Prvdr_State_FIPS"
        ].astype(str)
        dataframe = dataframe.drop(
            [
                "Rndrng_Prvdr_CCN",
                "Rndrng_Prvdr_St",  # "Rndrng_Prvdr_Zip5",
                "Rndrng_Prvdr_RUCA_Desc",
                "Rndrng_Prvdr_Org_Name",
                "Rndrng_Prvdr_City",
                "Rndrng_Prvdr_State_FIPS",
                "DRG_Desc",
            ],
            axis=1,
        )
        print("Dataframe shape: ", dataframe.shape)
        is_target = False
    elif dataset_log_name == "Criteo":
        ## Criteo Dataset:
        criteo_names = (
            ["target"]
            + ["intCol_{}".format(i) for i in range(13)]
            + ["catCol_{}".format(i) for i in range(26)]
        )
        criteo_dtypes = ["object"] + ["Int64"] * 13 + ["object"] * 26
        criteo_path = os.path.join(data_path, "Criteo_1M.csv")
        dataframe = pd.read_csv(
            criteo_path,
            delimiter="\t",
            names=criteo_names,
            dtype={i: j for i, j in zip(criteo_names, criteo_dtypes)},
        )
        dataframe[["catCol_{}".format(i) for i in range(26)]] = dataframe[
            ["catCol_{}".format(i) for i in range(26)]
        ].fillna("99")
        # dataframe = dataframe[['target']+['catCol_{}'.format(i) for i in range(5)]+
        #                      ['intCol_{}'.format(i) for i in range(5)]] #.iloc[:500000,]
        dataframe = dataframe[
            ["target"]
            + ["catCol_0"]
            + ["catCol_1"]
            + ["catCol_4"]
            + ["catCol_5"]
            + ["catCol_8"]
            + ["intCol_{}".format(i) for i in range(5)]
        ]  # .iloc[:200000,]
        dataframe = dataframe.dropna()
        dataframe = dataframe.astype(
            {i: "int64" for i in ["intCol_{}".format(i) for i in range(5)]}
        )
        dataframe = dataframe.astype({"target": int})
        print("Dataframe shape: ", dataframe.shape)
        is_target = True
    elif dataset_log_name == "MIMIC":
        ## Medical Charges Dataset
        MIMIC_path = os.path.join(data_path, "MIMIC_iii.csv")
        dataframe = pd.read_csv(MIMIC_path)
        dataframe = dataframe.astype({"ICD9_CODE": "object"})
        is_target = False
    elif dataset_log_name == "Simulated":
        ## Medical Charges Dataset
        simulated_path = os.path.join(data_path, "Simulated.csv")
        dataframe = pd.read_csv(simulated_path)
        is_target = False
    else:
        raise Exception("Sorry, no dataset match")

    return dataframe, is_target


# old process (needed for viz):
def get_cat_tokens(df, catCols):
    """Returns a dict of {catCol:numberUniques},
    the unique values of all the catCols in df.
    """
    col_tokens_all = {}
    for i in catCols:
        col_tokens_all[i] = len(df[i].unique())
    return col_tokens_all


# # old process:
# def get_emb_sizes(col_tokens_all):
#     """Assigns a recommended embedding size for each categorical column.

#     Args:
#         col_tokens_all (dict): the dictionary of categorical columns and
#                                 associated cardinality

#     Returns:
#         dic: the suggested embedding size for each categorical column
#     """
#     emb_sizes = {}
#     for key, value in col_tokens_all.items():
#         emb_sizes[key] = max(3, int(value ** (1 / 3))) if value > 2 else 2
#     return emb_sizes
