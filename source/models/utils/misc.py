from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import imp, os

def global_normalizer_minmax(df: pd.DataFrame, vital_cols: str=None):
    
    minmax_scaler = MinMaxScaler()
    
    if not vital_cols:
        raise ValueError('vital_cols cannot be None; please provide columns to normalize')

    for vs in vital_cols:
        df[vs] = minmax_scaler.fit_transform(df[vs].values.reshape(-1, 1))
            
    return df

class ExtractTimeSseriesBeforeDxOffset(BaseEstimator, TransformerMixin):
    def __init__(self, extracted_samples: int = 108) -> None:
        super().__init__()
        self.extracted_samples = extracted_samples

    def fit(self, X, y=None):
        return self

    def transform(
        self, 
        X: pd.DataFrame):

        target_dataset = []
        for grpID, grpData in X.groupby(by='groups'):
            grpData.reset_index(drop=True, inplace=True) # reset index
            grpData.reset_index(drop=False, inplace=True) # reset_index twice to remove previous index and then add new index as time_idx
            grpData.rename(columns={'index':'time_idx'}, inplace=True)

            sample = grpData.iloc[:self.extracted_samples, :]
            target_dataset.append(sample)

        return pd.concat(target_dataset).reset_index(drop=True)

def prepare_train_test_set(f_data: pd.DataFrame, extract_offset:bool=True):
    if extract_offset: ### Extract data before the DxOFFSET i.e., 9 hours of data
        target_data = ExtractTimeSseriesBeforeDxOffset(extracted_samples=108).transform(f_data.copy())
    else: target_data = f_data.copy()

# display(target_data.loc[target_data['groups']==1])
    for group in target_data['groups'].unique():
        if len(target_data[target_data['groups']==group]) != 108:
            print(group)

# Split dataset/groups in Train and Test dataset
    train_groups, test_groups = train_test_split(
    target_data["groups"].unique(), test_size=0.2, random_state=42
) # 80-20 split

    train_dataset = target_data[target_data["groups"].isin(train_groups)]
    test_val_dataset = target_data[target_data["groups"].isin(test_groups)]

# create train and validation datasets
    test_groups, val_groups = train_test_split(
    test_val_dataset["groups"].unique(), test_size=0.5, random_state=42
)

    test_dataset = test_val_dataset[test_val_dataset["groups"].isin(test_groups)]
    val_dataset = test_val_dataset[test_val_dataset["groups"].isin(val_groups)]

# Reset group numbers in train_dataset and test_dataset # not sure if required.
    train_dataset["groups"]=[group for group in range(len(train_dataset["groups"].unique())) for _ in range(len(train_dataset["time_idx"].unique()))]
    val_dataset["groups"]=[group for group in range(len(val_dataset["groups"].unique())) for _ in range(len(val_dataset["time_idx"].unique()))]
    test_dataset["groups"]=[group for group in range(len(test_dataset["groups"].unique())) for _ in range(len(test_dataset["time_idx"].unique()))]

    return train_dataset, val_dataset, test_dataset


def prepare_train_test_validation(f_data: pd.DataFrame, t_data: pd.DataFrame):

    target_data = f_data.copy()

# Split dataset/groups in Train and Test dataset
    train_groups, test_groups = train_test_split(
                        target_data["groups"].unique(), test_size=0.2, random_state=42) # 80-20 split

    train_dataset = target_data[target_data["groups"].isin(train_groups)]
    train_y_dataset = t_data[t_data['groups'].isin(train_groups)]

    test_val_dataset = target_data[target_data["groups"].isin(test_groups)]
    test_val_y_dataset = t_data[t_data['groups'].isin(test_groups)]


# create train and validation datasets
    test_groups, val_groups = train_test_split(
                        test_val_dataset["groups"].unique(), test_size=0.5, random_state=42)

    test_dataset = test_val_dataset[test_val_dataset["groups"].isin(test_groups)]
    test_y_dataset = test_val_y_dataset[test_val_y_dataset['groups'].isin(test_groups)]
    
    val_dataset = test_val_dataset[test_val_dataset["groups"].isin(val_groups)]
    val_y_dataset = test_val_y_dataset[test_val_y_dataset['groups'].isin(val_groups)]


# Reset group numbers in train_dataset and test_dataset # not sure if required.
    train_dataset["groups"]=[group for group in range(len(train_dataset["groups"].unique())) for _ in range(len(train_dataset["time_idx"].unique()))]
    train_y_dataset["groups"]=[group for group in range(len(train_y_dataset["groups"].unique())) for _ in range(len(train_y_dataset["time_idx"].unique()))]
    
    val_dataset["groups"]=[group for group in range(len(val_dataset["groups"].unique())) for _ in range(len(val_dataset["time_idx"].unique()))]
    val_y_dataset["groups"]=[group for group in range(len(val_y_dataset["groups"].unique())) for _ in range(len(val_y_dataset["time_idx"].unique()))]
    
    test_dataset["groups"]=[group for group in range(len(test_dataset["groups"].unique())) for _ in range(len(test_dataset["time_idx"].unique()))]
    test_y_dataset["groups"]=[group for group in range(len(test_y_dataset["groups"].unique())) for _ in range(len(test_y_dataset["time_idx"].unique()))]

    return (train_dataset, train_y_dataset), (val_dataset, val_y_dataset), (test_dataset, test_y_dataset)


def load_dy_module(module_file_path): # 'experiment_logs/latest/nhits_wc_BPHR_bprm_fm'
    '''
    how to use:
    # load_module = get_me_my_module(r'experiment_logs/latest/nhits_wc_BPHR_bprm_fm')
    # print(load_module.MODEL_NAME)

    '''
    file_pointer, file_path, descr = imp.find_module(module_file_path)

    load_module = imp.load_module(module_file_path, file_pointer, file_path, descr)

    return load_module


def create_model_directory(path):
    ## check path exits:
    isExists = os.path.exists(path)
    if not isExists:
        print(f'creating directory {path}')
        ## create directory
        os.makedirs(path)