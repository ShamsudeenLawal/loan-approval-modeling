import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_data(data: pd.DataFrame, drop_missing: bool = False) -> pd.DataFrame:
    """
    This function cleans the dataset, with the following cleaning operation performed:
    column renaming, type casting, standardizing categorical variables,
    target mapping, dropping duplicates if exists, dropping irrelevant variables (LoanID),
    
    Args:
    - data: dataset frame to clean
    - drop_missing: a boolean value specifying whether to drop missing values or not, default to False
    """
    # rename columns for consistency
    data.columns = data.columns.str.replace("_", "")
    
    # dropping irrelevant features ("LoanID")
    data = data.drop(columns=["LoanID"])

    # dropping duplicates if exists
    data = data.drop_duplicates()

    # type casting
    data["CreditHistory"] = data["CreditHistory"].astype("category")
    
    # standardizing categorical variables
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.title()
    
    # map target variable
    data["LoanStatus"] = data["LoanStatus"].map(lambda target_value: "Yes" if target_value=="Y" else "No")

    # drop missing
    if drop_missing:
        data = data.dropna()
        data = data.reset_index(drop=True)

    return data


def get_preprocessor(numerical_features, categorical_features):
    
    numerical_preprocessor_pipeline = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    cat_preprocessor_pipeline = Pipeline(
        steps=[
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    # creating a transformer to combine the two pipelines
    preprocessor = ColumnTransformer([
        ("num_preprocessor", numerical_preprocessor_pipeline, numerical_features),
        ("cat_preprocessor", cat_preprocessor_pipeline, categorical_features)
    ])

    return preprocessor
