import pandas as pd 
import numpy as np

def select_columns_from_key_file(df, key_file):
    """ Function to drop the column without the "N" flag found in the column description excel file.
    --------
    Args: df; the pandas dataframe we want to apply this operation on.
          key_file; str, path the location of the excel file. this is built to run in notebook folder by default
    --------
    Returns: dataframe without the columns specified to drop.
    """
    df_key = pd.read_excel(key_file).dropna()
    cols_exclude = df_key[df_key['Exclude?']!='N']['LoanStatNew']
    drop_cols = []
    for i in cols_exclude:
        for c in df.columns:
            if i in c:
                drop_cols.append(c)
    new_df = df.drop(drop_cols,axis=1)
    return new_df

def load_data(dir_name):
    '''Loads csv file with loan data. Automatically eliminates unwanted columns and cols that are correlated with response.'''
    
    df = pd.read_csv(dir_name+'1.0_lc_initial_meanimp.csv', low_memory=False)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0',axis=1)
    
    df = select_columns_from_key_file(df, key_file=dir_name+'LCDataDictionary.xlsx')
    
    return df

def subset_lc_data(dir_name, new_fn):
    quant_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low',\
              'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',\
              'revol_bal', 'revol_util', 'total_acc', 'acc_now_delinq', 'tot_coll_amt',\
              'tot_cur_bal', 'tax_liens', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',\
              'installment']

    cat_cols = ['grade','emp_length', 'home_ownership','verification_status', 'zip_code', 'term',\
                'earliest_cr_line','initial_list_status','disbursement_method', 'application_type']

    response_col = 'loan_status'

    df = load_data(dir_name)
    sub_df = df[quant_cols+cat_cols+[response_col]].copy(deep=True)
    sub_df['age_of_cr_line'] = pd.to_datetime('today').year - pd.to_datetime(sub_df['earliest_cr_line']).dt.year
    sub_df.drop('earliest_cr_line',axis=1).dropna().to_csv(new_fn, index=False)
    
if __name__=='__main__':
    subset_lc_data(dir_name='./', new_fn='loans_data.csv')
    print('Done. File saved.')
