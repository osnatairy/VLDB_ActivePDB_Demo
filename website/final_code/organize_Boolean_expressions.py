import re

import pandas as pd
from boolean import boolean
from sklearn.ensemble import RandomForestClassifier
import os

from BooleanHelpers import BooleanHelper


def obtain_provenence(df: pd.DataFrame):
    # pysqldf = lambda q: sqldf(q, globals())
    # dallasDf = psql.sqldf("select * from toydf where City ='Dallas'")

    # q = 'select * from df where SOC_NAME = ANALYSTS and CASE_SUBMITTED_YEAR=2016'
    # q = 'select * from df'
    conditions = " SOC_NAME ='ANALYSTS' and CASE_SUBMITTED_YEAR='2017' and WAGE_RATE_OF_PAY_FROM>150000 and WORKSITE_STATE='CA' and CASE_SUBMITTED_MONTH=1"
    q = "select * from df where {}".format(conditions)
    print(q)
    projected_df = psql.sqldf(q, locals())
    print(projected_df)
    print("################################################################")
    q = "select count(Transaction_Number),CASE_STATUS from df where {} group by CASE_STATUS ".format(conditions)
    print(q)
    print(psql.sqldf(q, locals()))
    # print(psql.sqldf(
    #    "select count(Transaction_Number),CASE_SUBMITTED_MONTH from df where {} group by CASE_SUBMITTED_MONTH ".format(conditions),
    #    locals()))

    rf = projected_df['Transaction_Number']
    transaction_ids = list(dict.fromkeys(list(rf.values)))
    return transaction_ids, projected_df
    """rt = df.loc[df['id reciever'] == int(prtcp)]
    rt = rt.loc[df["id sender"] == int(self.peer_to_share)]
    rt = rt.loc[df["EventNumber"] == int(self.event_number)]
    rt = rt['Transaction id']
    transaction_id = list(dict.fromkeys(list(rf.values)))[0]
    print(transaction_id)
    index = self.index_of(self.framework.X, transaction_id)
    transactions_numbers.append(index)"""
    # print(sub_data)
    # return transactions_numbers


def contain_company(company_name, companies):
    temp = False
    for abbr_name in companies:
        if abbr_name in company_name:
            temp = True
    return temp


def contain_soc(soc_name):
    temp = False
    socs = ['ANALYSTS', 'COMPUTER OCCUPATION']
    for abbr_name in socs:
        if abbr_name in soc_name:
            temp = True
    return temp


def no_denied(case_status):
    if 'DEN' in case_status:
        return False
    return True


def no_withdrawn(case_status):
    if 'WITHDRAWN' in case_status:
        return False
    return True


def chunk_preprocessing(columns, chunk, contain_deny, contain_withdrawn, filter_soc_name):
    # df = pd.read_csv("File_{}_H1B.csv".format(file_number),encoding='latin-1')
    companies = ['IBM CORPORATION', 'APPLE INC', 'BERKSHIRE', 'JPMORGAN', 'BANK OF AMERICA', 'VERIZON', 'AMAZON',
                 'GOOGLE INC', 'INTEL CORP', 'HP INC'
                                             'ORACLE AMERICA INC', 'TATA'
        , 'CHEVRON CORP', 'WELLS FARGO BANK', 'MICROSOFT CORP', 'GENERAL ELECTRIC',
                 'GOLDMAN SACHS', 'WAL MART', 'ERICSSON INC', 'CERNER', 'MORGAN STANLEY', 'LandT', 'ASTIR', 'CAPGEMINI']

    temp_df = pd.DataFrame(columns=columns)
    if contain_deny == False and filter_soc_name == True:
        for index, row in chunk.iterrows():
            if (contain_company(str(row['EMPLOYER_NAME']), companies) == True and contain_soc(
                    str(row['SOC_NAME'])) and no_denied(str(row['CASE_STATUS']))):
                temp_df = temp_df.append(row.to_dict(), ignore_index=True)
            # temp_df.append(pd.DataFrame(row))
    if contain_deny == False and filter_soc_name == False:
        for index, row in chunk.iterrows():
            if (contain_company(str(row['EMPLOYER_NAME']), companies) == True and no_denied(str(row['CASE_STATUS']))):
                temp_df = temp_df.append(row.to_dict(), ignore_index=True)
            # temp_df.append(pd.DataFrame(row))
    if contain_deny == True:
        if contain_withdrawn == True:
            for index, row in chunk.iterrows():
                if (contain_company(str(row['EMPLOYER_NAME']), companies) == True and contain_soc(
                        str(row['SOC_NAME']))):
                    temp_df = temp_df.append(row.to_dict(), ignore_index=True)
            # temp_df.append(pd.DataFrame(row))
        if contain_withdrawn == False:
            for index, row in chunk.iterrows():
                if (contain_company(str(row['EMPLOYER_NAME']), companies) == True and contain_soc(
                        str(row['SOC_NAME'])) and no_withdrawn(str(row['CASE_STATUS']))):
                    temp_df = temp_df.append(row.to_dict(), ignore_index=True)
            # temp_df.append(pd.DataFrame(row))

    print(temp_df)
    return temp_df


def chunk_preprocessing_1(columns, file_number: int):
    df = pd.read_csv("Master_H1B_Dataset.csv", encoding='latin-1')

    # df = pd.read_csv("File_{}_H1B.csv".format(file_number),encoding='latin-1')

    companies = ['IBM CORPORATION', 'APPLE INC', 'BERKSHIRE', 'JPMORGAN', 'BANK OF AMERICA', 'VERIZON', 'AMAZON',
                 'GOOGLE INC', 'INTEL CORP', 'HP INC'
                                             'ORACLE AMERICA INC', 'ACCENTURE', 'CAPGEMINI', 'SYNTEL', 'TATA',
                 'INFOSYS', 'DELOITTE', 'CHEVRON CORP', 'WELLS FARGO BANK', 'MICROSOFT CORP', 'GENERAL ELECTRIC',
                 'GOLDMAN SACHS', 'WAL MART']
    temp_df = pd.DataFrame(columns=columns)
    for index, row in df.iterrows():
        if (contain_company(str(row['EMPLOYER_NAME']), companies) == True):
            temp_df = temp_df.append(row.to_dict(), ignore_index=True)
            # temp_df.append(pd.DataFrame(row))
    return temp_df


def preprocess_data_set_Data():
    df_chunk = pd.read_csv("Master_H1B_Dataset.csv", encoding='latin-1')

    # print(columns_dict)

    # fltd = pd.read_csv("Fltd_file.csv")
    df_chunk['Transaction_Number'] = range(1, df_chunk.shape[0] + 1)

    df_chunk.loc[(df_chunk.CASE_STATUS == 'CERTIFIEDWITHDRAWN'), 'CASE_STATUS'] = 'WITHDRAWN'

    df_filtered = df_chunk[df_chunk['CASE_STATUS'] != 'DENIED']
    df_filtered.to_csv("fltd_master.csv", index=None, header=True)


def preprocess_data(contain_deny=False, contain_withdrawn=True, filter_soc_name=True):
    df_chunk = pd.read_csv("Master_H1B_Dataset.csv", encoding='latin-1', chunksize=30000)
    df = df_chunk.get_chunk(0)
    fltd = chunk_preprocessing(df.columns, df, contain_deny, contain_withdrawn, filter_soc_name)

    # print(columns_dict)
    for chnk in df_chunk:
        fltd = fltd.append(chunk_preprocessing(df.columns, chnk, contain_deny, contain_withdrawn, filter_soc_name),
                           ignore_index=True)

    print(fltd)
    # fltd = pd.read_csv("Fltd_file.csv")
    fltd['Transaction_Number'] = range(1, fltd.shape[0] + 1)

    if contain_deny == False:
        if filter_soc_name == True:
            fltd.loc[(fltd.CASE_STATUS == 'CERTIFIEDWITHDRAWN'), 'CASE_STATUS'] = 'WITHDRAWN'
            fltd.to_csv("Fltd_file_no_deny.csv", index=None, header=True)
        if filter_soc_name == False:
            fltd.loc[(fltd.CASE_STATUS == 'CERTIFIEDWITHDRAWN'), 'CASE_STATUS'] = 'WITHDRAWN'
            fltd.to_csv("Fltd_file_no_deny_no_soc_name_filtering.csv", index=None, header=True)
    else:
        if contain_withdrawn == True:
            fltd.to_csv("Fltd_file_with_deny.csv", index=None, header=True)
        if contain_withdrawn == False:
            fltd.to_csv("Fltd_file_with_deny_no_withdrawn.csv", index=None, header=True)


def divide_to_terms(df, by_naics: bool, cnf=True):
    if by_naics == True:

        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        q = "select count(Transaction_Number),NAICS_CODE_FLTD from df group by NAICS_CODE_FLTD "
        print(q)
        projected_df = psql.sqldf(q, locals())

        rf = projected_df['NAICS_CODE_FLTD']
        naics_codes = list(dict.fromkeys(list(rf.values)))
        if cnf == True:
            exp = TRUE
            for code in naics_codes:
                q = "select Transaction_Number from df where NAICS_CODE_FLTD=" + str(code)
                projected_df = psql.sqldf(q, locals())
                rf = projected_df['Transaction_Number']

                transactions = list(dict.fromkeys(list(rf.values)))
                transactions = [str("p") + str(val) for val in transactions]
                exp_temp = bh.convert_list_to_or_term(transactions)
                exp = AND(exp, exp_temp)
        else:
            exp = FALSE
            for code in naics_codes:
                q = "select Transaction_Number from df where NAICS_CODE_FLTD=" + str(code)
                projected_df = psql.sqldf(q, locals())
                rf = projected_df['Transaction_Number']

                transactions = list(dict.fromkeys(list(rf.values)))
                transactions = [str("p") + str(val) for val in transactions]
                exp_temp = bh.convert_list_to_and_term(transactions)
                exp = OR(exp, exp_temp)

    return exp


from boolean import boolean


def get_variables(list_exps):
    variables = set()

    for exp in list_exps:
        variables = variables.union(set(list(exp.get_symbols())))

    return list(set(list(variables)))


def seperate_string_number(string):
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i

        previous_character = i

        if x == len(string) - 2:
            groups.append(newword)
            newword = ''
    return groups


def assign_value_in_formulas(listExp, concept, value):
    algebra = boolean.BooleanAlgebra()
    for exp_num in range(0, len(listExp)):
        # print("exppppppppp beforerrrr: " + str(listExp))
        temp = listExp[exp_num]
        if concept in list(temp.get_symbols()):
            listExp[exp_num] = temp.subs({concept: algebra.parse((value))}, simplify=True)
    # print("exppppppppp afterrrrrrrrr: " + str(listExp))
    return listExp


import numpy as np
import pandas as pd

def add_Dnf_cnf_files(path,query_name):
    algebra = boolean.BooleanAlgebra()

    bh=BooleanHelper()
    file = open(str(path)+r'/UseCases_Expressions_NELL_{}.txt'.format(query_name))
    all_lines = file.readlines()
    cnf_s=[]
    dnf_s=[]
    for exp in all_lines:
        exp_temp=algebra.parse(str(exp))
        cnf_s.append(str(algebra.cnf(exp_temp)))
        dnf_s.append(str(algebra.dnf(exp_temp)))
    gather_and_generate_text_file_NELL(path, cnf_s, str(query_name + "_Cnf"))
    gather_and_generate_text_file_NELL(path, dnf_s, str(query_name + "_Dnf"))






def gather_and_generate_text_file_NELL(path, ls_of_expressions, file_name):
    list_exps = []
    for exp in ls_of_expressions:
        print(exp)
        temp = str(exp)
        list_exps.append(temp)
    with open(str(path) + r"\UseCases_Expressions_NELL_{}.txt".format(file_name), 'w') as file_handler:
        for item in list_exps:
            file_handler.write("{}\n".format(item))


def convert_variable_to_transaction_id(dataset_chunks, variables, regex):
    id_gens=list(dataset_chunks['id_gen'].values)
    print(id_gens)
    #new_id_gens=[str(id).split('_')[1] for id in id_gens]
    dataset_chunks['Transaction_id'] = dataset_chunks['Transaction_id'].astype(str)
    dict_variables = dict(zip(list(dataset_chunks['id_gen'].values), list(dataset_chunks['Transaction_id'].values)))
    #dict_variables = dict(zip(new_id_gens, list(dataset_chunks['Transaction_id'].values)))

    dict_variables = {k: str(str("v") + str(value)) for k, value in dict_variables.items()}
    print(dict_variables)
    return dict_variables


def Extract_expressions_from_file(path, query_name: str):
    """
    :param path:
    :param query_name: The name of the query we plan to run our system

    This function is used to convert the Boolean expressions from the separated files to one file. This function also maps the variables to their transaction_id in order to allow one format of variables

    """

    ls_exp = []
    dataset_chunks = pd.read_csv(str(path) + "/new_dataset.csv")

    algebra = boolean.BooleanAlgebra()
    my_list = os.listdir(str(path)+"/my_{}/".format(query_name))
    for i in my_list:
        file = open(path+'/my_{}/{}'.format(query_name, i))
        all_lines = file.readlines()
        #variables=str(all_lines[3])[14:-1].replace(" ","").split(",")


        exp = algebra.parse(str(all_lines[4]))
        ls_exp.append(exp)
    print(ls_exp)
    regex = re.compile('[^a-zA-Z]')

    num = 0
    variables = get_variables(ls_exp)

    transaction_variables_dict = convert_variable_to_transaction_id(dataset_chunks, variables, regex)
    temp_len_variables = len(variables)
    variables = get_variables(ls_exp)
    for var in variables:
        num += 1
        #ls_exp = assign_value_in_formulas(ls_exp, var, transaction_variables_dict[str(var).split('_')[1]])
        ls_exp = assign_value_in_formulas(ls_exp, var, transaction_variables_dict[str(var)])

        size_variables = temp_len_variables - num
        print(size_variables)
    gather_and_generate_text_file_NELL(path, ls_exp, '{}'.format(query_name))
    ls_exp = preprocess_expressions(ls_exp)
    gather_and_generate_text_file_NELL(path, ls_exp, '{}'.format(query_name))
def count_chars(x):
    length = len(x)
    digit = 0
    letters = 0
    space = 0
    other = 0
    for i in range(len(x)):
        if x[i].isalpha():
            letters += 1

    return letters

def preprocess_expressions(ls_exp):
    algebra = boolean.BooleanAlgebra()
    TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()

    variables = get_variables(ls_exp)

    for var in variables:
        var_str = str(var)
        if ("p" in var_str):
            trans_num = int(var_str.split('v')[-1])
            ls_exp = assign_value_in_formulas(ls_exp, var, "p{}".format(trans_num))
    return ls_exp


def gather_and_generate_text_file_H1B(path, ls_of_expressions, file_name):
    list_exps = []
    for exp in ls_of_expressions:
        print(exp)
        temp = str(exp)
        list_exps.append(temp)
    with open(str(path) + r"\UseCases_Expressions_TPC_H_{}.txt".format(file_name), 'w') as file_handler:
        for item in list_exps:
            file_handler.write("{}\n".format(item))


def organize_Dataset():
    #new_df = pd.read_excel("Nell_dataset_New.xlsx")
    new_df = pd.read_csv("new_dataset.csv")

    new_df = new_df.drop_duplicates(subset=['id'])
    new_df.to_csv("new_dataset.csv")



if __name__ == '__main__':
    # preprocess_data(contain_deny=False,contain_withdrawn=True,filter_soc_name=False)


    # Extract_expressions_from_file(8,2602)
    # generate_probabilities_tpch(how_many=51000)

    # dataset = pd.read_csv("TPCH_LABELINGS.csv")
    # extend_dataset(dataset,how_many=50000)

    # Extract_expressions_from_file("TPCH_RESULTS",3,11620)


    # Extract_expressions_from_file(path, 8, 2602)

    #Extract_expressions_from_file(path, "Q8_1000")

    #add_Dnf_cnf_files(path, 'Q8_1000')

    #Extract_expressions_from_file(path, 7)
    #add_Dnf_cnf_files(path,'Q7')

    #Extract_expressions_from_file(path, "Q10_5000")
    #add_Dnf_cnf_files(path, 'Q10_5000')



    #Extract_expressions_from_file(path, 5)
    #add_Dnf_cnf_files(path, 'Q5')
    #Extract_expressions_from_file(path, 7)
    #Extract_expressions_from_file(path, 3)
    #Extract_expressions_from_file(path, 5)

    path = "NELL"
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(13,14):
        Extract_expressions_from_file(path, "Q"+str(i))

        add_Dnf_cnf_files(path, 'Q'+str(i))
    #organize_Dataset()











