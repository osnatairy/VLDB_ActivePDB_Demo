import os

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from .BooleanEvaluationModule import algebra


class Setting:
    def get_Boolean_Provenance(self):
        pass

class TPCH_Q8(Setting):
    def read_dataset(self,tree_num):
        path = 'TPCH_RESULTS'
        if not os.path.exists(path):
            os.makedirs(path)
        dataset_str = str(path) + "\TPCH_Dataset_Q8_labels_Tree{}_labels0.csv".format(tree_num)
        self.dataset = pd.read_csv(dataset_str, low_memory=False)
    def __init__(self, tree_number):
        self.read_dataset(tree_num=tree_number)
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]
        self.preprocess_data()
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))

        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        self.ls_dnf,self.ls_cnf= self.get_Boolean_Provenance()

    

    def preprocess_data(self):
        lb_make = LabelEncoder()
        self.X = self.X.drop(columns=['key'])
        self.variables_real_probabilities = dict(
            zip(list(self.X['Transaction id'].values), list(self.X['Probability']).copy()))
        self.X = self.X.drop(columns=['Probability'])
        self.X['c_nationkey'] = self.X['c_nationkey'].astype(str).str.replace(" ", "").astype(str)
        self.X['c_mktsegment'] = self.X['c_mktsegment'].astype(str).str.replace(" ", "").astype(str)
        if 'p_type' in self.X.columns:
            self.X['p_type'] = self.X['p_type'].astype(str).str.replace(" ", "").astype(str)
            self.X['p_mfgr'] = self.X['p_mfgr'].astype(str).str.replace(" ", "").astype(str)
            self.X['p_brand'] = self.X['p_brand'].astype(str).str.replace(" ", "").astype(str)
        self.X['o_orderstatus'] = self.X['o_orderstatus'].astype(str).str.replace(" ", "").astype(str)
        self.X['o_orderpriority'] = self.X['o_orderpriority'].astype(str).str.replace(" ", "").astype(str)
        if 's_nationkey' in self.X.columns:
            self.X['s_nationkey'] = self.X['s_nationkey'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_returnflag'] = self.X['l_returnflag'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_linestatus'] = self.X['l_linestatus'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_shipinstruct'] = self.X['l_shipinstruct'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_shipmode'] = self.X['l_shipmode'].astype(str).str.replace(" ", "").astype(str)
        self.X['o_orderstatus'] = self.X['o_orderstatus'].astype(str).str.replace(" ", "").astype(str)
        self.X['c_nationkey'] = lb_make.fit_transform(self.X["c_nationkey"])
        self.X['c_mktsegment'] = lb_make.fit_transform(self.X["c_mktsegment"])
        if 'p_type' in self.X.columns:
            self.X['p_type'] = lb_make.fit_transform(self.X["p_type"])
            self.X['p_mfgr'] = lb_make.fit_transform(self.X["p_mfgr"])
            self.X['p_brand'] = lb_make.fit_transform(self.X["p_brand"])
        self.X['o_orderstatus'] = lb_make.fit_transform(self.X["o_orderstatus"])
        self.X['o_orderpriority'] = lb_make.fit_transform(self.X["o_orderpriority"])
        if 's_nationkey' in self.X.columns:
            self.X['s_nationkey'] = lb_make.fit_transform(self.X["s_nationkey"])
        self.X['l_returnflag'] = lb_make.fit_transform(self.X["l_returnflag"])
        self.X['l_linestatus'] = lb_make.fit_transform(self.X["l_linestatus"])
        self.X['l_shipinstruct'] = lb_make.fit_transform(self.X["l_shipinstruct"])
        self.X['l_shipmode'] = lb_make.fit_transform(self.X["l_shipmode"])
        self.X['o_orderstatus'] = lb_make.fit_transform(self.X["o_orderstatus"])
        self.X['type'] = lb_make.fit_transform(self.X["type"])
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))
        transactions = pd.DataFrame(self.X['Transaction id']).astype(int)
    def get_Boolean_Provenance(self):
        path="TPCH_RESULTS"
        text_file_dnf = open(str(path) + r"\UseCases_Expressions_TPC_H_Q8_Dnf.txt", "r")
        lines_dnf = text_file_dnf.readlines()

        text_file_cnf = open(str(path) + r"\UseCases_Expressions_TPC_H_Q8_Cnf.txt", "r")

        lines_cnf = text_file_cnf.readlines()

        expressions_dnf = []
        expressions_cnf = []

        for line in lines_dnf:
            dnf_form = algebra.parse(line)
            expressions_dnf.append(dnf_form)

        for line in lines_cnf:
            cnf_form = algebra.parse(line)
            expressions_cnf.append(cnf_form)

        return expressions_dnf, expressions_cnf
    
class H1B(Setting):
    def __init__(self):

        self.dataset = self.read_dataset()
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]
        self.preprocess_data()
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))
        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        self.dataset = self.dataset.replace({"CERTIFIED": 1, "WITHDRAWN": 0})
    def read_dataset(self):
        path = 'H1B_RESULTS'
        if not os.path.exists(path):
            os.makedirs(path)
        dataset_path_str = str(path) + "\H1B_Dataset.csv"

        dataset = pd.read_csv(dataset_path_str, low_memory=False)
        return dataset
    def preprocess_data(self):
        lb_make = LabelEncoder()
        self.X['VISA_CLASS'] = lb_make.fit_transform(self.X["VISA_CLASS"])
        self.X['EMPLOYER_NAME'] = lb_make.fit_transform(self.X["EMPLOYER_NAME"])
        self.X['EMPLOYER_STATE'] = lb_make.fit_transform(self.X["EMPLOYER_STATE"])
        self.X['EMPLOYER_COUNTRY'] = lb_make.fit_transform(self.X["EMPLOYER_COUNTRY"])
        self.X['SOC_NAME'] = lb_make.fit_transform(self.X["SOC_NAME"])
        self.X['PW_UNIT_OF_PAY'] = lb_make.fit_transform(self.X["PW_UNIT_OF_PAY"])
        self.X['PW_SOURCE'] = lb_make.fit_transform(self.X["PW_SOURCE"])
        self.X['PW_SOURCE_OTHER'] = lb_make.fit_transform(self.X["PW_SOURCE_OTHER"])
        self.X['WAGE_UNIT_OF_PAY'] = lb_make.fit_transform(self.X["WAGE_UNIT_OF_PAY"])
        self.X = DataFrameImputer().fit_transform(self.X)
        #
        #imp = Imputer(strategy="most_frequent")
        # imp.fit_transform(self.X)
        self.X['H1B_DEPENDENT'] = lb_make.fit_transform(self.X["H1B_DEPENDENT"])
        self.X['WORKSITE_STATE'] = lb_make.fit_transform(self.X["WORKSITE_STATE"])
        self.X['WILLFUL_VIOLATOR'] = lb_make.fit_transform(self.X["WILLFUL_VIOLATOR"])
        self.X['FULL_TIME_POSITION'] = lb_make.fit_transform(self.X["FULL_TIME_POSITION"])
        self.y = self.y.replace({"CERTIFIED": 1, "WITHDRAWN": 0})

    def read_Boolean_expressions_H1B(self, path, query_number, num_of_companies):
        expressions_dnf = []
        expressions_cnf = []
        regular = []
        for query in query_number:
            dnf_file = open(str(path) + r"\H1B_{}_Q{}\UseCases_H1B_DNF_Expressions.txt".format(num_of_companies, query),
                            "r")
            lines_dnf = dnf_file.readlines()
            for line in lines_dnf:
                line = line.replace("p", "v")
                expressions_dnf.append(algebra.parse(line))
            cnf_file = open(str(path) + r"\H1B_{}_Q{}\UseCases_H1B_CNF_Expressions.txt".format(num_of_companies, query),
                            "r")
            lines_cnf = cnf_file.readlines()
            for line in lines_cnf:
                line = line.replace("p", "v")
                expressions_cnf.append(algebra.parse(line))
        return expressions_dnf,expressions_cnf
    def get_Boolean_Provenance(self):
        pass
    

class H1B_S3(H1B):
    def __init__(self):
        super().__init__()
        self.get_Boolean_Provenance()

    def get_Boolean_Provenance(self):
        num_of_companies = 3
        num_of_uc = 1652
        queries_numbers = list(range(1, num_of_uc + 1))
        dnf_s, cnf_s = self.read_Boolean_expressions_H1B("H1B_RESULTS", query_number=queries_numbers,
                                                         num_of_companies=num_of_companies)
        self.ls_dnf = dnf_s
        self.ls_cnf = cnf_s
        return self.ls_dnf.copy(), self.ls_cnf.copy()


class PDB(Setting):
    def read_dataset(self):
        path = 'PDB_RESULTS'
        if not os.path.exists(path):
            os.makedirs(path)
        dataset_str = str(path) + "\PDB_DATASET_large_expriment.xlsx"
        self.dataset = pd.read_excel(dataset_str )
        print("read")
    def __init__(self):
        self.read_dataset()
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]
        self.preprocess_data()
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))
        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        self.dataset = self.dataset.replace({"1": 1, "0": 0})

    def preprocess_data(self):
        lb_make = LabelEncoder()
        self.X = self.X.drop(columns=['key'])
        self.X.fillna(0, inplace=True)
        self.X['Acquired'] = self.X['Acquired'].astype(str)
        self.X['Acquired'] = lb_make.fit_transform(self.X["Acquired"])

        self.X['Acquiring'] = self.X['Acquiring'].astype(str)
        self.X['Acquiring'] = lb_make.fit_transform(self.X["Acquiring"])

        self.X['Amount'] = self.X['Amount'].astype(str)
        self.X['Amount'] = lb_make.fit_transform(self.X["Amount"])

        self.X['Date'] = self.X['Date'].astype(str)
        self.X['Date'] = lb_make.fit_transform(self.X["Date"])

        self.X['Graduate'] = self.X['Graduate'].astype(str)
        self.X['Graduate'] = lb_make.fit_transform(self.X["Graduate"])

        self.X['Institution'] = self.X['Institution'].astype(str)
        self.X['Institution'] = lb_make.fit_transform(self.X["Institution"])

        self.X['Major'] = self.X['Major'].astype(str)
        self.X['Major'] = lb_make.fit_transform(self.X["Major"])

        self.X['GraduationYear'] = self.X['GraduationYear'].astype(str)
        self.X['GraduationYear'] = lb_make.fit_transform(self.X["GraduationYear"])

        self.X['Company'] = self.X['Company'].astype(str)
        self.X['Company'] = lb_make.fit_transform(self.X["Company"])

        self.X['Position'] = self.X['Position'].astype(str)
        self.X['Position'] = lb_make.fit_transform(self.X["Position"])

        self.X['PositionHolder'] = self.X['PositionHolder'].astype(str)
        self.X['PositionHolder'] = lb_make.fit_transform(self.X["PositionHolder"])

        self.X = DataFrameImputer().fit_transform(self.X)

    def get_Boolean_Provenance(self):
        path = "PDB_RESULTS"
        text_file_dnf = open(str(path) + r"\PDB_Boolean_Expressions_large_experiment.txt", "r")
        lines_dnf = text_file_dnf.readlines()



        expressions_dnf = []
        expressions_cnf = []

        for line in lines_dnf:
            dnf_form = algebra.parse(line)
            expressions_dnf.append(dnf_form)
            cnf_form=algebra.cnf(dnf_form)
            expressions_cnf.append(cnf_form)


        return expressions_dnf, expressions_cnf


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value

        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]

                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class NELL(Setting):
    def read_dataset(self):
        print('__file__:    ', __file__)
        path = 'C:/Users/user/Documents/PhD/VLDB_Demo/website/final_code/NELL'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        dataset_str = str(path) + "/new_dataset.csv"
        cwd = os.getcwd()  # Get the current working directory (cwd)
        files = os.listdir('./website/final_code')  # Get all the files in that directory
        print("Files in %r: %s" % (cwd, files))
        # dataset_str = './new_dataset.csv'
        self.dataset = pd.read_csv(dataset_str)

    def __init__(self,query):
        self.read_dataset()
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]
        self.query=query
        self.preprocess_data()
        self.dict_variables = dict(zip(list(self.X['Transaction_id'].values), list(self.y)))

        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        self.ls_dnf, self.ls_cnf = self.get_Boolean_Provenance()

    def preprocess_data(self):
        lb_make = LabelEncoder()
        self.X = self.X.drop(columns=['id'])
        self.X = self.X.drop(columns=['id_gen'])

        self.variables_real_probabilities = dict(
            zip(list(self.X['Transaction_id'].values), list(self.X['probability']).copy()))
        self.X = self.X.drop(columns=['probability'])
        self.X['entity'] = self.X['entity'].astype(str).str.replace(" ", "").astype(str)
        self.X['relation'] = self.X['relation'].astype(str).str.replace(" ", "").astype(str)

        self.X['value'] = self.X['value'].astype(str).str.replace(" ", "").astype(str)
        self.X['source'] = self.X['source'].astype(str).str.replace(" ", "").astype(str)

        self.X['action'] = self.X['action'].astype(str).str.replace(" ", "").astype(str)
        self.X['iteration'] = self.X['iteration'].astype(str).str.replace(" ", "").astype(str)
        self.X['entity_words'] = self.X['entity_words'].astype(str).str.replace(" ", "").astype(str)
        self.X['value_word'] = self.X['value_word'].astype(str).str.replace(" ", "").astype(str)
        self.X['entity_letters'] = self.X['entity_letters'].astype(str).str.replace(" ", "").astype(str)
        self.X['value_letters'] = self.X['value_letters'].astype(str).str.replace(" ", "").astype(str)
        self.X['has_probability'] = self.X['has_probability'].astype(str).str.replace(" ", "").astype(str)
        self.X['e_generalizations'] = self.X['e_generalizations'].astype(str).str.replace(" ", "").astype(str)
        self.X['e_instancetype'] = self.X['e_instancetype'].astype(str).str.replace(" ", "").astype(str)
        self.X['e_populate'] = self.X['e_populate'].astype(str).str.replace(" ", "").astype(str)
        self.X['e_antireflexive'] = self.X['e_antireflexive'].astype(str).str.replace(" ", "").astype(str)
        self.X['e_antisymmetric'] = self.X['e_antisymmetric'].astype(str).str.replace(" ", "").astype(str)
        self.X['e_range'] = self.X['e_range'].astype(str).str.replace(" ", "").astype(str)
        self.X['v_generalizations'] = self.X['v_generalizations'].astype(str).str.replace(" ", "").astype(str)
        self.X['v_instancetype'] = self.X['v_instancetype'].astype(str).str.replace(" ", "").astype(str)
        self.X['v_populate'] = self.X['v_populate'].astype(str).str.replace(" ", "").astype(str)
        self.X['v_antireflexive'] = self.X['v_antireflexive'].astype(str).str.replace(" ", "").astype(str)
        self.X['v_antisymmetric'] = self.X['v_antisymmetric'].astype(str).str.replace(" ", "").astype(str)
        self.X['v_range'] = self.X['v_range'].astype(str).str.replace(" ", "").astype(str)




        self.X['entity'] = lb_make.fit_transform(self.X["entity"])
        self.X['relation'] = lb_make.fit_transform(self.X["relation"])
        self.X['value'] = lb_make.fit_transform(self.X["value"])
        self.X['source'] = lb_make.fit_transform(self.X["source"])
        self.X['action'] = lb_make.fit_transform(self.X["action"])
        self.X['iteration'] = lb_make.fit_transform(self.X["iteration"])
        self.X['entity_words'] = lb_make.fit_transform(self.X["entity_words"])
        self.X['value_letters'] = lb_make.fit_transform(self.X["value_letters"])
        self.X['has_probability'] = lb_make.fit_transform(self.X["has_probability"])
        self.X['e_generalizations'] = lb_make.fit_transform(self.X["e_generalizations"])
        self.X['e_instancetype'] = lb_make.fit_transform(self.X["e_instancetype"])
        self.X['e_populate'] = lb_make.fit_transform(self.X["e_populate"])
        self.X['e_antireflexive'] = lb_make.fit_transform(self.X["e_antireflexive"])
        self.X['e_antisymmetric'] = lb_make.fit_transform(self.X["e_antisymmetric"])
        self.X['e_range'] = lb_make.fit_transform(self.X["e_range"])
        self.X['v_generalizations'] = lb_make.fit_transform(self.X["v_generalizations"])
        self.X['v_instancetype'] = lb_make.fit_transform(self.X["v_instancetype"])
        self.X['v_populate'] = lb_make.fit_transform(self.X["v_populate"])
        self.X['v_antireflexive'] = lb_make.fit_transform(self.X["v_antireflexive"])
        self.X['v_antisymmetric'] = lb_make.fit_transform(self.X["v_antisymmetric"])
        self.X['v_range'] = lb_make.fit_transform(self.X["v_range"])


        self.dict_variables = dict(zip(list(self.X['Transaction_id'].values), list(self.y)))
        transactions = pd.DataFrame(self.X['Transaction_id']).astype(int)

    def  get_Boolean_Provenance(self):
        path = 'C:/Users/user/Documents/PhD/VLDB_Demo/website/final_code/NELL/'#"NELL"
        file_str = str(path) + "UseCases_Expressions_NELL_{}_Dnf.txt".format(self.query)
        text_file_dnf = open(file_str,"r")#open(str(path) + r"/UseCases_Expressions_NELL_{}_Dnf.txt".format(self.query), "r")
        lines_dnf = text_file_dnf.readlines()

        file_str = str(path) + "UseCases_Expressions_NELL_{}_Cnf.txt".format(self.query)
        text_file_cnf = open(file_str,"r")#open(str(path) + r"/UseCases_Expressions_NELL_{}_Cnf.txt".format(self.query), "r")

        lines_cnf = text_file_cnf.readlines()

        expressions_dnf = []
        expressions_cnf = []

        for line in lines_dnf:
            dnf_form = algebra.parse(line)
            expressions_dnf.append(dnf_form)

        for line in lines_cnf:
            cnf_form = algebra.parse(line)
            expressions_cnf.append(cnf_form)

        return expressions_dnf, expressions_cnf
