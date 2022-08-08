from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from .final_code.BooleanEvaluationModule import *
from .final_code.Scenario import Scenarios, NELL
from .final_code.LearnerModule import Learn_Once
from .final_code.ProbeSelectorModule import UtilityOnly
from .final_code.ActiveConsentEvaluation import ActiveConsentEvaluation
from .final_code.KnownProbesRepository import KnownProbesRepository
from sklearn.ensemble import RandomForestClassifier
import concurrent

db = SQLAlchemy()
DB_NAME = "demoDB.db"

repo13 = None
scen13 = None
architecture13 = None

repo8 = None
scen8 = None
architecture8 = None

Q_Value_Algorithm = Q_Value()

def set_repo13(r):
    global repo13
    repo13 = r

def set_repo8(r):
    global repo8
    repo8 = r

def set_scen13(s):
    global scen13
    scen13 = s

def set_scen8(s):
    global scen8
    scen8 = s

def set_architecture13(a):
    global architecture13
    architecture13 = a

def set_architecture8(a):
    global architecture8
    architecture8 = a

def get_repo13():
    global repo13
    return repo13

def get_scen13():
    global scen13
    return scen13

def get_architecture13():
    global architecture13
    return architecture13

def get_repo8():
    global repo8
    return repo8

def get_scen8():
    global scen8
    return scen8

def get_architecture8():
    global architecture8
    return architecture8

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    from .models import User, Note

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    set_query_data13()
    set_query_data8()

    return app


def create_database(app):
    print('/website/' + DB_NAME)
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')

def set_query_data13():
    print( 'set_query_data13')
    global repo13
    global scen13
    global architecture
    Q_Value_Algorithm = Q_Value()
    scen13 = Scenarios(NELL('Q13'))

    initial_idx = randomize_known_probes(scen13, 80)
    scen13.experiment_init(initial_idx)
    repo13 = KnownProbesRepository(X_train=(scen13.get_X())[initial_idx].astype(int),
                                 y_train=(scen13.get_y())[initial_idx].astype(int))
    print("generate repo13 and scen")
    learnerModule = Learn_Once(classifier=RandomForestClassifier(n_estimators=100))
    # learnerModule = Learn_Once(classifier=GaussianNB())
    BooleanEvaluationModule_Q_Value = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Q_Value, ProbeSelectorModule)

def set_query_data8():
    print('set_query_data8')
    global repo8
    global scen8
    global architecture8
    Q_Value_Algorithm = Q_Value()
    scen8 = Scenarios(NELL('Q8'))

    initial_idx = randomize_known_probes(scen8, 80)
    scen8.experiment_init(initial_idx)
    repo8 = KnownProbesRepository(X_train=(scen8.get_X())[initial_idx].astype(int),
                                 y_train=(scen8.get_y())[initial_idx].astype(int))
    print("generate repo8 and scen")
    learnerModule = Learn_Once(classifier=RandomForestClassifier(n_estimators=100))
    # learnerModule = Learn_Once(classifier=GaussianNB())
    BooleanEvaluationModule_Q_Value = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture8 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Q_Value, ProbeSelectorModule)


def randomize_known_probes(scen, init_number):
    variables_indices = []
    variables=get_variables(scen.get_ls_dnf())
    for concept in variables:
        transaction_num = int(str(concept).split('v')[-1])
        index_X = scen.index_of(scen.get_X(), transaction_num)
        variables_indices.append(index_X)
    initial_indices=np.random.choice(list(set(range(2,scen.get_X().shape[0])).difference(set(variables_indices))), size=init_number, replace=False)
    return initial_indices

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def get_variables(ls_dnf):
    def foo(list_exps_dnf):
        variables = set()

        for exp in list_exps_dnf:
            variables = variables.union(set(list(exp.get_symbols())))

        return list(variables)

    k = 10
    chunks = chunkify(ls_dnf, k)
    threads_list = list()
    results = set()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(k):
            t = executor.submit(foo, chunks[i])
            threads_list.append(t)

        for t in threads_list:
            results = results.union(set(t.result()))

    results = list(set(results)).copy()
    # print(len(results))
    return results