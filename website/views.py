from flask import Blueprint, render_template, request, session
from .models import Query, Variable, Boolean

'''
from sklearn.ensemble import RandomForestClassifier
from .final_code.BooleanEvaluationModule import *
from .final_code.KnownProbesRepository import KnownProbesRepository
from .final_code.Scenario import Scenarios, NELL
from .final_code.LearnerModule import Learn_Once
from .final_code.ProbeSelectorModule import UtilityOnly
from .final_code.ActiveConsentEvaluation import ActiveConsentEvaluation
'''
import copy
from . import *
import json

Q_Value_Algorithm = Q_Value()
idx = 0
concept = None
bool_dic = {}
curr_repo = None
curr_scen = None
curr_architecture = None


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    print("home")


    idx = 0
    queries = Query.query.all()
    for q in queries:
        q.body = q.body.replace("\"", "'")
    print(queries)

    return render_template("home.html", queries = queries)



@views.route('/evaluate1', methods=['GET', 'POST'])
def evaluate1():
    global idx
    global concept
    global bool_dic

    print(request.form.get('query'))

    global curr_scen
    global curr_repo
    global curr_architecture
    # set_scen(None)
    # set_repo(None)
    # set_architecture(None)
    if '8' in (request.form.get('query')):
        curr_scen = copy.deepcopy(get_scen8())
        curr_repo = copy.deepcopy(get_repo8())
        curr_architecture = copy.deepcopy(get_architecture8())

    if '13' in (request.form.get('query')):
        curr_scen = copy.deepcopy(get_scen13())
        curr_repo = copy.deepcopy(get_repo13())
        curr_architecture = copy.deepcopy(get_architecture13())

    print(request.form.get('auto'))
    session['g_query'] = request.form.get('query')
    session['g_auto'] = True if request.form.get('auto') == 'on' else False

    # repo = get_repo()
    # scen = get_scen()
    # architecture_1 = get_architecture()
    concept, variables_probabilities, variables_utilities, variables_weights, variables_uncertainties = curr_architecture.get_current_consent(
        curr_repo, curr_scen, idx)

    value = curr_scen.get_value_of_concept(concept) if session['g_auto'] else None
    print("value " + str(value))
    #session['concept'] = str(concept)
    result = Variable.query.with_entities(Variable.Transaction_name, Variable.probability).all()
    info = create_probability_info(variables_probabilities, variables_utilities, variables_weights,
                                   variables_uncertainties, dict(result))
    c = Variable.query.filter_by(Transaction_name=str(concept)).first()
    bool_id = 0
    for bool in curr_scen.listExp_dnf:
        temp_bool = str(bool)
        bool_result = Boolean.query.filter_by(original_dnf=temp_bool).first()
        bool_dic[bool_id] = bool_result
        bool_id += 1
    query_data = Query.query.filter_by(id=session['g_query']).first()
    for bool in bool_dic.values():
        bool.original_dnf = str(curr_scen.listExp_dnf[bool.bool_id - 1])
    print("concept -> " + str(c.transaction_id))

    return render_template("evaluation.html", concept=c, booleans=bool_dic.values(), query_data=query_data,
                           probability_info=info, idx=idx, auto=session['g_auto'], value=value)


@views.route('/evaluate2', methods=['GET', 'POST'])
def evaluate2():
    global idx
    global concept
    global bool_dic

    global curr_scen
    global curr_repo
    global curr_architecture

    idx += 1
    print("evaluate2 post")
    print(request.form.get('ans'))
    print(request.form.get('auto'))
    session['g_auto'] = True if request.form.get('auto') == 'True' else False
    ans = True if request.form.get('ans') == "Correct" else False

    evaluated, truthValue, counter_expressions, curr_scen = curr_architecture.set_consent_value(curr_scen, concept, ans,idx)
    if evaluated:
        query_data = Query.query.filter_by(id=session['g_query']).first()
        for bool in bool_dic.values():
            bool.original_dnf = str(curr_scen.listExp_dnf[bool.bool_id - 1])
        return render_template("done.html",idx=idx, query_data=query_data, booleans=bool_dic.values())

    concept, variables_probabilities, variables_utilities, variables_weights, variables_uncertainties = curr_architecture.get_current_consent(
        curr_repo, curr_scen, idx)
    value = curr_scen.get_value_of_concept(concept)
    print("value " +str(value))
    result = Variable.query.with_entities(Variable.Transaction_name, Variable.probability).all()

    info = create_probability_info(variables_probabilities, variables_utilities, variables_weights,
                                   variables_uncertainties, dict(result))
    c = Variable.query.filter_by(Transaction_name=str(concept)).first()
    query_data = Query.query.filter_by(id=session['g_query']).first()
    for bool in bool_dic.values():
        bool.original_dnf = str(curr_scen.listExp_dnf[bool.bool_id - 1])
    print("concept -> " + str(c.transaction_id))

    return render_template("evaluation.html", concept=c, booleans=bool_dic.values(), query_data=query_data,
                           probability_info=info, idx=idx, auto=session['g_auto'], value=value)


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

class prob_info:
    def __init__(self, name, probability, utility, uncertainty, weight,nell):
        self.name = name
        self.uncertainty = uncertainty
        self.utility = utility
        self.probability = probability
        self.weight = weight
        self.nell = nell



def create_probability_info(variables_probabilities, variables_utilities, variables_weights, variables_uncertainties, nell_probability):
    probabilities_info = []
    variables = variables_utilities.keys()
    for var in variables:
        temp = prob_info(str(var),variables_probabilities[var],round(variables_utilities[var],2), variables_uncertainties[var],variables_weights[var],nell_probability[str(var)])
        probabilities_info.append(temp)
    return probabilities_info

