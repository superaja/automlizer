import numpy as np
import pandas as pd
from deap import creator
from tpot.export_utils import generate_pipeline_code, expr_to_tree
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from os import path
from aml_config import aml_config
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import mean_squared_error
from math import sqrt


def runTPOT(X, y, metric, algo):
    aml_config_dict = aml_config()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)
    
    if algo == "Classifier":
        pipeline_optimizer = TPOTClassifier(generations=1, population_size=5, verbosity=2, warm_start=True)
        pipeline_optimizer.fit(X_train, y_train)
        print(pipeline_optimizer.score(X_test, y_test))
    elif algo == 'Regressor':
        def aml_reg_scorer(y_pred, y_test):
            rsme = sqrt(mean_squared_error(y_test, y_pred))
            return rsme
        aml_custom_scorer = make_scorer(aml_reg_scorer, greater_is_better=False)
        pipeline_optimizer = TPOTRegressor(generations=1, population_size=5, verbosity=2, warm_start=True, scoring=aml_custom_scorer)
        pipeline_optimizer.fit(X_train, y_train)
        print(pipeline_optimizer.score(X_test, y_test)) 
    else: 
        raise Exception('Incorrect Problem Type')
    return pipeline_optimizer, pipeline_optimizer.score(X_test, y_test), len(pipeline_optimizer.evaluated_individuals_)
    
def createsklearnPipeline(pipeline_optimizer, pipes):
    # generate operator list
    pp_operators = []
    for k, v in pipeline_optimizer.operators_context.items():
        if 'sklearn.preprocessing' in str(v) or \
            'sklearn.decomposition' in str(v) or \
            'tpot.builtins' in str(v) or \
            'sklearn.cluster' in str(v) or \
            'sklearn.feature_selection' in str(v):
            pp_operators.append(k.lower())
        else:
            pass
    pp_operators.remove('stackingestimator') # remove stacking estimator from operators
    n = 1 + pipes
    p = {}
    plist = []
    for pipeline_string, attrib in sorted(pipeline_optimizer.evaluated_individuals_.items()):
        # convert pipeline string to scikit-learn pipeline object
        deap_pipeline = creator.Individual.from_string(pipeline_string, pipeline_optimizer._pset)
        sklearn_pipeline = pipeline_optimizer._toolbox.compile(expr=deap_pipeline)
        # print sklearn pipeline string
        sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(deap_pipeline, pipeline_optimizer._pset), pipeline_optimizer.operators)
        #print(n, sklearn_pipeline.steps)
        if attrib.get('internal_cv_score') > 0: # handle bad data in cv_score
            cv_score = attrib.get('internal_cv_score')
        else: 
            cv_score = abs(attrib.get('internal_cv_score')) # change this from None to abs for Regression
        for num, l in enumerate(sklearn_pipeline.steps):
            if l[0] not in 'featureunion': # ignore feature union for now
                #print(n, sklearn_pipeline.steps[num][1])
                params = sklearn_pipeline.steps[num][1].get_params()
                if 'stackingestimator' in l[0]:# identify stacking estimator
                    stack = 'Y'
                    algoName = str(params['estimator']).split('(')[0].lower()+'_stack'
                    params = params['estimator'].get_params()
                else: 
                    stack = 'N'
                    algoName = l[0]
                #pp_operators = ppoperator(pipeline_optimizer) # identify preprocessing algos 
                if l[0].startswith(tuple(pp_operators)):
                    pp_flag = 'Y'
                    params = l[1].get_params()
                    if l[0] in ['selectfrommodel', 'rfe']:
                        algoName = l[0]
                        params=params['estimator'].get_params()
                    else: 
                        algoName = str(l[1]).split('(')[0].lower()
                        params = l[1].get_params()
                else: 
                    pp_flag = 'N'
                p = {"PIPELINE": n, "ALGO_NAME":algoName, "STACK_FLG": stack, "PP_FLAG": pp_flag, "SCORE": cv_score}
                p.update(params)
                plist.append(p)         
        n = n+1 # update pipeline number
    master = pd.DataFrame()
    for i in plist:
        pip = int(i['PIPELINE'])
        alg = i['ALGO_NAME']
        algtype = i['STACK_FLG']
        score = i['SCORE']
        ppflag = i['PP_FLAG']
        pipeList = []
        aList = []
        atypeList = []
        hList = []
        vList = []
        sList = []
        ppList = []
        htypeList = []
        for k, v in i.items():
            if k not in ['PIPELINE', 'ALGO_NAME', 'SCORE', 'STACK_FLG', 'PP_FLAG']:
                pipeList.append(pip)
                aList.append(alg)
                atypeList.append(algtype)
                hList.append(k)
                if type(v) in [bool, str]: # check hyper value type
                    htype = 'C'
                else: 
                    htype = 'N'
                vList.append(v)
                htypeList.append(htype)
                sList.append(score)
                ppList.append(ppflag)
        df_dict = {'PIPELINE': pipeList, 'ALGO_NAME': aList, 'STACK_FLG': atypeList, 'PP_FLAG': ppList, 'SCORE': sList, "HYPER_NAME": hList,"HYPER_TYPE":htypeList, "HYPER_VALUE": vList}
        df = pd.DataFrame(df_dict)
        master = master.append(df)
    #stack_df = master[master['STACK_FLG']=='Y'].drop_duplicates()
    # drop bad pipelines
    '''
    if type(pipeline_optimizer) == 'TPOTRegressor':
        master.drop(master[master['SCORE'] > master['SCORE'].std()*4].index, inplace=True)
    '''
    # check if file exists
    if path.exists('pipeline.csv'):
        master.to_csv('pipeline.csv', mode='a', header=False, index=False)
    else:
        master.to_csv('pipeline.csv', index=False)

def pipelineRef(algo): # later in Mongo Ref Table

    # Generate ref table
    if algo == 'Classifier':
        tpot_obj = TPOTClassifier()
    else: 
        tpot_obj = TPOTRegressor()
    algoList = []
    totalHyperList = []
    hyperParaList =[]
    hyperParaType = []
    hyperValueList = []
    stack_algo = ['decisiontreeclassifier_stack',
                 'randomforestclassifier_stack',
                 'bernoullinb_stack',
                 'gradientboostingclassifier_stack',
                 'logisticregression_stack',
                 'gaussiannb_stack',
                 'kneighborsclassifier_stack',
                 'xgbclassifier_stack',
                 'xgbregressor_stack',
                 'lassolarscv_stack',
                 'elasticnetcv_stack',
                 'decisiontreeregressor_stack'
                 ]    

    for k, v in tpot_obj.default_config_dict.items():
        for key, value in v.items():
            #print(k.split('.')[1].lower(), k)
            if k.split('.')[1].lower() in ['feature_selection', 'decomposition', 'preprocessing', 'cluster', "builtins"]: # get only algos
                pass
            else: 
                if len(k.split('.')) == 2:
                    algo = k.split('.')[1].lower()
                else:
                    algo = k.split('.')[2].lower()
                algoList.append(algo)
                if type(value) == range: # ensure range is calculated correctly
                    vLength = value.stop - value.start + 1
                else:
                    vLength = len(v)
                # check hyperType
                if type(value[0]) in [bool, str]:
                    hType = 'S'
                else: 
                    hType = 'N'
                totalHyperList.append(vLength)
                hyperParaList.append(key)
                hyperValueList.append(len(value))
                hyperParaType.append(hType)
                # add the stacker algo to the reference
                if algo in [al.split('_')[0] for al in stack_algo]:
                    algoList.append(algo+'_stack')
                    totalHyperList.append(vLength)
                    hyperParaList.append(key)
                    hyperValueList.append(len(value))
                    hyperParaType.append(hType)
                else: 
                    pass
        ref_table = {"ALGO_NAME":algoList, "TOTAL_HYPER_COUNT": totalHyperList, "HYPER_NAME": hyperParaList, "HYPER_TYPE":hyperParaType, "TOTAL_HYPER_VALUE_COUNT": hyperValueList}
    df_ref = pd.DataFrame(ref_table)
    df_ref.to_csv('ref.csv', index=False)
    return df_ref


    