import argparse
import json
import logging
import os
from shutil import rmtree
from time import time

import InterpretME.utils as utils
import numpy as np
import optuna
import pandas as pd
import validating_models.visualizations.decision_trees as constraint_viz
from InterpretME import preprocessing_data, sampling_strategy, classification, dtreeviz_lib
from InterpretME.pipeline import current_milli_time
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from validating_models.checker import Checker
from validating_models.constraint import ShaclSchemaConstraint
from validating_models.dataset import BaseDataset, ProcessedDataset
from validating_models.models.decision_tree import get_shadow_tree_from_checker
from validating_models.shacl_validation_engine import ReducedTravshaclCommunicator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(os.path.join('output', 'execution.log')))

consider_val_res_feature = False
use_heuristics = False
validation = True

model_quality = ''


def binary_classification(sampled_data, sampled_target, imp_features, cross_validation,
                          classes, st, test_split, model, results, min_max_depth, max_max_depth):
    """A modified version of InterpretME's binary classification method."""
    sampled_target['class'] = sampled_target['class'].astype(int)
    X = sampled_data
    y = sampled_target['class']

    X_input, y_input = X.values, y.values
    if model == 'Random Forest' or model == 'RFG':
        estimator = RandomForestClassifier(max_depth=max_max_depth, random_state=0)
    elif model == 'RFE':
        estimator = RandomForestClassifier(max_depth=max_max_depth, random_state=0, criterion='entropy')
    elif model == 'RFL':
        estimator = RandomForestClassifier(max_depth=max_max_depth, random_state=0, criterion='log_loss')
    elif model == 'AdaBoost':
        estimator = AdaBoostClassifier(random_state=0)
    elif model == 'Gradient Boosting' or model == 'GBLF':
        estimator = GradientBoostingClassifier(random_state=0)
    elif model == 'GBLS':
        estimator = GradientBoostingClassifier(random_state=0, loss='log_loss', criterion='squared_error')
    elif model == 'GBEF':
        estimator = GradientBoostingClassifier(random_state=0, loss='exponential', criterion='friedman_mse')
    elif model == 'GBES':
        estimator = GradientBoostingClassifier(random_state=0, loss='exponential', criterion='squared_error')

    cv = StratifiedShuffleSplit(n_splits=cross_validation, test_size=test_split, random_state=123)
    important_features = set()
    important_features_size = imp_features

    # Classification report for every iteration
    for i, (train, test) in enumerate(cv.split(X_input, y_input)):
        estimator.fit(X_input[train], y_input[train])
        y_predicted = estimator.predict(X_input[test])  # TODO: Is it necessary to do the prediction here if nothing happens with the prediction?

        fea_importance = estimator.feature_importances_
        indices = np.argsort(fea_importance)[::-1]
        for f in range(important_features_size):
            important_features.add(X.columns.values[indices[f]])

    data = classification.plot_feature_importance(estimator.feature_importances_, X.columns)
    results['feature_importance'] = data

    # Taking important features
    new_sampled_data = sampled_data[list(important_features)]
    indices = new_sampled_data.index.values
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
        new_sampled_data.values, sampled_target['class'].values, indices, random_state=123
    )

    feature_names = new_sampled_data.columns

    utils.pbar.total += 100
    utils.pbar.set_description('Model Training', refresh=True)
    # Hyperparameter Optimization using AutoML
    study = optuna.create_study(direction="maximize")
    automl_optuna = classification.AutoMLOptuna(min_max_depth, max_max_depth, X_train, y_train)
    study.optimize(automl_optuna, n_trials=100, callbacks=[classification.AdvanceProgressBarCallback()])
    params = study.best_params
    del params['classifier']
    best_clf = tree.DecisionTreeClassifier(**params)
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)

    # Saving the classification report
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    classificationreport = pd.DataFrame(report).transpose()
    classificationreport.loc[:, 'run_id'] = st
    classificationreport = classificationreport.reset_index()
    classificationreport = classificationreport.rename(columns={classificationreport.columns[0]: 'classes'})
    global model_quality
    model_quality += '\t' + str(classificationreport.iloc[2]['precision'])  # accuracy
    model_quality += '\t' + str(classificationreport.iloc[3]['precision'])  # P_macro
    model_quality += '\t' + str(classificationreport.iloc[3]['recall'])     # R_macro
    model_quality += '\t' + str(classificationreport.iloc[3]['f1-score'])   # F1_macro
    model_quality += '\t' + str(classificationreport.iloc[4]['precision'])  # P_micro
    model_quality += '\t' + str(classificationreport.iloc[4]['recall'])     # R_micro
    model_quality += '\t' + str(classificationreport.iloc[4]['f1-score'])   # F1_micro

    utils.pbar.set_description('Preparing Plots Data', refresh=True)
    bool_feature = []
    for feature in new_sampled_data.columns:
        values = new_sampled_data[feature].unique()
        if len(values) == 2:
            values = sorted(values)
            if values[0] == 0 and values[1] == 1:
                bool_feature.append(feature)

    viz = dtreeviz_lib.dtreeviz(best_clf, new_sampled_data, sampled_target['class'], target_name='class',
                                feature_names=feature_names, class_names=classes, fancy=True,
                                show_root_edge_labels=True, bool_feature=bool_feature)
    results['dtree'] = viz
    utils.pbar.update(1)

    return new_sampled_data, best_clf, results


def read_KG(input_data, st):
    """A modified version of InterpretME's read_KG() method."""
    endpoint = input_data['Endpoint']
    independent_var = []
    dependent_var = []
    classes = []
    class_names = []
    definition = []

    seed_var = input_data['Index_var']
    sampling = input_data['sampling_strategy']
    cv = input_data['cross_validation_folds']
    test_split = input_data['test_split']
    num_imp_features = input_data['number_important_features']
    train_model = input_data['model']
    min_max_depth = input_data.get('min_max_depth', 4)
    max_max_depth = input_data.get('max_max_depth', 6)

    # Create the dataset generating query
    query_select_clause = "SELECT "
    query_where_clause = """WHERE { """
    for k, v in input_data['Independent_variable'].items():
        independent_var.append(k)
        query_select_clause = query_select_clause + "?" + k + " "
        query_where_clause = query_where_clause + v
        definition.append(v)

    for k, v in input_data['Dependent_variable'].items():
        dependent_var.append(k)
        query_select_clause = query_select_clause + "?" + k + " "
        query_where_clause = query_where_clause + v
        target_name = k
        definition.append(v)
        query_where_clause = query_where_clause + "}"

    sparqlQuery = query_select_clause + " " + query_where_clause

    def hook(results):
        bindings = [{key: value['value'] for key, value in binding.items()}
                    for binding in results['results']['bindings']]
        df = pd.DataFrame.from_dict(bindings)
        for column in df.columns:
            df[column] = df[column].str.rsplit('/', n=1).str[-1]
        return df

    shacl_engine_communicator = ReducedTravshaclCommunicator(
        '', endpoint,
        {
            "backend": "travshacl",
            "start_with_target_shape": use_heuristics,
            "replace_target_query": False,
            "prune_shape_network": use_heuristics,
            "output_format": "simple",
            "outputs": True
        }
    )

    base_dataset = BaseDataset.from_knowledge_graph(endpoint, shacl_engine_communicator, sparqlQuery,
                                                    target_name, seed_var=seed_var,
                                                    raw_data_query_results_to_df_hook=hook)

    constraints = [ShaclSchemaConstraint.from_dict(constraint) for constraint in input_data['Constraints']] if validation else []

    utils.pbar.total += len(constraints)
    utils.pbar.set_description('SHACL Validation', refresh=True)
    shacl_validation_results = base_dataset.get_shacl_schema_validation_results(
            constraints, rename_columns=True, replace_non_applicable_nans=True
    ) if validation else pd.DataFrame([])
    utils.pbar.update(len(constraints))

    sample_to_node_mapping = base_dataset.get_sample_to_node_mapping().rename('node')

    annotated_dataset = pd.concat(
        (base_dataset.df, shacl_validation_results, sample_to_node_mapping), axis='columns'
    )

    annotated_dataset = annotated_dataset.drop_duplicates()
    annotated_dataset = annotated_dataset.set_index(seed_var)

    for k, v in input_data['classes'].items():
        classes.append(v)
        class_names.append(k)

    annotated_dataset = annotated_dataset.drop(columns=['node'])
    if validation and not consider_val_res_feature:
        num = len(input_data['Constraints'])
        annotated_dataset = annotated_dataset.iloc[:, :-num]

    return seed_var, independent_var, dependent_var, classes, class_names, annotated_dataset, constraints, base_dataset, st, input_data['3_valued_logic'], sampling, test_split, num_imp_features, train_model, cv, min_max_depth, max_max_depth


def modified_pipeline(path_config):
    """A modified version of the pipeline() function from InterpretME.

    The modified version keeps the sample to node mapping so that the user can actually
    analyze the anomalies observed in the decision trees. Additionally, since this script
    is about the validating models' capability of understanding the anomalies, no KG
    with the metadata from the generation of the predictive model is created.
    """
    st = current_milli_time()
    results = {'run_id': st}

    if not os.path.exists('interpretme/files'):
        os.makedirs('interpretme/files')

    utils.pbar = utils.tqdm(total=3, miniters=1, desc='InterpretME Pipeline', unit='task', leave=False)
    with open(path_config, 'r') as input_file_descriptor:
        input_data = json.load(input_file_descriptor)

    utils.pbar.set_description('Read input data', refresh=True)
    seed_var, independent_var, dependent_var, classes, class_names, annotated_dataset, constraints, base_dataset, st, non_applicable_counts, sampling, test_split, imp_features, model, cv, min_max_depth, max_max_depth = read_KG(input_data, st)
    utils.pbar.update(1)
    utils.pbar.set_description('Preprocessing', refresh=True)
    encoded_data, encode_target = preprocessing_data.load_data(seed_var, dependent_var, classes, annotated_dataset)
    utils.pbar.update(1)
    utils.pbar.set_description('Sampling', refresh=True)
    sampled_data, sampled_target, results = sampling_strategy.sampling_strategy(encoded_data, encode_target, sampling, results)
    utils.pbar.update(1)

    new_sampled_data, clf, results = binary_classification(sampled_data, sampled_target, imp_features, cv, classes, st, test_split, model, results, min_max_depth, max_max_depth)
    processed_df = pd.concat((new_sampled_data, sampled_target), axis='columns')
    processed_df.reset_index(inplace=True)

    utils.pbar.total += 1
    utils.pbar.set_description('Preparing Plots Data', refresh=True)
    processed_dataset = ProcessedDataset.from_node_unique_columns(
        processed_df,
        base_dataset,
        base_columns=[seed_var],
        target_name='class',
        categorical_mapping={'class': {i: classes[i] for i in range(len(classes))}}
    )
    checker = Checker(clf.predict, processed_dataset)

    shadow_tree = get_shadow_tree_from_checker(clf, checker)

    checker.validate(constraints)
    results['checker'] = checker
    results['shadow_tree'] = shadow_tree
    results['constraints'] = constraints
    results['non_applicable_counts'] = non_applicable_counts
    utils.pbar.update(1)

    rmtree('interpretme', ignore_errors=True) # delete the LIME results
    return results


def plot_results(results, path):
    dtree = results['dtree']
    dtree.save(os.path.join(path, 'dtree.svg'))

    checker = results['checker']
    shadow_tree = results['shadow_tree']
    constraints = results['constraints']
    non_applicable_counts = results['non_applicable_counts']

    # overall
    plot = constraint_viz.dtreeviz(
        shadow_tree, checker, constraints, coverage=True, non_applicable_counts=non_applicable_counts
    )
    plot.save(os.path.join(path, 'dtree_val_all.svg'))

    # each constraint separately
    for i, constraint in enumerate(constraints, start=1):
        plot = constraint_viz.dtreeviz(
            shadow_tree, checker, [constraint], coverage=False, non_applicable_counts=non_applicable_counts
        )
        plot.save(os.path.join(path, f'dtree_val_{i}.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executing ConstrainTree.')
    parser.add_argument('config_path', type=str, help='Path to the ConstrainTree config file.')
    parser.add_argument('-v', '--validation_as_feature', action='store_true', help='Consider validation results as feature.', default=False)
    parser.add_argument('-o', '--heuristics', action='store_true', help='Apply the proposed heuristics for the constraint validation.', default=False)
    parser.add_argument('-n', '--no_validation', action='store_false', help='Disables the integrity constraint validation.', default=True)
    parser.add_argument('-p', '--plot_trees', action='store_true', help='Plot the decision tree trained.', default=False)
    args = parser.parse_args()

    config_path = args.config_path
    consider_val_res_feature = args.validation_as_feature
    use_heuristics = args.heuristics
    validation = args.no_validation  # this is correct, the 'no' is just because of the flags name

    start = time()
    result = modified_pipeline(config_path)
    end = time()
    duration = end - start

    config_name = os.path.basename(config_path)[:-5]
    if consider_val_res_feature:
        config_name += '_val-feature'
    if use_heuristics:
        config_name += '_heuristics'
    if not validation:
        config_name += '_no-validation'

    logger.info(config_name + '\t' + str(duration) + model_quality)

    if args.plot_trees:
        output_path = os.path.join('output', config_name)
        os.makedirs(output_path, exist_ok=True)
        plot_results(result, output_path)
