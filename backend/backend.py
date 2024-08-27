from flask import Flask, request
from flask_cors import CORS
from ranking_refinements.fair import Ranking, Constraints, Constraint, Group, RefinementMethod, UsefulMethod
from werkzeug.utils import secure_filename
import duckdb as d
import numpy as np
import glob
import os

app = Flask(__name__)
CORS(app)

methods = {
    'Most similar query': UsefulMethod.QUERY_DISTANCE,
    'Most similar set': UsefulMethod.MAX_ORIGINAL,
    'Most similar ranking': UsefulMethod.KENDALL_DISTANCE
}

@app.route('/datasets')
def datasets():
    return [dataset.split('/')[-1] for dataset in glob.glob('./data/*.csv')]

@app.route('/dataset/<dataset>')
def dataset(dataset):
    attributes = list(set(d.sql(f'SELECT * FROM "./data/{dataset}"').fetchnumpy().keys()))
    def domain(attribute):
        return d.sql(f'SELECT DISTINCT {attribute} FROM "./data/{dataset}"').fetchnumpy()[attribute].filled()
    return {'dataset': dataset, 'attributes': attributes, 'domains': {attribute: np.sort(domain(attribute)).tolist() for attribute in attributes}}

# TODO: send undefined instead of NULL string
@app.route('/distribution/<dataset>')
def distribution(dataset):
    '''Returns the distribution of the dataset according to hierarchy of attributes. Formatted for use with Plotly sunburst chart.'''
    attrs = request.args.getlist('attr')
    if not attrs:
        return {"error": "`attr` query parameter must be specified"}, 400
    groupby = f'ROLLUP ({", ".join(attrs)})'

    distr = {'ids': [], 'labels': [], 'parents': [], 'values': []}

    con = d.connect()
    df = con.sql(f'''SELECT {", ".join(attrs) if attrs else '*'}, COUNT(*) AS count FROM "./data/{dataset}" GROUP BY {groupby}''').df().fillna('NULL')
    con.close()
    for index, row in df.iterrows():
        # Every attribute after most_specific_attr is null, i.e. not grouped on
        # Note that after this, all attrs are NULL since we perform a ROLL UP
        # level specifies the level of the hierarchy most_specific_attr is at
        most_specific_attr, level = None, -1
        for attr in attrs:
            if row[attr] != 'NULL':
                most_specific_attr = attr
                level += 1

        # Discard grand total
        if most_specific_attr == None:
            continue
    
        # Hierarchy of each value to use as ID
        distr['ids'].append(' - '.join([str(row[attr]) for attr in attrs]))

        # Label like 'attr = val', where attr is the most specific attribute set (i.e. last in order of `attrs` that is not null)
        distr['labels'].append(f'{most_specific_attr} = {row[most_specific_attr]}')

        if level == 0:
            distr['parents'].append('')
        else:
            distr['parents'].append(' - '.join([str(row[attr]) if attr != most_specific_attr else 'NULL' for attr in attrs]))

        distr['values'].append(row['count'])

    return distr

def diff(original, refinement, k_star):
    utility = Ranking(refinement).utility
    return d.sql(f'''SELECT * FROM (SELECT *, '-' AS __diff FROM (({original} LIMIT {k_star}) EXCEPT ({refinement} LIMIT {k_star})) UNION ALL
    SELECT *, '+' AS __diff FROM (({refinement} LIMIT {k_star}) EXCEPT ({original} LIMIT {k_star})) UNION ALL
    SELECT *, '*' AS __diff FROM (({refinement} LIMIT {k_star}) INTERSECT ({original} LIMIT {k_star}))) {utility}''').df().fillna('NULL').to_dict(orient='records')

@app.route('/refine', methods=['POST'])
def refine():
    data = request.get_json()
    print(data)
    ranking = Ranking(data['query'])
    constraints = []
    for constraint in data['constraints']:
        attrs, vals = tuple(g['attribute'] for g in constraint['groups']), tuple(g['value'] for g in constraint['groups'])
        print(attrs, vals)
        constraints.append(Constraint(Group(attrs, vals), (constraint['k'], constraint['cardinality']), sense='L' if constraint['operator'] == '>=' else 'U'))
    constraints = Constraints(*constraints)
    refinement, times = ranking.refine(constraints, only_lower_bound_card_constraints=False, max_deviation=data['eps'], useful_method=methods[data['distance']])
    if refinement:
        refined_query = str(ranking.query.where(str(refinement.conditions), append=False))
        conditions = [condition.to_dict() for condition in refinement.conditions]
        print(refinement.conditions)
    return {'query': refined_query, 'conditions': conditions, 'data': diff(str(ranking.query), refined_query, constraints.p_star)} if refinement else {'query': 'None', 'conditions': [], 'data': None}

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    
    if file:
        file.save(os.path.join('data/', secure_filename(file.filename)))
        return {'datasets': datasets()}
        
    return {'error': 'File upload failure'}, 500

# @app.route('/')
# def example():
#     constraints = Constraints(
#         Constraint(Group("sex", "F"), (10, 5)),
#         Constraint(Group("address", "R"), (10, 5)),
#     )

#     ranking = Ranking(
#         'SELECT * FROM "data/student-por.csv" WHERE guardian = \'other\' AND famrel >= 1 AND famrel <= 3 AND Dalc >= 3 AND Dalc <= 5 ORDER BY G1+G2+G3 DESC')

#     refinement, times = ranking.refine(constraints, only_lower_bound_card_constraints=True, max_deviation=0, debug=True, method=RefinementMethod.MILP_OPT)
#     return str(refinement.conditions)
