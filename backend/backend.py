from flask import Flask, request
from flask_cors import CORS
from ranking_refinements.fair import Ranking, Constraints, Constraint, Group, RefinementMethod, UsefulMethod
import duckdb as d
import numpy as np
import glob

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
        constraints.append(Constraint(Group(constraint['attribute'], constraint['value']), (constraint['k'], constraint['cardinality']), sense='L' if constraint['operator'] == '>=' else 'U'))
    constraints = Constraints(*constraints)
    refinement, times = ranking.refine(constraints, only_lower_bound_card_constraints=False, max_deviation=data['eps'], useful_method=methods[data['distance']])
    if refinement:
        refined_query = str(ranking.query.where(str(refinement.conditions), append=False))
        print(refinement.conditions)
    return {'query': refined_query, 'data': diff(str(ranking.query), refined_query, constraints.p_star)} if refinement else {'query': 'None', 'data': None}

@app.route('/')
def hello():

    constraints = Constraints(
        Constraint(Group("sex", "F"), (10, 5)),
        Constraint(Group("address", "R"), (10, 5)),
    )

    ranking = Ranking(
        'SELECT * FROM "data/student-por.csv" WHERE guardian = \'other\' AND famrel >= 1 AND famrel <= 3 AND Dalc >= 3 AND Dalc <= 5 ORDER BY G1+G2+G3 DESC')

    refinement, times = ranking.refine(constraints, only_lower_bound_card_constraints=True, max_deviation=0, debug=True, method=RefinementMethod.MILP_OPT)
    return str(refinement.conditions)
