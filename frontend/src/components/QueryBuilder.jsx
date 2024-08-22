import { useState } from 'react'
import DatasetSelector from './DatasetSelector'
import Ranking from './Ranking'
import ConditionsBuilder from './ConditionsBuilder';
import Query from './Query'

import { buildQuery } from '../util'

function QueryBuilder({ datasets, dataset, setDataset, conditions, addCondition, deleteCondition, modifyCondition, ranking, modifyRanking, refreshDatasets }) {
  return (
    <>
      <DatasetSelector datasets={datasets} dataset={dataset.dataset} handleModify={setDataset} refreshDatasets={refreshDatasets} />
      <ConditionsBuilder dataset={dataset} conditions={conditions} handleAdd={addCondition} handleModify={modifyCondition} handleDelete={deleteCondition} />
      <Ranking ranking={ranking} handleModify={modifyRanking} />
    </>
  )
}
export default QueryBuilder
