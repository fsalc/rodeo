import { useEffect, useState } from 'react';
import { AppShell, Button, Center, Container, Group, Image, Loader, Stepper, Text } from '@mantine/core';
import { IconChevronLeft, IconChevronsRight, IconSettings } from '@tabler/icons-react';

import logo from "./assets/rodeo.png";

import { getDatasets, getMetadata, postRefinement } from './provider';
import { buildQuery } from './util';
import Query from './components/Query';
import QueryBuilder from "./components/QueryBuilder";
import ConstraintsBuilder from './components/ConstraintsBuilder';
import Refinement from "./components/Refinement";
import DataViewer from './components/DataViewer';

let nextId = 0;

function App() {
  // Application state
  const [datasets, setDatasets] = useState(['Loading...'])
  const [dataset, setDataset] = useState({dataset: '', attributes: []})
  const [conditions, setConditions] = useState([])
  const [ranking, setRanking] = useState({ranking: '', order: ''})
  const [constraints, setConstraints] = useState([])
  const [refinement, setRefinement] = useState({eps: 0, distance: 'Most similar query'})
  const [data, setData] = useState(undefined)

  const addCondition = () => setConditions([...conditions, {id: nextId++, attribute: undefined, operator: undefined, constant: undefined}])
  const deleteCondition = (id) => () => setConditions(conditions.filter(condition => condition.id !== id))
  const modifyCondition = (id) => (part) => (e) => setConditions(conditions.map((condition) => condition.id === id ? { ...condition, [part]: typeof e === 'string' ? e : e.target.value } : condition))

  const modifyRanking = (part) => (e) => setRanking({ ...ranking, [part]: typeof e === 'string' ? e : e.target.value })

  const addConstraint = () => setConstraints([...constraints, {id: nextId++, groups: [{id: nextId++, attribute: undefined, value: undefined}], k: undefined, cardinality: undefined, operator: undefined}])
  const deleteConstraint = (id) => () => setConstraints(constraints.filter(constraint => constraint.id !== id))
  const modifyConstraint = (id) => (part) => (e) => setConstraints(constraints.map((constraint) => constraint.id === id ? { ...constraint, [part]: typeof e === 'string' || typeof e === 'number' || typeof e === 'object' ? e : e.target.value } : constraint))

  const addGroup = (groups) => [...groups, {id: nextId++, attribute: undefined, value: undefined}]
  const modifyGroup = (groups, id, part) => (e) => groups.map((group) => group.id === id ? { ...group, [part]: typeof e === 'string' || typeof e === 'number' ? e : e.target.value } : group)

  const modifyRefinement = (part) => (e) => setRefinement({ ...refinement, [part]: typeof e === 'string' || typeof e === 'number' ? e : e.target.value })

  useEffect(() => {
    getDatasets().then((datasets) => setDatasets(datasets))
  }, [])

  const refreshDatasets = () => getDatasets().then((datasets) => setDatasets(datasets));

  // Stepper state & buttons
  const [active, setActive] = useState(0);
  const nextStep = () => setActive((current) => (current < 3 ? current + 1 : current));
  const prevStep = () => {
    if(active === 3) {
      setData(undefined);
    }
    setActive((current) => (current > 0 ? current - 1 : current))
  };

  const nextStepButton = <Button onClick={nextStep} rightSection={<IconChevronsRight size={14} />}>Next step</Button>
  const prevStepButton = <Button variant="default" onClick={prevStep} leftSection={<IconChevronLeft size={14} />}>Back</Button>

  const refine = () => {
    nextStep();
    postRefinement(buildQuery(dataset, conditions, ranking, true), constraints, refinement.eps, refinement.distance).then((data) => {
      setData(data)
      console.log(data)
    }).catch((e) => {
      alert(e)
    })
  }

  const refineButton = <Button onClick={refine} rightSection={<IconSettings size={14} />}>Refine</Button>

  return (
    <AppShell header={{ height: 80 }} footer={{ height: 40 }} padding="md">
      <AppShell.Header>
        <Container>
          <Center>
            <Image src={logo} style={{width: 140}} mt="md"/>
          </Center>
        </Container>
      </AppShell.Header>
      <AppShell.Main padding="md">
        <Container>
          <Stepper active={active} onStepClick={setActive}>
            <Stepper.Step label="Build query">
              <Query query={buildQuery(dataset, conditions, ranking)} />
              <QueryBuilder datasets={datasets} dataset={dataset} setDataset={(value) => getMetadata(value).then((dataset) => setDataset(dataset))} conditions={conditions} addCondition={addCondition} deleteCondition={deleteCondition} modifyCondition={modifyCondition} ranking={ranking} modifyRanking={modifyRanking} refreshDatasets={refreshDatasets} />
            </Stepper.Step>
            <Stepper.Step label="Set constraints">
              <Query query={buildQuery(dataset, conditions, ranking)} />
              <ConstraintsBuilder dataset={dataset} query={buildQuery(dataset, conditions, ranking)} constraints={constraints} handleAdd={addConstraint} handleDelete={deleteConstraint} handleModify={modifyConstraint} addGroup={addGroup} modifyGroup={modifyGroup} />
            </Stepper.Step>
            <Stepper.Step label="Configure refinement">
              <Query query={buildQuery(dataset, conditions, ranking)} />
              <Refinement query={buildQuery(dataset, conditions, ranking)} constraints={constraints} refinement={refinement} handleModify={modifyRefinement} />
            </Stepper.Step>
            <Stepper.Completed>
              <Center>
                  <DataViewer originalConditions={conditions} constraints={constraints} data={data} />
              </Center>
            </Stepper.Completed>
          </Stepper>

          <Center mt="md">
            <Group>
              {prevStepButton}
              {active == 2 ? refineButton : active != 3 ? nextStepButton : ''}
            </Group>
          </Center>
        </Container>
      </AppShell.Main>
      <AppShell.Footer>
        <Container>
          <Text c="dimmed">Database Group at Ben-Gurion University of the Negev</Text>
        </Container>
      </AppShell.Footer>
    </AppShell>
  )
}

export default App
