import { useState, useEffect } from 'react';
import { ActionIcon, Center, Group, NumberInput, Select, TextInput, Popover, Stack, Box, Divider } from "@mantine/core";
import { IconPlus, IconTrash } from "@tabler/icons-react";
import { getDistribution } from "../provider";
import Plot from 'react-plotly.js';

function Constraint({ dataset, constraint, handleDelete, handleModify, addGroup, modifyGroup }) {
  const [focused, setFocused] = useState(false);
  const [plotData, setPlotData] = useState({ids: [], labels: [], parents: [], values: []});

  const updateDistribution = (attrs) => {
    getDistribution(dataset.dataset, attrs.map(attr => attr.attribute)).then((distribution) => {
      setPlotData(distribution);
    }).catch((e) => {
      console.error(e);
    })
  }

  useEffect(() => {
    updateDistribution(constraint.groups);
  }, [constraint])

  return (
    <Group>
      <Stack>
        {constraint.groups.map((group) =>
          <Group key={group.id}>
            <Select onChange={(e) => {
              console.log(modifyGroup(constraint.groups, group.id, 'attribute')(e));
              handleModify('groups')(modifyGroup(constraint.groups, group.id, 'attribute')(e))
            }} 
              style={{width: 200}} placeholder="Attribute"
              data={dataset.attributes} 
              label="Attribute" 
              value={group.attribute} 
              searchable
            />
            <Select onChange={(e) => handleModify('groups')(modifyGroup(constraint.groups, group.id, 'value')(e))} 
              placeholder="Value" label="Value" 
              value={group.value} 
              data={dataset.domains[group.attribute]} 
              searchable 
            />
          </Group>
        )}
      </Stack>
      <Center>
        <ActionIcon mt="24px" variant="default" radius="xl" onClick={() => handleModify('groups')(addGroup(constraint.groups))}>
          <IconPlus size={14}/>
        </ActionIcon>
      </Center>
      <Select onChange={handleModify('operator')} style={{width: 100}} placeholder="Op." data={['>=', '<=']} label="Operator" value={constraint.operator} />
      <Popover opened={focused} position="bottom" withArrow>
        <Popover.Target>
          <NumberInput onChange={handleModify('cardinality')} style={{width: 100}} placeholder="Card." label="Cardinality" value={constraint.cardinality} min={0} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)} />
        </Popover.Target>
        <Popover.Dropdown>
            <Plot
              data={[
                {
                  type: 'sunburst',
                  ids: plotData.ids,
                  labels: plotData.labels,
                  parents: plotData.parents,
                  values: plotData.values,
                  insidetextorientation: 'horizontal',
                  branchvalues: 'total'
                }
              ]}
              layout={ {width: 160, height: 120, margin: {l: 0, r: 0, t: 0, b: 0}} }
            />
        </Popover.Dropdown>
      </Popover>
      <NumberInput onChange={handleModify('k')} style={{width: 100}} placeholder="Top-k" label="Top-k" value={constraint.k} min={0}/>
      <Center>
        <ActionIcon mt="24px" variant="default" radius="xl" onClick={handleDelete}>
          <IconTrash size={14}/>
        </ActionIcon>
      </Center>
    </Group>
  )
}

export default Constraint
