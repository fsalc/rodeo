import { Fieldset, Group, Select } from '@mantine/core';
import FileUploader from './FileUploader';

function DatasetSelector({ datasets, dataset, handleModify, refreshDatasets }) {
    return (
      <Fieldset legend="Dataset" mb="md">
        <Group>
          <Select onChange={handleModify} placeholder="Dataset" data={datasets} label="Dataset" value={dataset} />
          <FileUploader refreshDatasets={refreshDatasets} />
        </Group>
      </Fieldset>
    )
  }
  
  export default DatasetSelector