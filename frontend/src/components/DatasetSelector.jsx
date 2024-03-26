import { ActionIcon, Button, Fieldset, Group, Select } from '@mantine/core';
import { IconUpload } from '@tabler/icons-react';

function DatasetSelector({ datasets, dataset, handleModify }) {
    return (
      <Fieldset legend="Dataset" mb="md">
        <Group>
          <Select onChange={handleModify} placeholder="Dataset" data={datasets} label="Dataset" value={dataset} />
          <Button mt="24px" leftSection={<IconUpload size={14} />} variant="light">
            Upload
          </Button>
        </Group>
      </Fieldset>
    )
  }
  
  export default DatasetSelector