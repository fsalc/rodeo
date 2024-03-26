import { Button, Fieldset, Stack } from '@mantine/core'
import { IconPlus } from '@tabler/icons-react'

import Condition from './Condition'

function ConditionsBuilder({ dataset, conditions, handleAdd, handleDelete, handleModify }) {
    return (
      <Fieldset legend="Requirements" mb="md">
        <Stack>
          <Stack>
            {conditions.map(condition => <Condition key={condition.id} condition={condition} attributes={dataset.attributes} handleDelete={handleDelete(condition.id)} handleModify={handleModify(condition.id)} />)}
          </Stack>
          <div>
            <Button leftSection={<IconPlus size={14} />} variant="light" onClick={handleAdd}>
              Add
            </Button>
          </div>
        </Stack>
      </Fieldset>
    )
  }
  
  export default ConditionsBuilder