import { Button, Fieldset, Stack } from '@mantine/core'
import { IconPlus } from '@tabler/icons-react'

import Constraint from './Constraint'

function ConstraintsBuilder({ dataset, constraints, handleAdd, handleDelete, handleModify }) {
    return (
      <>
      <Fieldset legend="Constraints">
        <Stack>
          <Stack>
            {constraints.map(constraint => <Constraint key={constraint.id} dataset={dataset} constraint={constraint} handleDelete={handleDelete(constraint.id)} handleModify={handleModify(constraint.id)} />)}
          </Stack>
          <div>
            <Button leftSection={<IconPlus size={14} />} variant="light" onClick={handleAdd}>
              Add
            </Button>
          </div>
        </Stack>
      </Fieldset>
      </>
    )
  }
  
export default ConstraintsBuilder
