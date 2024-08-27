import { Button, Fieldset, Stack, Divider } from '@mantine/core'
import { IconPlus } from '@tabler/icons-react'

import Constraint from './Constraint'

function ConstraintsBuilder({ dataset, constraints, handleAdd, handleDelete, handleModify, addGroup, modifyGroup }) {
    return (
      <>
      <Fieldset legend="Constraints">
        <Stack>
          <Stack>
            {constraints.map((constraint, index) => 
              <>
                <Constraint key={constraint.id} dataset={dataset} constraint={constraint} handleDelete={handleDelete(constraint.id)} handleModify={handleModify(constraint.id)} addGroup={addGroup} modifyGroup={modifyGroup} />
                {index !== constraints.length - 1 && <Divider mt="sm" mb="sm" />} 
              </>
            )}
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
