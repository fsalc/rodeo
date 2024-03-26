import { ActionIcon, Center, Group, NumberInput, Select, TextInput } from "@mantine/core"
import { IconPlus, IconTrash } from "@tabler/icons-react"

function Constraint({ constraint, attributes, domains, handleDelete, handleModify }) {
  return (
    <Group>
      <Select onChange={handleModify('attribute')} style={{width: 200}} placeholder="Attribute" data={attributes} label="Attribute" value={constraint.attribute} searchable />
      <Select onChange={handleModify('value')} placeholder="Value" label="Value" value={constraint.value} data={domains[constraint.attribute]} searchable />
      <Center>
        <ActionIcon mt="24px" variant="default" radius="xl" onClick={handleDelete}>
          <IconPlus size={14}/>
        </ActionIcon>
      </Center>
      <Select onChange={handleModify('operator')} style={{width: 100}} placeholder="Op." data={['>=', '<=']} label="Operator" value={constraint.operator} />
      <NumberInput onChange={handleModify('cardinality')} style={{width: 100}} placeholder="Card." label="Cardinality" value={constraint.cardinality} min={0}/>
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
