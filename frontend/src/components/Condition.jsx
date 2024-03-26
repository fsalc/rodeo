import { ActionIcon, Center, Group, Select, TextInput } from "@mantine/core"
import { IconTrash } from "@tabler/icons-react"

function Condition({ condition, attributes, handleDelete, handleModify }) {
  return (
    <Group>
      <Select onChange={handleModify('attribute')} placeholder="Attribute" data={attributes} label="Attribute" value={condition.attribute} searchable />
      <Select onChange={handleModify('operator')} placeholder="Operator" data={['>', '>=', '<', '<=', '=', 'IN']} label="Operator" value={condition.operator} />
      <TextInput placeholder="Constant" onChange={handleModify('constant')} label="Constant" value={condition.constant} />
      <ActionIcon mt="24px" variant="default" radius="xl" onClick={handleDelete}>
        <IconTrash size={14}/>
      </ActionIcon>
    </Group>
  )
}

export default Condition
