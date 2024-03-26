import { TextInput, Select, Group, Fieldset } from "@mantine/core"

function Ranking({ ranking, handleModify }) {
    return (
      <Fieldset legend="Ranking">
        <Group>
          <TextInput placeholder="Ranking expression" onChange={handleModify('ranking')} label="Expression" value={ranking.ranking} />
          <Select placeholder="Order" onChange={handleModify('order')} data={['ASC', 'DESC']} label="Order" value={ranking.order} />
        </Group>
      </Fieldset>
    )
  }
  
  export default Ranking
  