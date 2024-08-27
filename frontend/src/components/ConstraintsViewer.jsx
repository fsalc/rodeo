import { Badge, RangeSlider, Table } from "@mantine/core"

function ConstraintsViewer({ constraints, epsilon }) {

    const constraintRows = constraints.map(constraint => {
        let lower = epsilon ? Math.max(Math.ceil(constraint.cardinality - epsilon * constraints.length * constraint.cardinality), 0) : constraint.cardinality
        let upper = epsilon ? Math.min(Math.floor(constraint.cardinality + epsilon * constraints.length * constraint.cardinality), constraint.k) : constraint.cardinality
        let relaxed = constraint.operator === '<=' ? upper : lower;
        return (
            <Table.Tr key={constraint.id}>
                <Table.Td><Badge color="blue">{constraint.groups.map(group => `${group.attribute} = ${group.value}`).join(" ∧ ")}</Badge></Table.Td>
                <Table.Td>{constraint.operator === '<=' ? 'At most' : 'At least'} {constraint.cardinality} in top-{constraint.k}</Table.Td>
                <Table.Td><RangeSlider max={constraint.k} value={constraint.operator === '<=' ? [0, upper] : [lower, constraint.k]} marks={[
                    {value: 0, label: '0'}, 
                    {value: constraint.cardinality, label: `${constraint.cardinality}`},
                    {value: relaxed, label: `${relaxed}`}, 
                    {value: constraint.k, label: `${constraint.k}`}]
                    } disabled labelAlwaysOn/></Table.Td>
            </Table.Tr>
        )
    })

    return (
        <Table>
            <Table.Thead>
                <Table.Tr>
                    <Table.Th>Group</Table.Th>
                    <Table.Th>Cardinality / Top-k</Table.Th>
                    <Table.Th>Range&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</Table.Th>
                </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
                {constraintRows}
            </Table.Tbody>
        </Table>
    )
}

export default ConstraintsViewer