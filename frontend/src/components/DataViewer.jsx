import { Badge, Center, Container, Loader, ScrollArea, Text, Table, Alert, Stack, Divider, Title } from "@mantine/core"
import { IconDatabase, IconInfoCircle } from "@tabler/icons-react";
import Query from "./Query";
import Summary from "./Summary";

let dataId = 0;

const rowProps = (row) => {
    if(row.__diff === '-') {
        return {
            c: 'red',
            td: 'line-through'
        }
    } else if(row.__diff === '+') {
        return {
            c: 'green'
        }
    }
    return {}
}

const rowGroups = (row, groups) => {
    const badges = groups.filter((group) => satisfiesGroup(row, group)).map((group) => {
        const groupStrs = group.attributes.map((attribute, index) => `${attribute} = ${group.values[index]}`)
        return (<Badge color="blue">{groupStrs.join(" ∧ ")}</Badge>)
    })
    return badges.length === 0 ? <Center>•</Center> : badges
}

const constraintGroups = (constraint) => {
    let attributes = [];
    let values = [];
    constraint.groups.forEach((group) => {
        attributes.push(group.attribute);
        values.push(group.value)
    });
    return {attributes: attributes, values: values}
}

// TODO: combine with satisfiesConstraint in Summary component
const satisfiesGroup = (row, group) => {
    let match = true;
    let {attributes, values} = group;
    attributes.forEach((attribute, i) => {
        if(row[attribute] != values[i]) {
            match = false;
        }
    })
    return match;
}

function DataViewer({ originalConditions, constraints, data }) {

    if(!data) {
        return <Loader color="blue" />
    }

    const rowsData = data.data
    const query = data.query
    const refinedConditions = data.conditions
    const groups = constraints.map((constraint) => constraintGroups(constraint))

    if(query !== 'None') {
        const attributes = Object.keys(rowsData[0]).map((attribute) => attribute !== '__diff' ? <Table.Th>{attribute}</Table.Th> : <></>)
        const rowToColumns = (row) => [<Table.Td>{rowGroups(row, groups)}</Table.Td>].concat(Object.entries(row).map(([attribute, value]) => attribute !== '__diff' ? <Table.Td><Text {...rowProps(row)}>{value}</Text></Table.Td> : <></>))
        const rows = rowsData.map((row) => <Table.Tr>{rowToColumns(row)}</Table.Tr>)

        return (
            <Container>
                <Stack>
                    <div>
                        <Alert variant="light" color="blue" title="Query successfully refined" icon={<IconInfoCircle />} />
                    </div>
                    <Query query={query} />
                    <Summary constraints={constraints} originalConditions={originalConditions} refinedConditions={refinedConditions} data={rowsData} />
                    <Divider />
                    <Center>
                        <Title order={3}><IconDatabase size={18} />&nbsp;&nbsp;Data Difference Viewer</Title>
                    </Center>
                    <ScrollArea>
                        <Table>
                            <Table.Thead>
                                <Table.Tr>
                                    <Table.Th><Badge color="grey">Group</Badge></Table.Th>
                                    {attributes}
                                </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                                {rows}
                            </Table.Tbody>
                        </Table>
                    </ScrollArea>
                </Stack>
            </Container>
        )
    } else {
        return (
            <Container>
                <Alert variant="light" color="red" title="No refinement exists for given input" icon={<IconInfoCircle />} />
            </Container>
        )
    }
}

export default DataViewer