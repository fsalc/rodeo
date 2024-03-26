import { Badge, Center, Container, Loader, ScrollArea, Text, Table, Alert, Stack } from "@mantine/core"
import { IconInfoCircle } from "@tabler/icons-react";
import Query from "./Query";

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
    const badges = groups.filter((group) => row[group.attribute] == group.value).map((group) => <Badge color="blue">{group.attribute} = {group.value}</Badge>)
    return badges.length === 0 ? <Center>•</Center> : badges
}

function DataViewer({ constraints, data }) {

    if(!data) {
        return <Loader color="blue" />
    }

    const rowsData = data.data
    const query = data.query
    const groups = constraints.map((constraint) => ({attribute: constraint.attribute, value: constraint.value}))

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