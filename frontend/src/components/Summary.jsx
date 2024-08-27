// import { CodeHighlight } from "@mantine/code-highlight"
import { Badge, Center, Grid, Group, List, RangeSlider, Stack, Table, Text, Title } from "@mantine/core";
import { IconCode, IconSettings } from "@tabler/icons-react";

const originalConditionValue = (refinedCondition, originalConditions) => {
    // TODO: robustness, should match exactly 0 or 1 items
    // TODO: standardize naming e.g. attribute vs attr
    const matching = originalConditions.filter((condition) => condition.attribute === refinedCondition.attr && condition.operator === refinedCondition.op)
    if(matching.length > 0) {
        return matching[0].constant
    } else {
        return undefined;
    }
}

const opToString = (op) => {
    if(op === 'IN') {
        return "is one of"
    }
    if(op === '<') {
        return "is less than"
    }
    if(op === '<=') {
        return "is at most"
    }
    if(op === '>') {
        return "is greater than"
    }
    if(op === '>=') {
        return "is at least"
    }
    if(op === '=') {
        return "equals"
    }
}

const opToSign = (op) => {
    if(op === '>=') {
        return '≥'
    }
    if(op === '<=') {
        return '≤'
    }
    return op;
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

const satisfiesConstraint = (row, constraint) => {
    let match = true;
    let {attributes, values} = constraintGroups(constraint);
    attributes.forEach((attribute, i) => {
        if(row[attribute] != values[i]) {
            match = false;
        }
    })
    return match;
}

const countGroups = (constraints, data) => {
    var running = {'all': 0}, count = {};

    // TODO: robust hashing
    constraints.forEach((constraint) => {
        const {attributes, values} = constraintGroups(constraint);
        running[`${attributes}${values}`] = 0
        count[`${attributes}${values}`] = {}
        count[`${attributes}${values}`][constraint.k] = 0
    })

    data.forEach((row) => {
        if(row.__diff !== '-') {
            running['all'] += 1
            constraints.filter((constraint) => satisfiesConstraint(row, constraint)).forEach((constraint) => {
                const {attributes, values} = constraintGroups(constraint);
                running[`${attributes}${values}`] += 1;  
            })
            // If we have seen k tuples, and associated constraint is for k, then store running count giving #group in top-k
            constraints.forEach((constraint) => {
                if(running['all'] === constraint.k) {
                    const {attributes, values} = constraintGroups(constraint);
                    count[`${attributes}${values}`][constraint.k] = running[`${attributes}${values}`];
                }
            })
        }
    })

    console.log(count)
    return count
}

function Summary({ constraints, originalConditions, refinedConditions, data }) {
    const condsSummary = refinedConditions.map((condition) => {
        // TODO: handle diffing original values for categorical attributes
        if(Array.isArray(condition.value)) {
            var values = condition.value.map((value) => {
                return <div><Badge color="blue">{value}</Badge><br/></div>
            });
        } else {
            // Not categorical, i.e. guaranteed to be numeric
            const originalValue = originalConditionValue(condition, originalConditions);
            console.log(condition, originalValue);
            var values = <><Text fw={700}>{ condition.value }</Text>&nbsp;<Text c="dimmed" td="line-through">{ originalValue }</Text></>
        }
        return (
            <Table.Tr>
                <Table.Td>
                    <Center>
                        <Badge color="grey">{condition.attr}</Badge>&nbsp;{opToString(condition.op)}&nbsp;<div>{values}</div>
                    </Center>
                </Table.Td>
            </Table.Tr>
        )
    });

    const groupsCount = countGroups(constraints, data);
    const constraintsSummary = constraints.map((constraint) => {
        const {attributes, values} = constraintGroups(constraint);
        console.log(attributes, values);
        const count = groupsCount[`${attributes}${values}`][constraint.k];
        return (
            <Table.Tr>
                <Table.Td>
                    <Center>
                        <Stack>
                            <Center>
                                <Badge color="blue">{constraint.groups.map(group => `${group.attribute} = ${group.value}`).join(" ∧ ")}</Badge>
                            </Center>
                            <Center>
                                <Text mt={-15} mb={-10}>{opToSign(constraint.operator)}{constraint.cardinality} in top-{constraint.k}</Text>
                            </Center>
                        </Stack>
                    </Center>
                </Table.Td>
                <Table.Td>
                    <Center>
                        <RangeSlider mb={10} max={constraint.k} value={[0, count]} marks={[
                            {value: 0, label: '0'}, 
                            {value: constraint.cardinality, label: `${constraint.cardinality}`},
                            {value: count, label: `${count}`}, 
                            {value: constraint.k, label: `${constraint.k}`}]
                        } disabled labelAlwaysOn style={{width: 300}}/>
                    </Center>
                </Table.Td>
            </Table.Tr>
        )
    })
    
    return (
        <Grid>
            <Grid.Col span={6}>
                <Center>
                    <Title order={3}><IconCode size={18} />&nbsp;&nbsp;Refined Query Summary</Title>
                </Center>
                <Center>
                    <Table verticalSpacing="md">
                        <Table.Tbody>
                            { condsSummary }
                        </Table.Tbody>
                    </Table>
                </Center>
            </Grid.Col>
            <Grid.Col span={6}>
                <Center>
                    <Title order={3}><IconSettings size={18} />&nbsp;&nbsp;Constraint Satisfaction Summary</Title>
                </Center>
                <Center>
                    <Table verticalSpacing="md">
                        <Table.Tbody>
                            { constraintsSummary }
                        </Table.Tbody>
                    </Table>
                </Center>
            </Grid.Col>
        </Grid>
    )
  }
  
export default Summary