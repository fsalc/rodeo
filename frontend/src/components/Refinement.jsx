import { Center, Fieldset, SegmentedControl, Slider, Table, Text } from "@mantine/core"
import ConstraintsViewer from "./ConstraintsViewer"

function Refinement({ query, constraints, refinement, handleModify }) {
    const maxEpsilon = Math.max(...constraints.map((constraint) => constraint.operator === ">=" ? 1 : (constraint.k - constraint.cardinality) / constraint.cardinality))

    return (
        <>
        <Fieldset legend="Allowable deviation from constraints" mb="md">
            <ConstraintsViewer constraints={constraints} epsilon={refinement.eps} />
            <Text size="sm" mt="sm">Maximum average deviation from constraints</Text>
            <Slider onChange={handleModify('eps')} defaultValue={0} min={0} max={maxEpsilon} step={0.01} label={(value) => `${value.toFixed(2)}`} value={refinement.eps} style={{width: 460}} mb="sm"/>
        </Fieldset>

        <Fieldset legend="Refinement properties">
            <Center>
                <SegmentedControl onChange={handleModify('distance')} color="blue" data={['Most similar query', 'Most similar set', 'Most similar ranking']} value={refinement.distance} />
            </Center>
        </Fieldset>
        </>
    )
}

export default Refinement