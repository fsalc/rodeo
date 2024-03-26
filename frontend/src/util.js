const conditionsToString = (conditions) => conditions.map((condition) => {
    if(condition.operator && condition.operator === 'IN') {
        if(condition.constant) {
            // TODO: Very fragile
            let r = condition.constant.replace('(', '[').replace(')', ']').replaceAll("'", '"');
            try {
                console.log(r);
                let c = JSON.parse(r).map(v => `${condition.attribute || '...'} = '${v}'`).join(' OR ')
                return `(${c})`
            } catch(e) {
                `${condition.attribute || '...'} ${condition.operator || '...'} ${condition.constant || '...'}`
            }
        } else {
            `${condition.attribute || '...'} ${condition.operator || '...'} ${condition.constant || '...'}`
        }
    }
    return `${condition.attribute || '...'} ${condition.operator || '...'} ${condition.constant || '...'}`
}).join(' AND ')
const buildQuery = (dataset, conditions, ranking, wire) => `SELECT * FROM "${wire ? 'data/' : ''}${dataset.dataset || '...'}" WHERE ${conditions.length !== 0 ? conditionsToString(conditions) : '...'} ORDER BY ${ranking.ranking || '...'} ${ranking.order || '...'}`

export { conditionsToString, buildQuery }