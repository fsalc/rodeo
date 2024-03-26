// stub

function getDatasets() {
    return new Promise((resolve, reject) => {
        fetch('http://127.0.0.1:5000/datasets').then((response) => response.json()).then((datasets) => {
            resolve(datasets)
        }).catch((error) => reject(error))
    });
}

function getMetadata(dataset) {
    return new Promise((resolve, reject) => {
        if(dataset === 'Loading...') {
            resolve({dataset: dataset, attributes: ['Loading'], domains: {'Loading...': ['Loading...']}})
        } else {
            fetch(`http://127.0.0.1:5000/dataset/${dataset}`).then((response) => response.json()).then((dataset) => {
                resolve(dataset)
            }).catch((error) => reject(error))
        }
    })
}

function postRefinement(query, constraints, epsilon, method) {
    return new Promise((resolve, reject) => {
        fetch('http://127.0.0.1:5000/refine', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({query: query, constraints: constraints, eps: epsilon, distance: method})
        }).then((response) => response.json()).then((data) => {
            resolve(data)
        }).catch((error) => {
            reject(error)
        })
    })
}

export { getDatasets, getMetadata, postRefinement }
