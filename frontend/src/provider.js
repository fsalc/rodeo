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
        if (dataset === 'Loading...') {
            resolve({ dataset: dataset, attributes: ['Loading'], domains: { 'Loading...': ['Loading...'] } })
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
            body: JSON.stringify({ query: query, constraints: constraints, eps: epsilon, distance: method })
        }).then((response) => response.json()).then((data) => {
            resolve(data)
        }).catch((error) => {
            reject(error)
        })
    })
}

function uploadFile(file, updateProgress) {
    return new Promise((resolve, reject) => {
        const data = new FormData();
        data.append('file', file);

        const request = new XMLHttpRequest();
        request.open('POST', 'http://127.0.0.1:5000/upload', true);

        request.upload.onprogress = (event) => {
            if (event.lengthComputable) {
                const percentCompleted = Math.round(event.loaded / event.total * 100);
                updateProgress(percentCompleted);
            }
        };

        request.onload = () => {
            if (request.status === 200) {
                resolve();
            } else {
                reject();
            }
        };

        request.onerror = () => {
            reject();
        };

        request.send(data);
    })
}

export { getDatasets, getMetadata, postRefinement, uploadFile }
