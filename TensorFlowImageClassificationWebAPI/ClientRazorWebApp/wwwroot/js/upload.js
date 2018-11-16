const url = 'http://localhost:5000/api/ImageClassification/classifyImage';
const form = document.querySelector('form');

form.addEventListener('submit', e => {
    e.preventDefault();

    //alert('Before image submit');

    const files = document.querySelector('[type=file]').files;
    const formData = new FormData();

    formData.append('imageFile', files[0]);

    //for (let i = 0; i < files.length; i++) {
    //    let file = files[i];
    //
    //    formData.append('imageFile[]', file);
    //}

/*    
    fetch(url, {
        method: 'POST',
        body: formData
    }).then(response => {
        alert('Got response');
        console.log('Logging to console');
        console.log(response);
        //alert(response);
        });
 */   

    fetch(url, {
        method: 'POST',
        body: formData
    }).then((resp) => resp.json())
      .then(function (response) {
            console.info('fetch()', response);
            alert('Prediction is: ' + 'Label: ' + response.predictedLabel + ' Probability: ' + response.probability);
            return response;
        });


});