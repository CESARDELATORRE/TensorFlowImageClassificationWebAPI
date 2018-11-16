

Use the model for one-time prediction.
Make the prediction function object. Note that, on average, this call takes around 200x longer than one prediction, so you might want to cache and reuse the prediction function, instead of creating one per prediction.

var predictionFunc = model.MakePredictionFunction<IrisInput, IrisPrediction>(mlContext);

Obtain the prediction. Remember that 'Predict' is not reentrant. 
If you want to use multiple threads for simultaneous prediction, make sure each thread is using its own PredictionFunction.

var prediction = predictionFunc.Predict(mySample);