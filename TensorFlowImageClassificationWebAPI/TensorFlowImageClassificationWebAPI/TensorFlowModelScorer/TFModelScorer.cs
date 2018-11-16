using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using TensorFlowImageClassificationWebAPI.ImageDataStructures;

namespace TensorFlowImageClassificationWebAPI.TensorFlowModelScorer
{
    /// <summary>
    /// Interface to use with DI/IoC
    /// </summary>
    ///
    public interface ITFModelScorer
    {
        PredictionFunction<ImageInputData, ImageLabelPredictions> CreatePredictionFunction();
        ImagePredictedLabelWithProbability PredictLabelForImage(byte[] imageData, string imageFilePath);
    }

    /// <summary>
    /// Class implementation to be injected by DI/IoC
    /// </summary>
    ///
    public class TFModelScorer : ITFModelScorer
    {
        private readonly string _modelLocation;
        private readonly string _labelsLocation;
        private readonly string _imagesTmpFolder;
        private readonly MLContext _mlContext;

        #pragma warning disable IDE0032
        private readonly PredictionFunction<ImageInputData, ImageLabelPredictions> _predictionFunction;
        public PredictionFunction<ImageInputData, ImageLabelPredictions> PredictionFunction
        {
            get => _predictionFunction;
        }

        //public TFModelScorer(string name) => Name = name;
        public TFModelScorer()
        {
            var assetsPath = ModelHelpers.GetFolderFullPath(@"TensorFlowModel");

            this._modelLocation = Path.Combine(assetsPath, "model.pb");
            this._labelsLocation = Path.Combine(assetsPath, "labels.txt");
            this._imagesTmpFolder = ModelHelpers.GetFolderFullPath(@"ImagesTemp");

            _mlContext = new MLContext(seed: 1);  //Setting seed so predictions are deterministic

            // Create the prediction function in the constructor, once, as it is an expensive operation
            // Note that, on average, this call takes around 200x longer than one prediction, so you want to cache it
            // and reuse the prediction function, instead of creating one per prediction.
            // IMPORTANT: Remember that the 'Predict()' method is not reentrant. 
            // If you want to use multiple threads for simultaneous prediction, 
            // make sure each thread is using its own PredictionFunction (e.g. In DI/IoC use .AddScoped())
            // or use a critical section when using the Predict() method.
            _predictionFunction = this.CreatePredictionFunction();
        }

        private struct ImageTransformationsSettings
        {
            public const int imageHeight = 227;
            public const int imageWidth = 227;
            public const float mean = 117;         //
            public const bool channelsLast = true; //
        }

        private struct TensorFlowModelSettings
        {
            // For checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string inputTensorName = "Placeholder";

            // output tensor name
            public const string outputTensorName = "loss";
        }

        public ImagePredictedLabelWithProbability PredictLabelForImage(byte[] imageData, string imageFile)
        {           
            var imageLabelPredicted = PredictLabelWithProbability(imageFile);

            return imageLabelPredicted;
        }

        public PredictionFunction<ImageInputData, ImageLabelPredictions> CreatePredictionFunction()
        {
            try
            {
                var dataView = CreateDataView();

                var pipeline = ImageEstimatorsCatalog.LoadImages(catalog: _mlContext.Transforms, imageFolder: this._imagesTmpFolder, columns: ("ImagePath", "ImageReal"))
                            .Append(ImageEstimatorsCatalog.Resize(_mlContext.Transforms, "ImageReal", "ImageReal", ImageTransformationsSettings.imageHeight, ImageTransformationsSettings.imageWidth))
                            .Append(ImageEstimatorsCatalog.ExtractPixels(_mlContext.Transforms, new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", TensorFlowModelSettings.inputTensorName, interleave: ImageTransformationsSettings.channelsLast, offset: ImageTransformationsSettings.mean) }))
                            .Append(new TensorFlowEstimator(_mlContext, _modelLocation, new[] { TensorFlowModelSettings.inputTensorName }, new[] { TensorFlowModelSettings.outputTensorName }));

                var model = pipeline.Fit(dataView);

                var predictionFunction = model.MakePredictionFunction<ImageInputData, ImageLabelPredictions>(_mlContext);

                return predictionFunction;

            }
            catch (Exception e)
            {
                throw e;
            }
        }

        protected ImagePredictedLabelWithProbability PredictLabelWithProbability(string imageFile)
        {
            try
            {
                //Read TF model's labels (labels.txt) to classify the image across those labels
                var labels = ModelHelpers.ReadLabels(this._labelsLocation);

                //Set the specific image data
                var imageInputData = new ImageInputData { ImagePath = imageFile };
                float[] imageLabelPredictions;

                //Set the critical section if using Singleton for the TFModelScorer object
                //               
                lock (_predictionFunction)
                {
                    imageLabelPredictions = _predictionFunction.Predict(imageInputData).PredictedLabels;
                }
                //
                // Note that if using Scoped instead of singleton in DI/IoC you can remove the critical section
                // It depends if you want better performance in single Http calls (using singleton) 
                // versus better scalability ann global performance if you have many Http requests/threads 
                // since the critical section is a bottleneck reducing the execution to one thread for that particular call
                //

                //Set a single label as predicted or even none if probabilities were lower than 70%
                var imageBestLabelPrediction = new ImagePredictedLabelWithProbability()
                {
                    ImagePath = imageInputData.ImagePath,
                };

                (imageBestLabelPrediction.PredictedLabel, imageBestLabelPrediction.Probability) = GetBestLabel(labels, imageLabelPredictions);

                return imageBestLabelPrediction;

            }
            catch (Exception e)
            {
                throw e;
            } 
        }

        private (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);


            if (max > 0.7)
                return (labels[index], max);
            else
                return ("None", max);
        }

        private IDataView CreateDataView()
        {
            //Create empty DataView. We just need the schema to call fit()
            List<ImageInputData> list = new List<ImageInputData>();
            //list.Add(new ImageInputData() { ImagePath = "image-name.jpg" });   //Since we just need the schema, no need to provide anything here
            IEnumerable<ImageInputData> enumerableData = list;
            var dv = _mlContext.CreateStreamingDataView(enumerableData);
            return dv;
        }

        // This approach is still not supported by ML.NET 0.7
        //
        //private IDataView CreateDataViewFromStreams(byte[] imageData)
        //{
        //    //Convert image to Bitmap
        //    Bitmap bmp;
        //    using (var ms = new MemoryStream(imageData))
        //    {
        //        bmp = new Bitmap(ms);
        //    }

        //    //Create DataView with image to process
        //    List<ImageInputData> list = new List<ImageInputData>();
        //    list.Add(new ImageInputData() { Image = new Bitmap(bmp) });
        //    IEnumerable<ImageInputData> enumerableData = list;
        //    var dv = mlContext.CreateStreamingDataView(enumerableData);

        //    return dv;
        //}



    }
}
