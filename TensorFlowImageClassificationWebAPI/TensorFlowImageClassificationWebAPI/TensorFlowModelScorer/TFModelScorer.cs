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
        PredictionFunction<ImageInputData, ImageNetPrediction> CreatePredictionFunction();
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
        private readonly PredictionFunction<ImageInputData, ImageNetPrediction> _predictionFunction;
        public PredictionFunction<ImageInputData, ImageNetPrediction> PredictionFunction
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

            //Create the prediction function in the constructor, once, as it is an expensive operation
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

        public ImagePredictedLabelWithProbability PredictLabelForImage(byte[] imageData, string imageFilePath)
        {           
            var imageLabelPredicted = PredictLabelWithProbability(imageFilePath);

            return imageLabelPredicted;
        }

        public PredictionFunction<ImageInputData, ImageNetPrediction> CreatePredictionFunction()
        {
            try
            {
                var dataView = CreateDataView();

                var pipeline = ImageEstimatorsCatalog.LoadImages(catalog: _mlContext.Transforms, imageFolder: this._imagesTmpFolder, columns: ("ImagePath", "ImageReal"))
                            .Append(ImageEstimatorsCatalog.Resize(_mlContext.Transforms, "ImageReal", "ImageReal", ImageTransformationsSettings.imageHeight, ImageTransformationsSettings.imageWidth))
                            .Append(ImageEstimatorsCatalog.ExtractPixels(_mlContext.Transforms, new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", TensorFlowModelSettings.inputTensorName, interleave: ImageTransformationsSettings.channelsLast, offset: ImageTransformationsSettings.mean) }))
                            .Append(new TensorFlowEstimator(_mlContext, _modelLocation, new[] { TensorFlowModelSettings.inputTensorName }, new[] { TensorFlowModelSettings.outputTensorName }));

                var model = pipeline.Fit(dataView);

                var predictionFunction = model.MakePredictionFunction<ImageInputData, ImageNetPrediction>(_mlContext);

                return predictionFunction;

            }
            catch (Exception e)
            {
                throw e;
            }
        }

        protected ImagePredictedLabelWithProbability PredictLabelWithProbability(string imageFilePath)
        {
            //Read TF model's labels (labels.txt) to classify the image across those labels
            var labels = ModelHelpers.ReadLabels(this._labelsLocation);

            //Set the specific image data
            var imageInputData = new ImageInputData { ImagePath = imageFilePath };

            var imageLabelPredictions = this.PredictionFunction.Predict(imageInputData).PredictedLabels;

            //Set a single label as predicted or even none if probabilities were lower than 70%
            var imageBestLabelPrediction = new ImagePredictedLabelWithProbability()
            {
                ImagePath = imageInputData.ImagePath,
            };
            (imageBestLabelPrediction.PredictedLabel, imageBestLabelPrediction.Probability) = GetBestLabel(labels, imageLabelPredictions);

            return imageBestLabelPrediction;

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
