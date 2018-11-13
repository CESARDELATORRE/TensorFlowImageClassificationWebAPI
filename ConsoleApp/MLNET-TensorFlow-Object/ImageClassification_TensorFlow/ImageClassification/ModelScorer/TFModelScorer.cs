using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;
using Microsoft.ML;

using ImageClassification.ImageDataStructures;
using static ImageClassification.ModelScorer.ConsoleHelpers;
using static ImageClassification.ModelScorer.ModelHelpers;
using Microsoft.ML.Runtime.Api;
using System.Data;

namespace ImageClassification.ModelScorer
{
    public class TFModelScorer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly string labelsLocation;
        private readonly MLContext mlContext;

        public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.labelsLocation = labelsLocation;
            mlContext = new MLContext();
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 227;
            public const int imageWidth = 227;
            public const float mean = 117;         //
            public const bool channelsLast = true; //
        }

        public struct InceptionSettings
        {
            // For checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string inputTensorName = "Placeholder";

            // output tensor name
            public const string outputTensorName = "loss";
        }

        public void Score()
        {
            var predFunction = CreatePredictionFunction(dataLocation, imagesFolder, modelLocation);

            var predictions = PredictDataUsingModel(dataLocation, imagesFolder, labelsLocation, predFunction).ToArray();

        }

        private PredictionFunction<ImageInputData, ImageNetPrediction> CreatePredictionFunction(string dataLocation, string imagesFolder, string modelLocation)
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");           
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}");

            //Define pieplie for image tansformations
            var pipeline = ImageEstimatorsCatalog.LoadImages(catalog: mlContext.Transforms, imageFolder: imagesFolder, columns: ("ImagePath", "ImageReal"))                          
                            .Append(ImageEstimatorsCatalog.Resize(mlContext.Transforms, "ImageReal", "ImageReal", ImageNetSettings.imageHeight, ImageNetSettings.imageWidth))
                            .Append(ImageEstimatorsCatalog.ExtractPixels(mlContext.Transforms, new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", "Placeholder", interleave: ImageNetSettings.channelsLast, offset: ImageNetSettings.mean) }))
                            .Append(new TensorFlowEstimator(mlContext, modelLocation, new[] { "Placeholder" }, new[] { "loss" }));
              
            var model = pipeline.Fit(CreateDataView());

            var predictionFunction = model.MakePredictionFunction<ImageInputData, ImageNetPrediction>(mlContext);

            return predictionFunction;
        }

        private IDataView CreateDataView()
        {
            //Create empty DataView. We just need the schema to call fit()
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(new ImageInputData() { ImagePath = "" });
            IEnumerable<ImageInputData> enumerableData = list;
            var dv = mlContext.CreateStreamingDataView(enumerableData);
            return dv;
        }

        protected IEnumerable<ImagePredictedLabelWithProbability> PredictDataUsingModel(string testLocation, string imagesFolder, string labelsLocation, PredictionFunction<ImageInputData, ImageNetPrediction> model)
        {
            ConsoleWriteHeader("Classificate images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");


            var labels = ModelHelpers.ReadLabels(labelsLocation);

            /////////////////////////////////////////////////////////////////////////////////////
            // IMAGE 1
            // Predict label for "green-office-chair-test.jpg"
            var image1 = new ImageInputData { ImagePath = imagesFolder + "\\" + "green-office-chair-test.jpg" };
            var image1Probabilities = model.Predict(image1).PredictedLabels;

            //Set a single label as predicted or even none if probabilities were lower than 70%
            var image1BestLabelPrediction = new ImagePredictedLabelWithProbability()
            {
                ImagePath = image1.ImagePath,
            };
            (image1BestLabelPrediction.PredictedLabel, image1BestLabelPrediction.Probability) = GetBestLabel(labels, image1Probabilities);

            image1BestLabelPrediction.ConsoleWrite();

            yield return image1BestLabelPrediction;


            /////////////////////////////////////////////////////////////////////////////////////
            // IMAGE 2
            // Predict label for "high-metal-office-chair.jpg"
            var image2 = new ImageInputData { ImagePath = imagesFolder + "\\" + "high-metal-office-chair.jpg" };
            var image2Probabilities = model.Predict(image2).PredictedLabels;

            //Set a single label as predicted or even none if probabilities were lower than 70%
            var image2BestLabelPrediction = new ImagePredictedLabelWithProbability()
            {
                ImagePath = image2.ImagePath,
            };
            (image2BestLabelPrediction.PredictedLabel, image2BestLabelPrediction.Probability) = GetBestLabel(labels, image2Probabilities);

            image2BestLabelPrediction.ConsoleWrite();

            yield return image1BestLabelPrediction;

        }
    }
}
