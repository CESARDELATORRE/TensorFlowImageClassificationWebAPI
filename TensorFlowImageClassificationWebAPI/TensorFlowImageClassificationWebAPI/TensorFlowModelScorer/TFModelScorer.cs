using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using TensorFlowImageClassificationWebAPI.ImageDataStructures;

namespace TensorFlowImageClassificationWebAPI.TensorFlowModelScorer
{
    public class TFModelScorer
    {
        private readonly string modelLocation;
        private readonly string labelsLocation;
        private readonly MLContext mlContext;

        public TFModelScorer()
        {
            var assetsPath = ModelHelpers.GetAssetsPath(@"TensorFlowModel");

            this.modelLocation = Path.Combine(assetsPath, "model.pb");
            this.labelsLocation = Path.Combine(assetsPath, "labels.txt");

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

        public PredictionFunction<ImageInputData, ImageNetPrediction> CreatePredictionFunction(byte[] imageData)
        {
            try
            {
                //Convert image to Bitmap
                Bitmap bmp;
                using (var ms = new MemoryStream(imageData))
                {
                    bmp = new Bitmap(ms);
                }

                //Create DataView with image to process
                List<ImageInputData> list = new List<ImageInputData>();
                list.Add(new ImageInputData() { Image = new Bitmap(bmp) });
                IEnumerable<ImageInputData> enumerableData = list;
                var dv = mlContext.CreateStreamingDataView(enumerableData);


                ////Define pipeline for image transformations
                //var pipeline = ImageEstimatorsCatalog.Resize(mlContext.Transforms, "ImageReal", "ImageReal", ImageNetSettings.imageHeight, ImageNetSettings.imageWidth))
                //                .Append(ImageEstimatorsCatalog.ExtractPixels(mlContext.Transforms, new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", "Placeholder", interleave: ImageNetSettings.channelsLast, offset: ImageNetSettings.mean) }))
                //                .Append(new TensorFlowEstimator(mlContext, modelLocation, new[] { "Placeholder" }, new[] { "loss" }));

                //var model = pipeline.Fit(CreateDataView());

                //var predictionFunction = model.MakePredictionFunction<ImageInputData, ImageNetPrediction>(mlContext);

                //return predictionFunction;

            }
            catch (Exception e)
            {
                throw e;
            }


            return null;
        }

    }
}
