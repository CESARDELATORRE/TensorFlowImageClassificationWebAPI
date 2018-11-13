
using Microsoft.ML.Runtime.Api;

namespace TensorFlowImageClassificationWebAPI.ImageDataStructures
{
    public class ImageNetPrediction
    {
        //TODO: Change to fixed output column name for TensorFlow model
        [ColumnName("loss")]
        public float[] PredictedLabels;
    }
}
