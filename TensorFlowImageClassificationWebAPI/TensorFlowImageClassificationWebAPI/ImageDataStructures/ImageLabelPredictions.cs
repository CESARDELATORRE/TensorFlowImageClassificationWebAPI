
using Microsoft.ML.Runtime.Api;

namespace TensorFlowImageClassificationWebAPI.ImageDataStructures
{
    public class ImageLabelPredictions
    {
        //TODO: Change to fixed output column name for TensorFlow model
        [ColumnName("loss")]
        public float[] PredictedLabels;
    }
}
