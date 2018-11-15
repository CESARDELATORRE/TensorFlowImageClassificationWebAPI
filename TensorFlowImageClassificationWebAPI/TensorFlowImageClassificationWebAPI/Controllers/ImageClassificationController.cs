using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using TensorFlowImageClassificationWebAPI.ImageDataStructures;
using TensorFlowImageClassificationWebAPI.Infrastructure;
using TensorFlowImageClassificationWebAPI.TensorFlowModelScorer;

namespace TensorFlowImageClassificationWebAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageClassificationController : ControllerBase
    {
        //Dependencies
        private readonly IImageFileWriter _imageWriter; 
        private readonly string _imagesTmpFolder;
        private readonly ITFModelScorer _modelScorer;

        public ImageClassificationController(ITFModelScorer modelScorer, IImageFileWriter imageWriter) //When using DI/IoC (IImageFileWriter imageWriter)
        {
            //Get injected dependencies
            _modelScorer = modelScorer;
            _imageWriter = imageWriter;

            _imagesTmpFolder = ModelHelpers.GetFolderFullPath(@"ImagesTemp");
        }

        [HttpPost]
        //[Route("classifyImage/{approach}")]
        [Route("classifyImage")]
        public async Task<IActionResult> ClassifyImage(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return NoContent();

            //Save the temp image image into the temp-folder 
            var fileName = await _imageWriter.UploadImageAsync(imageFile, _imagesTmpFolder);
            string imageFilePath = Path.Combine(_imagesTmpFolder, fileName);
            
            //Convert image stream to byte[] - Image stream still not used in ML.NET 0.7 but through the file
            ImagePredictedLabelWithProbability imageLabelPrediction = null;
            var image = new MemoryStream();           
            await imageFile.CopyToAsync(image);
            var imageData = image.ToArray();
            if (!imageData.IsValidImage())
                return StatusCode(StatusCodes.Status415UnsupportedMediaType);


            //Measure execution time
            var watch = System.Diagnostics.Stopwatch.StartNew();
            
            //Predict the image's label (The one with highest probability)
            imageLabelPrediction = _modelScorer.PredictLabelForImage(imageData, imageFilePath);

            // the code that you want to measure comes here
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            imageLabelPrediction.PredictionExecutionTime = elapsedMs;

            //TODO: Commented as the file is still locked by TensorFlow or ML.NET?
            //_imageWriter.DeleteImageTempFile(imageFilePath);

            //return new ObjectResult(result);
            return Ok(imageLabelPrediction);
        }


        // GET api/ImageClassification
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            return new string[] { "ACK Heart beat 1", "ACK Heart beat 2" };
        }

    }
}
