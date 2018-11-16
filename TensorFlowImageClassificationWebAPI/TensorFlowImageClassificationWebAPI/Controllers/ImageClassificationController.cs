using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
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

        private readonly ILogger<ImageClassificationController> _logger;
        private readonly ITFModelScorer _modelScorer;

        public ImageClassificationController(ITFModelScorer modelScorer, ILogger<ImageClassificationController> logger, IImageFileWriter imageWriter) //When using DI/IoC (IImageFileWriter imageWriter)
        {
            //Get injected dependencies
            _modelScorer = modelScorer;
            _logger = logger;
            _imageWriter = imageWriter;

            _imagesTmpFolder = ModelHelpers.GetFolderFullPath(@"ImagesTemp");
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("classifyimage")]
        public async Task<IActionResult> ClassifyImage(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return BadRequest();

            string imageFilePath = "", fileName = "";
            try
            {
                //Save the temp image image into the temp-folder 
                fileName = await _imageWriter.UploadImageAsync(imageFile, _imagesTmpFolder);
                imageFilePath = Path.Combine(_imagesTmpFolder, fileName);

                //Convert image stream to byte[] 
                byte[] imageData = null;
                //
                //Image stream still not used in ML.NET 0.7 but only possible through a file
                //
                //MemoryStream image = new MemoryStream();           
                //await imageFile.CopyToAsync(image);
                //imageData = image.ToArray();
                //if (!imageData.IsValidImage())
                //    return StatusCode(StatusCodes.Status415UnsupportedMediaType);

                ImagePredictedLabelWithProbability imageLabelPrediction = null;
                _logger.LogInformation($"Start processing image file { imageFilePath }");

                //Measure execution time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                //Predict the image's label (The one with highest probability)
                imageLabelPrediction = _modelScorer.PredictLabelForImage(imageData, imageFilePath);

                //Stop measuring time
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                imageLabelPrediction.PredictionExecutionTime = elapsedMs;

                _logger.LogInformation($"Image processed in {elapsedMs} miliseconds");

                //TODO: Commented as the file is still locked by TensorFlow or ML.NET?
                //_imageWriter.DeleteImageTempFile(imageFilePath);

                //return new ObjectResult(result);
                return Ok(imageLabelPrediction);
            }
            finally
            {
                try
                {
                    if(imageFilePath != string.Empty)
                    {
                        _logger.LogInformation($"Deleting Image {imageFilePath}");
                        //TODO: Commented as the file is still locked by TensorFlow or ML.NET?
                        //_imageWriter.DeleteImageTempFile(imageFilePath);
                    }
                }
                catch (Exception)
                {
                    _logger.LogInformation("Error deleting image: " + imageFilePath);
                }
            }
        }


        // GET api/ImageClassification
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            return new string[] { "ACK Heart beat 1", "ACK Heart beat 2" };
        }

    }
}
