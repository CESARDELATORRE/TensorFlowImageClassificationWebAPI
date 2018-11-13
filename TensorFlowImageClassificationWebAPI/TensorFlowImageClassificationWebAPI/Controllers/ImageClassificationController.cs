using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using TensorFlowImageClassificationWebAPI.ImageDataStructures;
using TensorFlowImageClassificationWebAPI.Infrastructure;
using TensorFlowImageClassificationWebAPI.TensorFlowModelScorer;

namespace TensorFlowImageClassificationWebAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageClassificationController : ControllerBase
    {
        [HttpPost]
        //[Route("classifyImage/{approach}")]
        [Route("classifyImage")]
        public async Task<IActionResult> ClassifyImage(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return NoContent();

            IEnumerable<string> tags = null;
            using (var image = new MemoryStream())
            {
                await imageFile.CopyToAsync(image);
                var imageData = image.ToArray();
                if (!imageData.IsValidImage())
                    return StatusCode(StatusCodes.Status415UnsupportedMediaType);

                //Call the TFModelScorer
                TFModelScorer modelScorer = new TFModelScorer();
                var predictionFunction = modelScorer.CreatePredictionFunction(imageData);

                //tags = await visionStrategy.ClassifyImageAsync(imageData, classifyApproach);
            }

            return Ok(tags);
        }


        // GET api/values
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/values/5
        [HttpGet("{id}")]
        public ActionResult<string> Get(int id)
        {
            return "value";
        }

        // POST api/values
        [HttpPost]
        public void Post([FromBody] string value)
        {
        }

        // PUT api/values/5
        [HttpPut("{id}")]
        public void Put(int id, [FromBody] string value)
        {
        }

        // DELETE api/values/5
        [HttpDelete("{id}")]
        public void Delete(int id)
        {
        }
    }
}
