using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

using Swashbuckle.AspNetCore.Swagger;
using TensorFlowImageClassificationWebAPI.Infrastructure;
using TensorFlowImageClassificationWebAPI.TensorFlowModelScorer;

namespace TensorFlowImageClassificationWebAPI
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddMvc().SetCompatibilityVersion(CompatibilityVersion.Version_2_1);

            // Register the Swagger generator, defining 1 or more Swagger documents
            services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new Info { Title = "TensorFlow ImageClassification WebAPI", Version = "v1" });
            });

            // Register types (Interface/Class pairs) to use in DI/IoC
            services.AddTransient<IImageFileWriter, ImageFileWriter>();

            // Set TFModelScorer as Singleton so expensive initializations 
            // like prediction function is done once across Http calls
            services.AddSingleton<ITFModelScorer, TFModelScorer>();

        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IHostingEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            //Use this to set path of files outside the wwwroot folder
            //app.UseStaticFiles(new StaticFileOptions
            //{
            //    FileProvider = new PhysicalFileProvider(
            //        Path.Combine(Directory.GetCurrentDirectory(), "ImagesTemp")),
            //    RequestPath = "/ImagesTemp"
            //});

            //If using wwwroot/images folder
            //app.UseStaticFiles(); //letting the application know that we need access to wwwroot folder.

            // Enable middleware to serve generated Swagger as a JSON endpoint.
            app.UseSwagger();

            // Enable middleware to serve swagger-ui (HTML, JS, CSS, etc.), 
            // specifying the Swagger JSON endpoint.
            app.UseSwaggerUI(c =>
            {
                c.SwaggerEndpoint("/swagger/v1/swagger.json", "TensorFlow ImageClassification WebAPI - V1");
            });

            app.UseMvc();
        }
    }
}
