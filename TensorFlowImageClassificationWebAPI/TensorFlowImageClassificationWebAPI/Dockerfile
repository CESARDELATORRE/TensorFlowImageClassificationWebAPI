FROM microsoft/dotnet:2.1-aspnetcore-runtime AS base
WORKDIR /app
EXPOSE 80

# install System.Drawing native dependencies
RUN apt-get update \
    && apt-get install -y --allow-unauthenticated \
        libc6-dev \
        libgdiplus \
        libx11-dev \
     && rm -rf /var/lib/apt/lists/*

FROM microsoft/dotnet:2.1-sdk AS build
WORKDIR /src
COPY ["TensorFlowImageClassificationWebAPI/TensorFlowImageClassificationWebAPI.csproj", "TensorFlowImageClassificationWebAPI/"]
RUN dotnet restore "TensorFlowImageClassificationWebAPI/TensorFlowImageClassificationWebAPI.csproj"
COPY . .
WORKDIR "/src/TensorFlowImageClassificationWebAPI"
RUN dotnet build "TensorFlowImageClassificationWebAPI.csproj" -c Release -o /app

FROM build AS publish
RUN dotnet publish "TensorFlowImageClassificationWebAPI.csproj" -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "TensorFlowImageClassificationWebAPI.dll"]