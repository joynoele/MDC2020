using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.ML; // <-- need this general ML
using Microsoft.ML.Data; // <-- needed for Evaluation step

namespace MDC2020
{
    /**
     * Comments are from Microsoft's Price Prediction (regression) tutorial here:
     * https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/predict-prices
    **/
    class Program
    {
        static string _solutionRoot = Helper.NavigateUpDirectory(Environment.CurrentDirectory, 4);
        static readonly string _trainDataPath = Path.Combine(_solutionRoot, "StockPriceSimulation_train.csv");
        static readonly string _testDataPath = Path.Combine(_solutionRoot, "StockPriceSimulation_test.csv");
        static readonly string _modelOutputPath = Path.Combine(_solutionRoot, "RrtRegressionModel.zip");
        private static MLContext _mlContext;

        public static void Main(string[] args)
        {
            if (!CheckFilesExist()) return;
            Console.WriteLine("Let's model Rate of Return on investment!");

            _mlContext = new MLContext(seed: 0);
            IDataView trainData = LoadData(_trainDataPath);

            Console.WriteLine($"Creating model based on data from {_trainDataPath}");
            ITransformer model = Train(trainData);
            
            Console.WriteLine($"Now Evaluating the model with test data from {_testDataPath}");
            Evaluate(model, _testDataPath);

            Console.WriteLine("Predict data from tests");
            TestSinglePrediction(model);

            _mlContext.Model.Save(model, trainData.Schema, _modelOutputPath);

            //Console.WriteLine("Check feature importance");
            TrainAndPFI(model, trainData);
        }

        private static IDataView LoadData(string dataPath)
        {
            return _mlContext.Data.LoadFromTextFile<RateOfReturnSimulation>(dataPath, hasHeader: true, separatorChar: ',');
        }

        private static IEstimator<ITransformer> CreatePipeline(IDataView dataView)
        {
            string[] featureColumnNames = dataView.Schema.Select(column => column.Name).Where(columnName => columnName != "Label" && columnName != "Run").ToArray();

            var pipeline = _mlContext.Transforms.Categorical.OneHotEncoding(
                    new[] { new InputOutputColumnPair("Security"), new InputOutputColumnPair("IsSplit"), new InputOutputColumnPair("IsBust") })
                .Append(_mlContext.Transforms.DropColumns("Run"))
                .Append(_mlContext.Transforms.Concatenate("Features", featureColumnNames))
                //.Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        /// <summary>
        /// The Train() method executes the following tasks:
        /// 1. Loads the data.
        /// 2. Extracts and transforms the data.
        /// 3. Trains the model.
        /// 4. Returns the model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainDataPath">Location of data file</param>
        /// <returns></returns>
        private static ITransformer Train(IDataView dataView)
        {
            var pipeline = CreatePipeline(dataView)
                .Append(_mlContext.Regression.Trainers.FastTree());

            /**
             Available regression trainers:
                LbfgsPoissonRegressionTrainer - N/A because of negative values
                LightGbmRegressionTrainer
                SdcaRegressionTrainer - 0.32
                OlsTrainer - 0.34 (w/ normalization) https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.olstrainer?view=ml-dotnet
                OnlineGradientDescentTrainer - 0.32 (w/ normalization)
                FastTreeRegressionTrainer - https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttree.fasttreeregressiontrainer?view=ml-dotnet
                FastTreeTweedieTrainer - 0.78
                FastForestRegressionTrainer - 0.59
                GamRegressionTrainer - 0.35 (took seconds)
             **/

            var model = pipeline.Fit(dataView);

            return model;
        }

        /// <summary>
        /// The Evaluate method executes the following tasks:
        /// 1. Loads the test dataset
        /// 2. Creates the regression evaluator
        /// 3. Evaluates the model and creates metrics
        /// 4. Displays the metrics
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void Evaluate(ITransformer model, string testDataPath)
        {
            IDataView testData = LoadData(testDataPath);
            IDataView prediction = model.Transform(testData);
            RegressionMetrics metrics = _mlContext.Regression.Evaluate(prediction, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"****************************************************");
            Console.WriteLine($"*       Regression Model quality evaluation         ");
            Console.WriteLine($"*---------------------------------------------------");
            Console.WriteLine($"* {metrics.RSquared:0.##}\tRSquared Score (good = close to 1)");
            Console.WriteLine($"* {metrics.RootMeanSquaredError:0.##}\tRoot Mean Squared Error (good = lower)");
            Console.WriteLine($"* {metrics.MeanSquaredError:0.##}\tMean Squared Error");
            Console.WriteLine($"* {metrics.MeanAbsoluteError:0.##}\tMean Absolute Error");
            Console.WriteLine($"* {metrics.LossFunction:0.##}\tLoss Function");
        }

        /// <summary>
        /// The TestSinglePrediction method executes the following tasks:
        /// 1. Creates a single comment of test data.
        /// 2. Predicts fare amount based on test data.
        /// 3. Combines test data and predictions for reporting.
        /// 4. Displays the predicted results.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void TestSinglePrediction(ITransformer model)
        {
            var predictionFunction = _mlContext.Model.CreatePredictionEngine<RateOfReturnSimulation, RateOfReturnPrediction>(model);
            IDataView dataView = _mlContext.Data.LoadFromTextFile<RateOfReturnSimulation>(_testDataPath, hasHeader: true, separatorChar: ',');

            // 137,Pioneer,10,92,15,False,False,4,0.109963825208794
            var simulationSample = new RateOfReturnSimulation()
            {
                Run = 137,
                Security = "Pioneer",
                Year = 10,
                Price = 92,
                Delta = 15,
                IsSplit = false,
                IsBust = false,
                Yield = 4,
                AvgRateOfReturn = 0.109963825208794F // This gets predicted
            };

            var prediction = predictionFunction.Predict(simulationSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Rate of return: predicted: {prediction.AvgRateOfReturn:P2}, actual: {simulationSample.AvgRateOfReturn:P2}");
            Console.WriteLine($"**********************************************************************");
        }

        /// <summary>
        /// Run Pemutation Feature Importance (PFI) on the model
        /// Reference doce: https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/explain-machine-learning-model-permutation-feature-importance-ml-net
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainingData"></param>
        /// <param name="model"></param>
        private static void TrainAndPFI(ITransformer model, IDataView trainingData)
        {
            IDataView transformedData = model.Transform(trainingData);
            var estimator = _mlContext.Regression.Trainers.FastTree();
            var predictor = estimator.Fit(transformedData);

            ImmutableArray<RegressionMetricsStatistics> permutationFeatureImportance = _mlContext.Regression
                .PermutationFeatureImportance(predictor, transformedData, permutationCount: 10);

            // Order features by importance
            var featureImportanceMetrics =
                permutationFeatureImportance
                    .Select((metric, index) => new { index, metric.RSquared })
                    .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean));

            var featureEncodedColumnNames = transformedData.Schema.Select(x => x.Name + "-" + x.Type.RawType).ToArray();

            Console.WriteLine("PFI\tFeature");
            foreach (var feature in featureImportanceMetrics)
            {
                var featureName = "unknown";
                try
                {
                    featureName = featureEncodedColumnNames[feature.index];
                }
                catch (IndexOutOfRangeException)
                {
                    featureName = "OutOfRange";
                }
                Console.WriteLine($"{feature.RSquared.Mean:F6}\t{featureName}");
            }

        }

        private static bool CheckFilesExist()
        {
            if (!File.Exists(_trainDataPath))
            {
                Console.WriteLine($"Test data not found at {_trainDataPath}");
                return false;
            }
            if (!File.Exists(_testDataPath))
            {
                Console.WriteLine($"Training data not found at {_testDataPath}");
                return false;
            }
            return true;
        }
    }
}
