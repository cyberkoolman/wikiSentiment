using Common;
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace WikiSentiment
{
    public class SentimentIssue
    {
        [LoadColumn(0)] 
        public bool Label { get; set; }
        [LoadColumn(2)] 
        public string Text { get; set; }
    }

    /// <summary>
    /// The SentimentPrediction class represents a single sentiment prediction.
    /// </summary>
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] 
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }


    class Program
    {
        // private static string filename = "./data/wikiDetoxAnnotated40kRows.tsv";
        private static string filename = "./data/simpleTest.tsv";
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, filename);

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var mlContext = new MLContext(seed:99);

            // load the data file
            ConsoleHelper.ConsoleWriteHeader("=========== Loading Data ==========");
            var data = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataPath, hasHeader: true);

            // split the data into 80% training and 20% testing partitions
            var partitions = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            ConsoleHelper.ShowDataViewInConsole(mlContext, partitions.TrainSet, 8);
            ConsoleHelper.ShowDataViewInConsole(mlContext, partitions.TestSet, 2);

            // build a machine learning pipeline
            // step 1: featurize the text
            var pipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features", 
                inputColumnName: nameof(SentimentIssue.Text))

                // step 2: add a fast tree learner
                .Append(mlContext.BinaryClassification.Trainers.FastTree(                    
                    labelColumnName: nameof(SentimentIssue.Label), 
                    featureColumnName: "Features"));

            // train the model
            ConsoleHelper.ConsoleWriteHeader("============ Training model =============");
            var model = pipeline.Fit(partitions.TrainSet);

            ConsoleHelper.ConsoleWriteHeader("============ Evaluating model ============");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(           
                data:predictions, 
                labelColumnName: nameof(SentimentIssue.Label), 
                scoreColumnName: nameof(SentimentPrediction.Score));

            ConsoleHelper.PrintBinaryClassificationMetrics(model.ToString(), metrics);

            // create a prediction engine to make a single prediction
            ConsoleHelper.ConsoleWriteHeader("=========== Making a prediction ==========");
            // var newText = "With all due respect, you are a moron";
            var newText = "Not bad at all";
            var issue = new SentimentIssue { Text = newText };
            var engine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(model);

            // make a single prediction
            var prediction = engine.Predict(issue);

            // report results
            Console.WriteLine($"  Text:        {issue.Text}");
            Console.WriteLine($"  Prediction:  {prediction.Prediction}");
            Console.WriteLine($"  Probability: {prediction.Probability:P2}");
            Console.WriteLine($"  Score:       {prediction.Score}");
        }
    }
}