using NeuralNetworkRewrite2024;
using System;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
namespace NeuralNetworkRewrite2024
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string lastNetworkPath = @"C:\Users\david\source\repos\NeuralNetworkRewrite\lastNetwork.txt";
            string irisDataPath = @"C:\Users\david\source\repos\NeuralNetworkRewrite\data\iris.data";

            Driver driver = new Driver();
            //List<Matrix<double>> bestWeights = driver.TrainEvolutionBased(10);
            
            double score = driver.GetAverageScore();
            driver.TrainBackpropagationBased(400, 5, 0.1, 5);
            double score2 = driver.GetAverageScore();
            driver.TrainBackpropagationBased(400, 5, 2, 0.1);
            double score3 = driver.GetAverageScore();

            List<double> scores = driver.RunBackpropagationLoop(10, 400, 3, 0.1, 5);
            driver.PrintOutput(0, 20);
            int x = 2;
        }

    }
}