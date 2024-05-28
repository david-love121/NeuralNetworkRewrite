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
            DataList dataList = new DataList(irisDataPath);
            Function activationFunction = new LinearFunction(0, 1);
            Driver driver = new Driver();
            List<Matrix<double>> bestWeights = driver.TrainEvolutionBased(1);
            
            double scoreEvo = driver.GetAverageScore();
            Console.WriteLine($"Evolution complete: Score {scoreEvo}");
            driver.TrainBackpropagationBased(100, 10, 3);
            double scoreB = driver.GetAverageScore();
            Console.WriteLine($"Backprop pass 1 complete: Score {scoreB} | Delta: {scoreB - scoreEvo}");
            driver.TrainBackpropagationBased(100, 10, 0.5);
            double scoreB2 = driver.GetAverageScore();
            Console.WriteLine($"Backprop pass 2 complete: Score {scoreB2} | Delta: {scoreB - scoreB2}");


        }
    }
}