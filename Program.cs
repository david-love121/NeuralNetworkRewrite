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
            string path = @"C:\Users\david\source\repos\NeuralNetworkRewrite\lastNetwork.txt";
            Function activationFunction = new LinearFunction(0, 1);
            Driver driver = new Driver();
            List<Matrix<double>> bestWeights = driver.TrainEvolutionBased(10);
            
            double scoreEvo = driver.GetAverageScore();
            Console.WriteLine($"Evolution complete: Score {scoreEvo}");
            driver.TrainBackpropagationBased(10, 10, 0.1);
            double scoreB = driver.GetAverageScore();
            Console.WriteLine($"Backprop pass 1 complete: Score {scoreB} | Delta: {scoreB - scoreEvo}");

        }
    }
}