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
            Driver driver = new Driver(activationFunction);
            List<Matrix<double>> bestWeights = driver.TrainEvolutionBased(100000);
            driver.TestSerialization(path);
            
            double scoreEvo = driver.GetAverageScore();
            driver.TrainBackpropagationBased(3, 10, 0.01);
            double scoreB = driver.GetAverageScore();
            

        }
       
    }
}