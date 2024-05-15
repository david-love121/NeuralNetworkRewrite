using NeuralNetworkRewrite2024;
using System;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetworkRewrite2024
{
    internal class Program
    {
        static void Main(string[] args)
        {

            Function activationFunction = new LinearFunction(0, 1);
            Driver driver = new Driver(activationFunction);
            List<Matrix<double>> bestWeights = driver.TrainEvolutionBased(100000);
            double scoreEvo = driver.GetAverageScore();
            driver.TrainBackpropagationBased(3, 10, 0.01);
            double scoreB = driver.GetAverageScore();
            int x = 2;

        }
    }
}