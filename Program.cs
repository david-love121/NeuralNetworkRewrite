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
            driver.TrainBackpropagationBased(100, 100);
            int x = 2;

        }
    }
}