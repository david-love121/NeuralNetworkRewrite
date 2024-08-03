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
            Driver driver = new Driver();
            List<double> scores = driver.RunBackpropagationLoop(2, 10, 10, 0.1, 2);
            driver.PrintOutput(0, 20);

        }

    }
}