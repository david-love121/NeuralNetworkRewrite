using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetworkRewrite2024
{
    internal class Driver
    {
        private NeuralNetwork neuralNetwork;
        readonly int[] layerSizes = { 1, 1, 1 };
        LinearFunction linearFunction;
        DataGenerator dataGenerator;
        Function activationFunction;
        List<double> outputs;
        List<Matrix<double>> lastWeights;
        //Defines the activation function the neurons use
        internal Driver(Function activationFunction) 
        {
            this.activationFunction = activationFunction;
            linearFunction = new LinearFunction(2, 3);
            dataGenerator = new DataGenerator(linearFunction, 100, 1);
            neuralNetwork = new NeuralNetwork(layerSizes, activationFunction, 1);
            outputs = new List<double>();
            
        }
        internal Vector<double> GetScores(int totalPoints)
        {
            int index = 0;
            
            Vector<double> scores = Vector<double>.Build.Dense(totalPoints);
            //i represents the x value of the function, moves with step
            //index represents the index we're accessing from dataGenerator
            outputs.Clear();
            for (double i = 0; i < totalPoints; i += dataGenerator.GetStep())
            {
                Vector<double> outputVector = neuralNetwork.RunNetwork(i);
                
                outputs.Add(outputVector[0]);
                Vector<double> expectedOutput = Vector<double>.Build.Dense(1, dataGenerator.GetDataPoint(index));
                double score = neuralNetwork.ScoreOutput(outputVector, expectedOutput); //Based on average of MSE 
                scores[index] = score;
                index++;
                
            }
            return scores;
        }
        internal List<Matrix<double>> TrainEvolutionBased(int tests)
        {
            double minCost = double.MaxValue;
            List<Matrix<double>> bestResult = new List<Matrix<double>>();
            for (int i = 0; i < tests; i++)
            {
                int totalPoints = dataGenerator.GetSizeData();
                Vector<double> scoreVector = GetScores(totalPoints);
                double summedScores = scoreVector.Sum(); 
                double averageCost = summedScores / totalPoints;
                if (averageCost < minCost)
                {
                    minCost = averageCost;
                    bestResult = neuralNetwork.GetWeightsMatrixList();
                    lastWeights = bestResult;
                    
                }
                neuralNetwork.RandomizeWeights();
                
            }
            return bestResult;
        }
        internal void TrainBackpropagationBased(int epochs, int batchSize)
        {
            int totalPoints = dataGenerator.GetSizeData();
            Vector<double> scoreVector = GetScores(totalPoints);
            List<Vector<double>> weightDerivativesL = new List<Vector<double>>();
            List<Vector<double>> biasDerivativesL = new List<Vector<double>>();
            for (int i = 0; i < dataGenerator.GetSizeData(); i++)
            {
                Vector<double> weightDerivatives = Vector<double>.Build.Dense(2);
                Vector<double> biasDerivatives = Vector<double>.Build.Dense(2);
                int index = layerSizes.Length - 1;
                Layer lastLayer = neuralNetwork.GetLayer(index);
                //Vectors of 1 being used for scalability later //
                Vector<double> expectedOutput = Vector<double>.Build.Dense(1, dataGenerator.GetDataPoint(i));
                Vector<double> neuralNetworkOutput = neuralNetwork.RunNetwork(i);
                //del stands for partial derivative
                MSEFunction mseFunction = new MSEFunction();
                Vector<double> differenceVector = expectedOutput - neuralNetworkOutput;
                //Derivative of Cost wrt a
                Vector<double> delCDelAVector = mseFunction.ComputeDerivative(differenceVector);
                Vector<double> LastPreactivationValues = lastLayer.GetPreactivationValues();
                //Derivative of a wrt z
                Vector<double> delADelZ = Vector<double>.Build.Dense(LastPreactivationValues.Count);
                for (int k = 0; k < LastPreactivationValues.Count; k++)
                {
                    double DelValue = activationFunction.ComputeDerivative(LastPreactivationValues[k]);
                    delADelZ[k] = DelValue;
                }
                index = index - 1;
                
                Layer nextLayer = neuralNetwork.GetLayer(index);
                //Derivative of z wrt w
                double delCDelA = delCDelAVector.Sum();
                Vector<double> delZDelW = nextLayer.GetActiavtionValues();
                Vector<double> endChain = delADelZ.Multiply(delCDelA);
                Vector<double> delCDelW = endChain.PointwiseMultiply(delZDelW);
                Vector<double> delCDelB = endChain;
                weightDerivatives[0] = delCDelW[0];
                biasDerivatives[0] = delCDelB[0];
                double delCDelA_1 = endChain.Multiply(nextLayer.GetNeuron(0).GetConnectorOut(0).GetWeight())[0];
                Backpropagate(delCDelA_1, index, ref weightDerivatives, ref biasDerivatives);
                weightDerivativesL.Add(weightDerivatives);
                biasDerivativesL.Add(biasDerivatives);
                
            }

            int x = 2;
        }
        internal void AdjustWeights(double learningRate, List<Vector<double>> weightDerivatives, List<Vector<double>> biasDerivatives)
        {
            
        }
        internal void Backpropagate(double delCDelA, int index, ref Vector<double> wd, ref Vector<double> bd)
        {
            Layer lastLayer = neuralNetwork.GetLayer(index);
            
            Vector<double> LastPreactivationValues = lastLayer.GetPreactivationValues();
            //Derivative of a wrt z
            Vector<double> delADelZ = Vector<double>.Build.Dense(LastPreactivationValues.Count);
            for (int k = 0; k < LastPreactivationValues.Count; k++)
            {
                double DelValue = activationFunction.ComputeDerivative(LastPreactivationValues[k]);
                delADelZ[k] = DelValue;
            }
            index = index - 1;
            if (index >= 0)
            {
                Layer nextLayer = neuralNetwork.GetLayer(index);
                //Derivative of z wrt w
                Vector<double> delZDelW = nextLayer.GetActiavtionValues();
                Vector<double> endChain = delADelZ.Multiply(delCDelA);
                Vector<double> delCDelW = endChain.PointwiseMultiply(delZDelW);
                Vector<double> delCDelB = endChain;
                wd[1] = delCDelW[0];
                bd[1] = delCDelB[0];
                double delCDelA_1 = endChain.Multiply(nextLayer.GetNeuron(0).GetConnectorOut(0).GetWeight())[0];
            
                Backpropagate(delCDelA_1, index, ref wd, ref bd);
            }
        }
    }
}
