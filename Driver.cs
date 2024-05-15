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
                Vector<double> expectedOutput = Vector<double>.Build.Dense(1, dataGenerator.GetDataPoint(index));
                double score = neuralNetwork.ScoreOutput(outputVector, expectedOutput); //Based on average of MSE 
                scores[index] = score;
                index++;
                
            }
            return scores;
        }
        internal double GetAverageScore()
        {
            Vector<double> scores = GetScores(dataGenerator.GetSizeData());
            return scores.Average();
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
        internal void TrainBackpropagationBased(int epochs, int batchSize, double learningRate = 1.0)
        {
            int totalPoints = dataGenerator.GetSizeData();
            int batchesPerTrainingCycle = totalPoints / batchSize;
            Vector<double> scoreVector = GetScores(totalPoints);
            List<Vector<double>> weightDerivativesL = new List<Vector<double>>();
            List<Vector<double>> biasDerivativesL = new List<Vector<double>>();
            List<Vector<double>> weightDerivativeCollection = new List<Vector<double>>();
            List<Vector<double>> biasDerivativeCollection = new List<Vector<double>>();
            for (int a = 0; a < epochs; a++) 
            {
                for (int b = 0; b < batchesPerTrainingCycle; b++) 
                {
                    for (int i = batchSize*b; i < batchSize*(b+1); i++)
                        {
                        weightDerivativeCollection.Clear();
                        biasDerivativeCollection.Clear();
                        //Hard coding sizing for now
                        Vector<double> weightDerivatives = Vector<double>.Build.Dense(2);
                        Vector<double> biasDerivatives = Vector<double>.Build.Dense(3);
                        int index = layerSizes.Length - 1;
                        Layer lastLayer = neuralNetwork.GetLayer(index);
                        //Vectors of 1 being used for scalability later 
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
                        biasDerivatives[0] = delADelZ[0]; //I believe I need to sum this for the vector representation
                        index = index - 1;

                        Layer nextLayer = neuralNetwork.GetLayer(index);
                        //Derivative of z wrt w
                        double delCDelA = delCDelAVector.Sum();
                        Vector<double> delZDelW = nextLayer.GetActiavtionValues();
                        Vector<double> endChain = delADelZ.Multiply(delCDelA);
                        Vector<double> delCDelW = endChain.PointwiseMultiply(delZDelW);
                        Vector<double> delCDelB = endChain;
                        //Remember that these vectors begin at 0 index for the last layer
                        weightDerivatives[0] = delCDelW[0];
                        biasDerivatives[1] = delCDelB[0];
                        double delCDelA_1 = endChain.Multiply(nextLayer.GetNeuron(0).GetConnectorOut(0).GetWeight())[0];
                        Backpropagate(delCDelA_1, index, ref weightDerivatives, ref biasDerivatives);
                        weightDerivativeCollection.Add(weightDerivatives);
                        biasDerivativeCollection.Add(biasDerivatives);
                    }
                    //Maybe clip here
                    Vector<double> averageWeightDerivative = CalculateAverageVector(weightDerivativeCollection);
                    Vector<double> averageBiasDerivative = CalculateAverageVector(biasDerivativeCollection);
                    weightDerivativesL.Add(averageWeightDerivative);
                    biasDerivativesL.Add(averageBiasDerivative);
                }
                //TODO: Continue working on normalizing gradients, get L2 Norm and center around that
                AdjustWeights(learningRate, weightDerivativesL, biasDerivativesL);
            }
            
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
            //Ensure bias derivative gets the end bias value
            if (index >= 0)
            {
                Layer nextLayer = neuralNetwork.GetLayer(index);
                //Derivative of z wrt w
                Vector<double> delZDelW = nextLayer.GetActiavtionValues();
                Vector<double> endChain = delADelZ.Multiply(delCDelA);
                Vector<double> delCDelW = endChain.PointwiseMultiply(delZDelW);
                Vector<double> delCDelB = endChain;
                wd[wd.Count - index - 1] = delCDelW[0];
                bd[bd.Count - index - 1] = delCDelB[0];
                double delCDelA_1 = endChain.Multiply(nextLayer.GetNeuron(0).GetConnectorOut(0).GetWeight())[0];

                Backpropagate(delCDelA_1, index, ref wd, ref bd);
            }
        }
        //weightDerivatives and biasDerivatives need to be updated to lists of matrices, but do this once it works with vectors
        internal void AdjustWeights(double learningRate, List<Vector<double>> weightDerivatives, List<Vector<double>> biasDerivatives)
        {
            List<Matrix<double>> oldWeights = neuralNetwork.GetWeightsMatrixList();
            //For testing single layer perceptron networks
            Vector<double> WeightVector = Vector<double>.Build.Dense(oldWeights.Count);
            //Derivative wrt cost after clipping based on L2
            Vector<double> averageWeightDerivatives = CalculateAverageVectorWithClipping(weightDerivatives, 3);
            Vector<double> averageBiasDerivatives = CalculateAverageVectorWithClipping(biasDerivatives, 3);
            //Weird conversion because this is single perceptron for now, will change
            for (int i = 0; i < oldWeights.Count; i++)
            {
                WeightVector[i] = oldWeights[i][0, 0];
            }
            WeightVector += averageWeightDerivatives.Multiply(learningRate);
            Vector<double> biasVector = neuralNetwork.GetBiasVector();
            biasVector += averageBiasDerivatives.Multiply(learningRate);
            for (int i = 0; i < oldWeights.Count; i++)
            {
                 oldWeights[i][0, 0] = WeightVector[i];
            }
            //Getting extreme expansion here, maybe train with smaller batches of examples
            neuralNetwork.SetWeightsToList(oldWeights);

        }
        //Returns a vector that has an L2 equal to or below the threshold
        Vector<double> ApplyClipping(Vector<double> vector, double threshold)
        {
            double L2 = vector.L2Norm();
            if (L2 < threshold)
            {
                return vector;
            }
            vector = vector.Multiply(threshold);
            vector = vector.Divide(L2);
            return vector;
        }
        //This function returns the raw average vector of a list without normalization
        Vector<double> CalculateAverageVector(List<Vector<double>> vectors)
        {
            Vector<double> sum = Vector<double>.Build.Dense(vectors[0].Count);
            for (int i = 0; i < vectors.Count; i++)
            {
                sum += vectors[i];
            }
            sum /= vectors.Count;
            return sum;
        }
        //This function uses L2 clipping to normalize vectors and then averages them
        Vector<double> CalculateAverageVectorWithClipping(List<Vector<double>> vectors, double threshold)
        {
            Vector<double> sum = Vector<double>.Build.Dense(vectors[0].Count);
            for (int i = 0; i < vectors.Count; i++)
            {
                Vector<double> clippedVector = ApplyClipping(vectors[i], threshold);
                sum += clippedVector;
            }
            sum /= vectors.Count;
            return sum;
        }
    }
}
