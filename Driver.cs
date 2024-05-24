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
        readonly int[] layerSizes = { 3, 2, 1 };
        Function dataFunction;
        DataGenerator dataGenerator;
        Function activationFunction;
        List<double> outputs;
        //Defines the activation function the neurons use
        internal Driver() 
        {
            this.activationFunction = new LinearFunction(0, 1);
            dataFunction = new LinearFunction(2, 3);
            dataGenerator = new DataGenerator(dataFunction, 100, 1);
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
                }
                neuralNetwork.RandomizeWeights();
                
            }

            return bestResult;
        }
        //Update to support multilayer
        internal void TrainBackpropagationBased(int epochs, int batchSize, double learningRate = 1.0)
        {
            int totalPoints = dataGenerator.GetSizeData();
            int batchesPerTrainingCycle = totalPoints / batchSize;
            Vector<double> scoreVector = GetScores(totalPoints);
            //Stores average batch changes, matrix represents the changes for a layer, inner list represents changes for a pass, outer list represents all passes
            List<List<Matrix<double>>> weightDerivativesL = new List<List<Matrix<double>>>();
            List<Vector<double>> biasDerivativesL = new List<Vector<double>>();
            //Stores indvidual batch changes
            List<List<Matrix<double>>> weightDerivativeCollection = new List<List<Matrix<double>>>(); //Maybe create a different data types for these, messy
            List<Vector<double>> biasDerivativeCollection = new List<Vector<double>>();
            for (int a = 0; a < epochs; a++) 
            {
                for (int b = 0; b < batchesPerTrainingCycle; b++) 
                {
                    weightDerivativeCollection.Clear();
                    biasDerivativeCollection.Clear();
                    for (int i = batchSize*b; i < batchSize*(b+1); i++)
                        {
                        int layerIndex = layerSizes.Length - 1;
                        

                        //a = func(z) z = y*w + b
                        Layer lastLayer = neuralNetwork.GetLayer(layerIndex);
                        //Vectors of 1 being used for scalability later 
                        Vector<double> expectedOutput = Vector<double>.Build.Dense(1, dataGenerator.GetDataPoint(i));
                        Vector<double> neuralNetworkOutput = neuralNetwork.RunNetwork(i);
                        //del stands for partial derivative
                        MSEFunction mseFunction = new MSEFunction();
                        Vector<double> differenceVector = expectedOutput - neuralNetworkOutput;
                        //Derivative of Cost function (MSE) wrt a
                        Vector<double> delCDelAVector = mseFunction.ComputeDerivative(differenceVector);
                        Vector<double> LastPreactivationValues = lastLayer.GetPreactivationValues();
                        //Derivative of a wrt z
                        Vector<double> delADelZ = Vector<double>.Build.Dense(LastPreactivationValues.Count);
                        for (int k = 0; k < LastPreactivationValues.Count; k++)
                        {
                            //The last layer's del A del Z is the derivative of the activation function with the y values plugged in
                            //Always only one value in a network with one output, differs in a network with multiple
                            double DelValue = lastLayer.GetActivationFunction().ComputeDerivative(LastPreactivationValues[k]);
                            delADelZ[k] = DelValue;
                        }
                        layerIndex = layerIndex - 1;
                        //Represents 1 layer backwards from the end, working on the weights between next and prev
                        Layer nextLayer = neuralNetwork.GetLayer(layerIndex);
                        //Prev layer is later in the output than next, because working backwards
                        Layer prevLayer = neuralNetwork.GetLayer(layerIndex + 1);
                        //Format [LayerBehindNeurons, LayerNeurons] because we're working backwards
                        //Vector of matrices because each has different dimensionality. Index of the vector represents layer
                        List<Matrix<double>> weightDerivativesList = new List<Matrix<double>>(neuralNetwork.GetLayers().Count); 
                        Matrix<double> weightDerivativesLayer = Matrix<double>.Build.Dense(nextLayer.size, prevLayer.size);
                        Vector<double> biasDerivatives = Vector<double>.Build.Dense(neuralNetwork.GetLayers().Count);

                        //Summed because changing one a value changes the cost for all neurons
                        double delCDelA = delCDelAVector.Sum();
                        //Derivative of z wrt w when connected to each next neuron
                        Vector<double> delZDelW = nextLayer.GetActiavtionValues();
                        Vector<double> endChain = delADelZ.Multiply(delCDelA);
                       // Vector<double> delCDelW = endChain.Multiply(delZDelW);
                        //The derivative of z wrt b is 1, so this is simply endchain
                        Vector<double> delCDelB = endChain;
                        //Filling from the end because we calculate from the last layer back
                        int arrayIndex = 0;
                        for (int k = 0; k < nextLayer.size; k++)
                        {
                            for (int j = 0; j < prevLayer.size; j++) {
                                weightDerivativesLayer[k, j] = endChain[j] * delZDelW[k];
                            }
                        }
                        //The bias affects each preactivation value which is the input for delAdelZ
                        biasDerivatives[biasDerivatives.Count - arrayIndex - 1] = delCDelB.Sum();
                        arrayIndex++; //From end
                        //delCdelA_1 becomes a vector, each neuron in the next layer has different weights affecting it
                        Vector<double> delCDelA_1 = Vector<double>.Build.Dense(nextLayer.size);
                        for (int k = 0; k < nextLayer.size; k++)
                        {
                            for (int z = 0; z < prevLayer.size; z++)
                            {
                                Neuron currentNeuron = nextLayer.GetNeuron(k);

                                double currentDelCDelA_1 = currentNeuron.GetConnectorOut(z).GetWeight() * endChain[z];
                                delCDelA_1[k] += currentDelCDelA_1;
                            }

                        }
                        weightDerivativesList.Add(weightDerivativesLayer);
                        //Fills weightDerivativesList and biasDerivatives with results for the rest of the layers
                        if (layerIndex >= 0)
                        {
                            Backpropagate(delCDelA_1, layerIndex, arrayIndex, ref weightDerivativesList, ref biasDerivatives);
                        }
                        //Reverse to put changes for first layer at front
                        weightDerivativesList.Reverse();
                        weightDerivativeCollection.Add(weightDerivativesList);
                        biasDerivativeCollection.Add(biasDerivatives);
                    }
                    //Maybe clip here
                    List<Matrix<double>> averageWeightDerivative = CalculateAverageMatrices(weightDerivativeCollection);
                    Vector<double> averageBiasDerivative = CalculateAverageVector(biasDerivativeCollection);
                    weightDerivativesL.Add(averageWeightDerivative);
                    biasDerivativesL.Add(averageBiasDerivative);
                }
                AdjustNetwork(learningRate, 3, weightDerivativesL, biasDerivativesL);
            }
            
        }
        internal void Backpropagate(Vector<double> delCDelA, int layerIndex, int arrayIndex, ref List<Matrix<double>> wd, ref Vector<double> bd)
        {

            Layer prevLayer = neuralNetwork.GetLayer(layerIndex);
            Vector<double> LastPreactivationValues = prevLayer.GetPreactivationValues();
            //Derivative of a wrt z
            Vector<double> delADelZ = Vector<double>.Build.Dense(LastPreactivationValues.Count);
            for (int k = 0; k < LastPreactivationValues.Count; k++)
            {
                double DelValue = prevLayer.GetActivationFunction().ComputeDerivative(LastPreactivationValues[k]);
                delADelZ[k] = DelValue;
            }
            layerIndex--;
            //Ensure bias derivative gets the end bias value
            Layer nextLayer = neuralNetwork.GetLayer(layerIndex);
            Matrix<double> weightDerivativesLayer = Matrix<double>.Build.Dense(nextLayer.size, prevLayer.size);
            //Derivative of z wrt w when connected to each next neuron
            Vector<double> delZDelW = nextLayer.GetActiavtionValues();
            Vector<double> endChain = delADelZ.PointwiseMultiply(delCDelA);
            for (int k = 0; k < nextLayer.size; k++)
            {
                for (int j = 0; j < prevLayer.size; j++)
                {
                    weightDerivativesLayer[k, j] = endChain[j] * delZDelW[k];
                }
            }
            Vector<double> delCDelB = endChain;
            wd.Add(weightDerivativesLayer);
            bd[bd.Count - arrayIndex - 1] = delCDelB.Sum();
            layerIndex--;
            arrayIndex++;
            prevLayer = nextLayer;
            
            if (layerIndex >= 0)
            {
                nextLayer = neuralNetwork.GetLayer(layerIndex);
                Vector<double> delCDelA_1 = Vector<double>.Build.Dense(nextLayer.size);
                for (int k = 0; k < nextLayer.size; k++)
                {
                    for (int z = 0; z < prevLayer.size; z++)
                    {
                        Neuron currentNeuron = nextLayer.GetNeuron(k);

                        double currentDelCDelA_1 = currentNeuron.GetConnectorOut(z).GetWeight() * endChain[z];
                        delCDelA_1[k] += currentDelCDelA_1;
                    }
                    Backpropagate(delCDelA_1, layerIndex, arrayIndex, ref wd, ref bd);
                }
            } else
            {
                bd[bd.Count - arrayIndex - 1] = delCDelB.Sum();
            }
        }
        //weightDerivatives and biasDerivatives need to be updated to lists of matrices, but do this once it works with vectors
        internal void AdjustNetwork(double learningRate, double threshold, List<List<Matrix<double>>> weightDerivatives, List<Vector<double>> biasDerivatives)
        {
            List<Matrix<double>> oldWeights = neuralNetwork.GetWeightsMatrixList();
            //For testing single layer perceptron networks
            //Derivative wrt cost after clipping based on L2
            List<Matrix<double>> averageWeightDerivatives = CalculateAverageMatricesWithClipping(weightDerivatives, threshold);
            List<Matrix<double>> weightChanges = new List<Matrix<double>>();
            Vector<double> averageBiasDerivatives = CalculateAverageVectorWithClipping(biasDerivatives, threshold);
            //Weird conversion because this is single perceptron for now, will change
            for (int i = 0; i < averageWeightDerivatives.Count; i++)
            {
                Matrix<double> changes = oldWeights[i] + averageWeightDerivatives[i].Multiply(learningRate);
                weightChanges.Add(changes); 
            }          
            Vector<double> biasVector = neuralNetwork.GetBiasVector();
            biasVector += averageBiasDerivatives.Multiply(learningRate);
            //Getting extreme expansion here, maybe train with smaller batches of examples
            neuralNetwork.SetWeightsToList(weightChanges);
            neuralNetwork.SetBiasToList(biasVector);

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
        Matrix<double> ApplyClipping(Matrix<double> matrix, double threshold)
        {
            double L2 = matrix.L2Norm();
            if (L2 < threshold)
            {
                return matrix;
            }
            matrix = matrix.Multiply(threshold);
            matrix = matrix.Divide(L2);
            return matrix;
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
        List<Matrix<double>> CalculateAverageMatrices(List<List<Matrix<double>>> matricesList) 
        {
            int size = matricesList[0].Count;
            List<Matrix<double>> totalSum = new List<Matrix<double>>();
            List<Matrix<double>> finalOutput = new List<Matrix<double>>();
            for (int i = 0; i < matricesList.Count; i++)
            {
                List<Matrix<double>> currentSet = matricesList[i];
                for (int k = 0; k < size; k++)
                {
                    if (totalSum.Count < size)
                    {
                        totalSum.Add(currentSet[k]);
                    } else
                    {
                        totalSum[k] += currentSet[k];
                    }
                }
            }
            for (int i = 0; i < size; i++)
            {
                Matrix<double> averageMatrix = totalSum[i].Divide(matricesList.Count);
                finalOutput.Add(averageMatrix);
            }
            return finalOutput;
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
        List<Matrix<double>> CalculateAverageMatricesWithClipping(List<List<Matrix<double>>> matricesList, double threshold)
        {
            int size = matricesList[0].Count;
            List<Matrix<double>> totalSum = new List<Matrix<double>>();
            List<Matrix<double>> finalOutput = new List<Matrix<double>>();
            for (int i = 0; i < matricesList.Count; i++)
            {
                List<Matrix<double>> currentSet = matricesList[i];
                for (int k = 0; k < size; k++)
                {
                    Matrix<double> clippedMatrix = ApplyClipping(currentSet[k], threshold);
                    if (totalSum.Count < size)
                    {
                        totalSum.Add(clippedMatrix);
                    }
                    else
                    {
                        totalSum[k] += clippedMatrix;
                    }
                }
            }
            for (int i = 0; i < size; i++)
            {
                Matrix<double> averageMatrix = totalSum[i].Divide(matricesList.Count);
                finalOutput.Add(averageMatrix);
            }
            return finalOutput;
        }
        internal void SaveNetworkToStorage(string path)
        {
            neuralNetwork.SaveNetworkToStorage(path);
        }
        internal void TestSerialization(string path)
        {
            neuralNetwork.SaveNetworkToStorage(path);
            NeuralNetwork reconstructed = NeuralNetwork.LoadNetworkFromStorage(path);
            return;
        }
    }
}
