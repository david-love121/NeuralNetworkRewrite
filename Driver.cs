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
        //Todo: abstract out data classes
        private NeuralNetwork neuralNetwork;
        readonly int[] layerSizes;
        Function dataFunction;
        DataGenerator dataGenerator;
        DataList dataList;
        //Defines the activation function the neurons use
        Function activationFunction;
        List<double> outputs;
        //Hard coded characteristics of network are defined here
        internal Driver() 
        {
            this.activationFunction = new SigmoidFunction();
            dataFunction = new SinFunction();
            dataGenerator = new DataGenerator(dataFunction, 100, 1);
            string irisDataPath = @"C:\Users\david\source\repos\NeuralNetworkRewrite\data\iris.data";
            dataList = new DataList(irisDataPath);
            bool categorical = false;
            if (categorical)
            {
                layerSizes = [dataList.GetContainer(0).GetFeatures().Length, 3, 4, dataList.GetContainer(0).NumCategories];
            } else
            {
                layerSizes = [1, 3, 4, 1];
            }
            neuralNetwork = new NeuralNetwork(layerSizes, activationFunction, false, categorical, 1);
            outputs = new List<double>();
           

        }
        internal Matrix<double> GetScores(int totalPoints)
        {
            //Seperate index in case I want to use a step different than 1 
            int index = 0;
            //Create a matrix the size of [totalPoints, LastLayerSize]
            Matrix<double> scores = Matrix<double>.Build.Dense(totalPoints, layerSizes[^1]);
            //i represents the x value of the function, moves with step
            //index represents the index we're accessing from dataGenerator
            outputs.Clear();
            for (int i = 0; i < totalPoints; i += 1)
            {
                Vector<double> outputVector = neuralNetwork.RunNetwork(i);
                Vector<double> score;
                if (neuralNetwork.Categorical)
                {
                    DataContainer dataContainer = dataList.GetContainer(i);
                    Vector<double> oneHotCodedVector = Vector<double>.Build.Dense(dataContainer.GetOneHotCoded());
                    score = neuralNetwork.ScoreOutput(outputVector, oneHotCodedVector);
                }
                else
                {
                    Vector<double> expectedOutput = Vector<double>.Build.Dense(1, dataGenerator.GetDataPoint(i));
                    //Todo: Add categorical cross entropy loss function
                    score = neuralNetwork.ScoreOutput(outputVector, expectedOutput);
                }//Based on average of MSE 
                scores.SetRow(index, score);
                index++;
                
            }
            return scores;
        }
        internal double GetAverageScore()
        {
            Matrix<double> scores = GetScores(dataGenerator.GetSizeData());
            double scoreSum = 0.0;
            for (int i = 0; i < scores.RowCount; i++)
            {
                Vector<double> currentRow = scores.Row(i);
                scoreSum += currentRow.Sum();
            }
            double averageScores = scoreSum / dataGenerator.GetSizeData();
            return averageScores;
            
        }
        internal List<Matrix<double>> TrainEvolutionBased(int tests)
        {
            double minCost = double.MaxValue;
            List<Matrix<double>> bestResult = new List<Matrix<double>>();
            for (int i = 0; i < tests; i++)
            {

                //Verify that summation works 
                double averageCost = GetAverageScore();
                if (averageCost < minCost)
                {
                    minCost = averageCost;
                    bestResult = neuralNetwork.GetWeightsMatrixList();                    
                }
                neuralNetwork.RandomizeWeights(2);
                
            }
            neuralNetwork.SetWeightsToList(bestResult);
            return bestResult;
        }
        internal void TrainBackpropagationBased(int epochs, int batchSize, double learningRate = 1.0)
        {
            int totalPoints = dataGenerator.GetSizeData();
            int batchesPerTrainingCycle = totalPoints / batchSize;
            //Stores average batch changes, matrix represents the changes for a layer, inner list represents changes for a pass, outer list represents all passes
            List<List<Matrix<double>>> weightDerivativesL = new List<List<Matrix<double>>>();
            List<List<Vector<double>>> biasDerivativesL = new List<List<Vector<double>>>();
            //Stores indvidual batch changes
            List<List<Matrix<double>>> weightDerivativeCollection = new List<List<Matrix<double>>>(); //Maybe create a different data types for these, messy
            List<List<Vector<double>>> biasDerivativeCollection = new List<List<Vector<double>>>();
            for (int a = 0; a < epochs; a++) 
            {
                for (int b = 0; b < batchesPerTrainingCycle; b++) 
                {
                    weightDerivativeCollection.Clear();
                    biasDerivativeCollection.Clear();
                    for (int i = batchSize*b; i < batchSize*(b+1); i++)
                    {
                        //a = func(z) z = y*w + b
                        int layerIndex = layerSizes.Length - 1;
                        Layer lastLayer = neuralNetwork.GetLayer(layerIndex);
                        Vector<double> expectedOutput;
                        Vector<double> neuralNetworkOutput;
                        //del stands for partial derivative
                        //Derivative of Cost function wrt a
                        Vector<double> delCDelAVector;
                        if (neuralNetwork.Categorical)
                        {
                            DataContainer dataContainer = dataList.GetContainer(i);
                            expectedOutput = Vector<double>.Build.DenseOfArray(dataContainer.GetOneHotCoded());
                            Vector<double> inputVector = Vector<double>.Build.DenseOfArray(dataContainer.GetFeatures());
                            neuralNetworkOutput = neuralNetwork.RunNetwork(inputVector);

                            delCDelAVector = neuralNetworkOutput - expectedOutput;
                        }
                        else
                        {
                            expectedOutput = Vector<double>.Build.Dense(1, dataGenerator.GetDataPoint(i));
                            neuralNetworkOutput = neuralNetwork.RunNetwork(i);
                            MSEFunction mseFunction = new MSEFunction();
                            Vector<double> differenceVector = neuralNetworkOutput - expectedOutput;
                            delCDelAVector = mseFunction.ComputeDerivative(differenceVector);
                        }
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
                        List<Vector<double>> biasDerivativesList = new List<Vector<double>>(neuralNetwork.GetLayers().Count);
                        Matrix<double> weightDerivativesLayer = Matrix<double>.Build.Dense(nextLayer.size, prevLayer.size);
                        Vector<double> biasDerivativesLayer = Vector<double>.Build.Dense(prevLayer.GetSize());

                        //Derivative of z wrt w when connected to each next neuron
                        Vector<double> delZDelW = nextLayer.GetActiavtionValues();
                        Vector<double> endChain = delADelZ.PointwiseMultiply(delCDelAVector);
                        

                        //The derivative of z wrt b is 1, so this is simply endchain
                        for (int k = 0; k < prevLayer.size; k++)
                        {
                            biasDerivativesLayer[k] = endChain[k];
                        }
                        //Filling from the end because we calculate from the last layer back
                        int arrayIndex = 0;
                        for (int k = 0; k < nextLayer.size; k++)
                        {
                            for (int j = 0; j < prevLayer.size; j++) {
                                weightDerivativesLayer[k, j] = endChain[j] * delZDelW[k];
                                
                            }
                        }
                        //The bias affects each preactivation value which is the input for delAdelZ
                        //Todo: Update so individual neurons can have different biases

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
                        biasDerivativesList.Add(biasDerivativesLayer);
                        weightDerivativesList.Add(weightDerivativesLayer);
                        //Fills weightDerivativesList and biasDerivatives with results for the rest of the layers
                        if (layerIndex >= 0)
                        {
                            Backpropagate(delCDelA_1, layerIndex, arrayIndex, ref weightDerivativesList, ref biasDerivativesList);
                        }
                        //Reverse to put changes for first layer at front
                        weightDerivativesList.Reverse();
                        biasDerivativesList.Reverse();
                        //Each row represents the changes for an individual layer, columns neurons
                        
                        weightDerivativeCollection.Add(weightDerivativesList);
                        biasDerivativeCollection.Add(biasDerivativesList);
                    }
                    //Maybe clip here
                    List<Matrix<double>> averageWeightDerivative = CalculateAverageMatrices(weightDerivativeCollection);
                    List<Vector<double>> averageBiasDerivative = CalculateAverageListVector(biasDerivativeCollection);
                    weightDerivativesL.Add(averageWeightDerivative);
                    biasDerivativesL.Add(averageBiasDerivative);
                }
                AdjustNetwork(learningRate, 3, weightDerivativesL, biasDerivativesL);
            }
            
        }
        internal void Backpropagate(Vector<double> delCDelA, int layerIndex, int arrayIndex, ref List<Matrix<double>> wd, ref List<Vector<double>> bd)
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
            if (layerIndex > 0)
            {
                layerIndex--;
                //Ensure bias derivative gets the end bias value
                Layer nextLayer = neuralNetwork.GetLayer(layerIndex);
                Matrix<double> weightDerivativesLayer = Matrix<double>.Build.Dense(nextLayer.size, prevLayer.size);
                Vector<double> biasDerivativesLayer = Vector<double>.Build.Dense(prevLayer.size);
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
                wd.Add(weightDerivativesLayer);
                for (int k = 0; k < prevLayer.size; k++)
                {
                    biasDerivativesLayer[k] = endChain[k];
                }
                bd.Add(biasDerivativesLayer);
                arrayIndex++;
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
                Backpropagate(delCDelA_1, layerIndex, arrayIndex, ref wd, ref bd);
            }
            else
            {
                Vector<double> endChain = delADelZ.PointwiseMultiply(delCDelA);
                Vector<double> biasDerivativesLayer = Vector<double>.Build.Dense(prevLayer.size);
                for (int k = 0; k < prevLayer.size; k++)
                {
                    biasDerivativesLayer[k] = endChain[k];
                }
                bd.Add(biasDerivativesLayer);
            }
        }
        //weightDerivatives and biasDerivatives need to be updated to lists of matrices, but do this once it works with vectors
        internal void AdjustNetwork(double learningRate, double threshold, List<List<Matrix<double>>> weightDerivatives, List<List<Vector<double>>> biasDerivatives)
        {
            List<Matrix<double>> oldWeights = neuralNetwork.GetWeightsMatrixList();
            List<Vector<double>> oldBiases = neuralNetwork.GetBiasVectorList();
            //For testing single layer perceptron networks
            //Derivative wrt cost after clipping based on L2
            List<Matrix<double>> averageWeightDerivatives = CalculateAverageMatricesWithClipping(weightDerivatives, threshold);
            List<Matrix<double>> weightChanges = new List<Matrix<double>>();
            List<Vector<double>> averageBiasDerivatives = CalculateAverageListVectorWithClipping(biasDerivatives, threshold);
            List<Vector<double>> biasChanges = new List<Vector<double>>();
            //Weird conversion because this is single perceptron for now, will change
            for (int i = 0; i < averageWeightDerivatives.Count; i++)
            {
                Matrix<double> changes;
                changes = oldWeights[i] - averageWeightDerivatives[i].Multiply(learningRate);
                weightChanges.Add(changes); 
            }          
            for (int i = 0; i < averageBiasDerivatives.Count; i++)
            {
                Vector<double> changes;
                changes = oldBiases[i] - averageBiasDerivatives[i].Multiply(learningRate);
                biasChanges.Add(changes);

            }
            
            //Getting extreme expansion here, maybe train with smaller batches of examples
            neuralNetwork.SetWeightsToList(weightChanges);
            neuralNetwork.SetBiasToList(biasChanges);

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
        List<Vector<double>> CalculateAverageListVector(List<List<Vector<double>>> vectors)
        {
            List<Vector<double>> sumBatches = vectors[0];
            for (int i = 1; i < vectors.Count; i++)
            {
                for (int k = 0; k < vectors[i].Count; k++)
                {
                    sumBatches[k] += vectors[i][k];
                }
            }
            for (int i = 0; i < sumBatches.Count; i++)
            {
                sumBatches[i] = sumBatches[i].Divide(vectors.Count);
            }
            return sumBatches;
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
        List<Vector<double>> CalculateAverageListVectorWithClipping(List<List<Vector<double>>> vectors, double threshold)
        {
            List<Vector<double>> sumBatches = vectors[0];
            for (int i = 0; i < sumBatches.Count; i++)
            {
                sumBatches[i] = ApplyClipping(sumBatches[i], threshold);
            }
            for (int i = 1; i < vectors.Count; i++)
            {
                for (int k = 0; k < vectors[i].Count; k++)
                {
                    Vector<double> currentVector = vectors[i][k];
                    Vector<double> clippedVector = ApplyClipping(currentVector, threshold);
                    sumBatches[k] += clippedVector;
                }
            }
            for (int i = 0; i < sumBatches.Count; i++)
            {
                sumBatches[i] = sumBatches[i].Divide(vectors.Count);
            }
            return sumBatches;
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
