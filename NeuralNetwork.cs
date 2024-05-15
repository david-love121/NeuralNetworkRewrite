using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetworkRewrite2024
{
    internal class NeuralNetwork
    {
        List<Layer> layers;
        
        //Including input and output layers
        internal NeuralNetwork(int[] layerSizes, Function activationFunction, double bias = 1)
        {
            layers = new List<Layer>();
            PopulateLayers(ref layers, layerSizes, activationFunction, bias);
            ConnectNeuronsAndLayers();
        }
        static void PopulateLayers(ref List<Layer> layers, int[] layerSizes, Function activationFunction, double bias)
        {
            for (int i = 0; i < layerSizes.Length; i++)
            {
                Layer layer = new Layer(layerSizes[i], activationFunction, bias);
                layers.Add(layer);
            }
        }
        void ConnectNeuronsAndLayers()
        {
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Layer selectedLayer = layers[i];
                Layer nextLayer = layers[i + 1];
                selectedLayer.ConnectLayer(nextLayer);
                for (int k = 0; k < selectedLayer.GetSize(); k++)
                {
                    for (int j = 0; j < nextLayer.GetSize(); j++)
                    {
                        Connector connector = new Connector(selectedLayer.GetNeuron(k), nextLayer.GetNeuron(j), 1);
                        selectedLayer.GetNeuron(k).AddConnectionOut(connector);
                    }
                }
            }
        }
        internal Vector<double> RunNetwork(double input)
        {
            layers[0].RunNeurons(input);
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].RunNeurons();

            }
            Layer outputLayer = layers[layers.Count - 1];
            Vector<double> vectorizedOutput = outputLayer.OutputLayerAsVector();
            return vectorizedOutput;
            
        }
        internal double ScoreOutput(Vector<double> input, Vector<double> expectedOutput)
        {
            MSEFunction mseFunction = new MSEFunction();
            Vector<double> differenceVector = input - expectedOutput;
            Vector<double> MSEVector = mseFunction.Compute(differenceVector);
            double sum = MSEVector.Sum();
            double average = sum / MSEVector.Count;
            return average;
        }
        internal void RandomizeWeights()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].RandomizeWeights();
                
            }
            
        }
        internal List<Matrix<double>> GetWeightsMatrixList()
        {
            List<Matrix<double>> result = new List<Matrix<double>>();
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Matrix<double> nextMatrix = layers[i].WeightsAsMatrix();
                result.Add(nextMatrix);

            }
            return result;
        }
        internal Vector<double> GetBiasVector()
        {
            Vector<double> result = Vector<double>.Build.Dense(layers.Count);
            for (int i = 0; i < layers.Count; i++)
            {
                result[i] = layers[i].GetBias();
            }
            return result;
        }
        //This function runs with backward indexing. Because backprop starts from the last layer and moves back,
        //The 0 index of the list represents the changes to the last set of weights.
        internal void SetWeightsToList(List<Matrix<double>> list)
        {
            int total = list.Count;
            for (int i = list.Count - 1; i >= 0; i--)
            {
                
                Layer currentLayer = layers[i];
                Matrix<double> currentMatrix = list[total - i - 1];
                currentLayer.ChangeWeights(currentMatrix);
            }
        }
        
        internal Layer GetLayer(int index)
        {
            return layers[index];
        }
    }
}
