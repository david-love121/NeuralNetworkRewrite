using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class LayerMetadata
    {
        //Weights is null if this is the last layer
        public double[][]? Weights { get; set; }
        public bool LastLayer { get; set; }
        public int Size { get; set; }
        public double Bias { get; set; }
        public Function ActivationFunction { get; set; }
        //Extracts data from an existing layer object
        public LayerMetadata(Layer layer)
        {
            ActivationFunction = layer.GetActivationFunction();
            int nextSize = layer.GetNextSize();
            if (nextSize < 0)
            {
                LastLayer = true;
            }
            if (!LastLayer) {
                Weights = new double[layer.size][];
                for (int i = 0; i < layer.size; i++)
                {
                    Weights[i] = new double[nextSize];
                    for (int k = 0; k < nextSize; k++)
                    {
                        Weights[i][k] = layer.GetNeuron(i).GetConnectorOut(k).GetWeight();
                    }
                }
            }
            Size = layer.GetSize();
            Bias = layer.GetBias();
        }
        [JsonConstructor]
        public LayerMetadata(double[][]? weights, bool lastLayer, int size, double bias, Function activationFunction)
        {
            Weights = weights;
            LastLayer = lastLayer;
            Size = size;
            Bias = bias;
            ActivationFunction = activationFunction;
        }

        public string Serialize()
        {
            return JsonSerializer.Serialize(this);
        }
    }
}
