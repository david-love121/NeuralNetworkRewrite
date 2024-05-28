using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetworkRewrite2024
{
    internal class Layer
    {
        private List<Neuron> neurons;
        //Null if this is the last layer, as there is no next layer
        //Matrix of weights cannot be generated if last layer
        private Layer? nextLayer;
        //Null if the network was constructed through training instead of from storage
        internal double[][]? PresetWeights { get; set; }
        internal bool FromStorage { get; set; }
        internal int size;
        Function activationFunction;
        double bias;
        int LayerIndex { get; set; }
        internal Layer(int size, Function activationFunction, int layerIndex, double bias = 1)
        {
            this.activationFunction = activationFunction;
            this.size = size;
            this.bias = bias;
            LayerIndex = layerIndex;
            neurons = new List<Neuron>();
            PopulateNeurons();
        }
        //This constructor is for building a layer from a saved neural network
        internal Layer(LayerMetadata metadata)
        {
            PresetWeights = metadata.Weights;
            FromStorage = true;
            size = metadata.Size;
            activationFunction = metadata.ActivationFunction;
            bias = metadata.Bias;
            neurons = new List<Neuron>();
            PopulateNeurons();

        }
        void PopulateNeurons()
        {
            for (int i = 0; i < this.size; i++)
            {
                Neuron neuron = new Neuron(this.activationFunction, this.bias);
                this.neurons.Add(neuron);
            }
        }
        //This overload is for constructing from preknown weights
        void PopulateNeurons(ref List<Neuron> neurons, int nextSize, double[,] weights)
        {
            for (int i = 0; i < this.size;  i++)
            {
                for (int k = 0; k < nextSize; k++)
                {
                    Neuron neuron = new Neuron(this.activationFunction, this.bias);
                }
            }
        }
        internal Function GetActivationFunction()
        {
            return activationFunction;
        }
        internal void RunNeurons()
        {
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].RunNeuron();
            }
        }
        internal void RunNeurons(double input)
        {
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].RunNeuron(input);
            }
        }
        internal void RunNeurons(Vector<double> inputs)
        {
            if (inputs.Count != this.size)
            {
                throw new ArgumentException("Inputs doesn't match number of neurons!");
            }
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].RunNeuron(inputs[i]);
            }
        }
        //Originally I only had one bias per layer, but this has changed, remove eventually
        internal double GetBias()
        {
            return bias;
        }
        internal Vector<double> GetBiasVector()
        {
            Vector<double> result = Vector<double>.Build.Dense(neurons.Count);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = neurons[i].GetBias();
            }
            return result;
        }
        internal int GetSize()
        {
            return size;
        }
        internal Neuron GetNeuron(int index)
        {
            return neurons[index];
        }
        public Vector<double> OutputLayerAsVector()
        {
            int size = this.GetSize();
            Vector<double> output = Vector<double>.Build.Dense(size);
            for (int i = 0; i < size; i++)
            {
                output[i] = GetNeuron(i).GetLastValue();
            }
            return output;
        }
        internal void RandomizeWeights(double range)
        {
            for (int i = 0; i < size; i++)
            {
                this.GetNeuron(i).RandomizeWeights(range);
            }
        }
        //Of format [NeuronIndex, nextNeuronIndex]
        internal Matrix<double> WeightsAsMatrix()
        {
            if (nextLayer is null)
            {
                throw new Exception("You cannot generate a weights matrix for the last layer.");
            }
            Matrix<double> weightsMatrix = Matrix<double>.Build.Dense(size, nextLayer.GetSize());
            for (int i = 0; i < size; i++)
            {
                for (int k = 0; k < nextLayer.GetSize(); k++)
                {
                    weightsMatrix[i, k] = GetNeuron(i).GetConnectorOut(k).GetWeight();
                }
            }
            return weightsMatrix;
        }
        internal void ConnectLayer(Layer nextLayer)
        {
            this.nextLayer = nextLayer;
        }
        internal void ChangeWeights(Matrix<double> weights)
        {
            if (nextLayer is null)
            {
                throw new Exception("No weights to change; last layer");
            }
            for (int i = 0; i < size; i++)
            {
                for (int k = 0; k < nextLayer.GetSize(); k++)
                {
                    double currentWeight = weights[i, k];
                    GetNeuron(i).GetConnectorOut(k).SetWeight(currentWeight);
                }
            }
        }
        internal int GetNextSize()
        {
            if (nextLayer is null)
            {
                return -1;
            }
            return nextLayer.GetSize();
        }
        internal void ChangeBias(double bias)
        {
            this.bias = bias;
            //Consider changing neurons to reference their bias from the layer object
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].SetBias(bias);
            }
        }
        internal void ChangeBias(Vector<double> bias)
        {
            if (bias.Count !=  neurons.Count)
            {
                throw new ArgumentException("Dimensionality must match!");
            }
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].SetBias(bias[i]);
            }
        }
        internal Vector<double> GetPreactivationValues()
        {
            Vector<double> results = Vector<double>.Build.Dense(size);
            for (int i = 0; i < size; i++)
            {
                results[i] = this.GetNeuron(i).GetLastPreactivationValue();
            }
            return results;
        }
        internal Vector<double> GetActiavtionValues()
        {
            Vector<double> results = Vector<double>.Build.Dense(size);
            for (int i = 0; i < size; i++)
            {
                results[i] = this.GetNeuron(i).GetLastValue();
            }
            return results;
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            int nextSize = -1;
            if (nextLayer != null)
            {
                nextSize = nextLayer.GetSize();
            }
            //nextSize is -1 if last layer
            sb.Append($"[{size},{nextSize},{bias},{activationFunction.ToString()},");
            for (int i = 0; i < size; i++)
            {
                for (int k = 0; k < nextSize; k++)
                {
                    double currentWeight = GetNeuron(i).GetConnectorOut(k).GetWeight();
                    sb.Append(currentWeight);
                    sb.Append('|');
                }
            }
            sb.Append(']');
            return sb.ToString();
        }
    }
}
