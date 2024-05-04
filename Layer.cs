using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

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
        internal int size;
        Function activationFunction;
        double bias;
        internal Layer(int size, Function activationFunction, double bias = 1)
        {
            this.activationFunction = activationFunction;
            this.size = size;
            this.bias = bias;
            neurons = new List<Neuron>();
            PopulateNeurons(ref neurons, size);
        }
        void PopulateNeurons(ref List<Neuron> neurons, int size)
        {
            for (int i = 0; i < size; i++)
            {
                Neuron neuron = new Neuron(activationFunction, bias);
                neurons.Add(neuron);
            }

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
        internal void RandomizeWeights()
        {
            for (int i = 0; i < size; i++)
            {
                this.GetNeuron(i).RandomizeWeights();
            }
        }
        //Of format [nextNeuronIndex, NeuronIndex]
        internal Matrix<double> WeightsAsMatrix()
        {
            if (nextLayer is null)
            {
                throw new Exception("You cannot generate a weights matrix for the last layer.");
            }
            Matrix<double> weightsMatrix = Matrix<double>.Build.Dense(nextLayer.GetSize(), size);
            for (int i = 0; i < nextLayer.GetSize(); i++)
            {
                for (int k = 0; k < size; k++)
                {
                    weightsMatrix[i, k] = GetNeuron(k).GetConnectorOut(i).GetWeight();
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
            for (int i = 0; i < nextLayer.GetSize(); i++)
            {
                for (int k = 0; k < size; k++)
                {
                    double currentWeight = weights[i, k];
                    GetNeuron(k).GetConnectorOut(i).SetWeight(currentWeight);
                }
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
    }
}
