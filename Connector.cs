using NeuralNet2023;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class Connector
    {
        //A connector begins at the first neuron
        Neuron firstNeuron;
        Neuron secondNeuron;
        double weight;
        internal Connector(Neuron firstNeuron, Neuron secondNeuron, double weight = 1)
        {
            this.firstNeuron = firstNeuron;
            this.secondNeuron = secondNeuron;
            this.weight = weight;
        }
        internal void RunData(double value)
        {
            secondNeuron.AddInput(value * weight);
        }
        internal void RandomizeWeight(double range)
        {
            Random random = ManagedRandom.getRandom();
            weight = random.NextDouble()*range;
        }
        internal double GetWeight()
        {
            return weight;
        }
        internal void SetWeight(double weight)
        {
            this.weight = weight;
        }
        internal void ChangeWeight(double change)
        {
            this.weight = weight + change;
        }
    }
}
