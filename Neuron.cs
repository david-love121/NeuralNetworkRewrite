﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetworkRewrite2024
{
    internal class Neuron
    {
        //TODO: Fix properties so they can be serialized
        private List<double> dataIn;
        private List<Connector> dataOut;
        private Function activationFunction;
        private double bias;
        private double lastValue;
        private double lastPreactivationValue;
        internal Neuron(Function acitvationFunction, double bias)
        {
            dataIn = new List<double>();
            this.activationFunction = acitvationFunction;
            this.bias = bias;
            dataOut = new List<Connector>();
        }
        internal double RunNeuron()
        {
            double LastValues = dataIn.Sum();
            LastValues = LastValues + bias;
            lastPreactivationValue = LastValues;
            double result = activationFunction.Compute(LastValues);
            for (int i = 0; i < dataOut.Count; i++)
            {
                dataOut[i].RunData(result);
            }
            dataIn.Clear();
            lastValue = result; 
            return result;
        }
        //For running the input layer with inputs
        internal double RunNeuron(double input)
        {
            double LastValues = dataIn.Sum();
            LastValues = LastValues + bias + input;
            double result = activationFunction.Compute(LastValues);
            for (int i = 0; i < dataOut.Count; i++)
            {
                dataOut[i].RunData(result);
            }
            dataIn.Clear();
            lastValue = result;
            return result;
        }
        internal void AddInput(double value)
        {
            dataIn.Add(value);
            return;
        }
        internal void AddConnectionOut(Connector connector)
        {
            dataOut.Add(connector);
        }
        internal Vector<double> GetConnectionsOut()
        {
            Vector<double> connections = Vector<double>.Build.Dense(dataOut.Count);
            for (int i = 0; i < dataOut.Count; i++)
            {
                connections[i] = dataOut[i].GetWeight();
            }
            return connections;
        }
        internal double GetLastValue()
        {
            return lastValue;
        }
        internal double GetBias()
        {
            return bias;
        }
        internal void SetBias(double bias)
        {
            this.bias = bias;
        }
        internal void RandomizeWeights(double range)
        {
            for (int i = 0; i < dataOut.Count; i++)
            {
                dataOut[i].RandomizeWeight(range);
            }
        }
        internal Connector GetConnectorOut(int index)
        {
            return dataOut[index];
        }
        internal double GetLastPreactivationValue()
        {
            return lastPreactivationValue;
        }
        
    }
}
