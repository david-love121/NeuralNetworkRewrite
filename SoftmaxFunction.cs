using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class SoftmaxFunction 
    {
        internal Vector<double> Compute(Vector<double> vect)
        {
            Vector<double> outputs = Vector<double>.Build.Dense(vect.Count);
            double sum = 0;
            for (int i = 0; i < vect.Count; i++)
            {
                //e^input
                sum += Math.Exp(vect[i]);
            }
            for (int i = 0; i < vect.Count; i++)
            {
                outputs[i] = Math.Exp(vect[i]) / sum;
            }
            return outputs;
        }
        

        
    }
}
