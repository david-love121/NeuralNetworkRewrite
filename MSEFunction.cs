using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Threading.Tasks;


namespace NeuralNetworkRewrite2024
{
    //Mean square error function, one of my cost functions
    internal class MSEFunction : Function
    {
        /// <summary>
        /// Returns the Mean Square Error based on the a - y supplied
        /// </summary>
        /// <param name="outputDifference"></param>
        /// <returns></returns>
        internal override double Compute(double outputDifference)
        {
            double result = Math.Pow(outputDifference, 2);
            return result;
        }
        /// <summary>
        /// Returns the derivative of Mean Square Error based on the a - y supplied
        /// </summary>
        /// <param name="outputDifference"></param>
        /// <returns></returns>
        internal override double ComputeDerivative(double outputDifference)
        {
            double result = 2 * outputDifference;
            return result;
        }
        internal Vector<double> Compute(Vector<double> differenceVector)
        {
            for (int i = 0; i < differenceVector.Count; i++)
            {
                differenceVector[i] = Compute(differenceVector[i]);
            }
            return differenceVector;
        }
        internal Vector<double> ComputeDerivative(Vector<double> differenceVector)
        {
            for (int i =0; i < differenceVector.Count; i++)
            {
                differenceVector[i] = ComputeDerivative(differenceVector[i]);
            }
            return differenceVector;
        }
        
    }
}
