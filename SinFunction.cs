using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class SinFunction : Function
    {
        internal override double Compute(double x)
        {
            return Math.Sin(x);
        }

        internal override double ComputeDerivative(double x)
        {
            return Math.Cos(x);
        }
    }
}
