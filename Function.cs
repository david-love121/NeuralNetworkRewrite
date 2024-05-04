using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal abstract class Function
    {
        internal abstract double Compute(double x);
        internal abstract double ComputeDerivative(double x);
    }
}
