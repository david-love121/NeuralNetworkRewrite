using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class SigmoidFunction : Function
    {
        internal override double Compute(double x)
        {
            double denominator = 1 + Math.Exp(-x);
            return 1.0 / denominator;
        }

        internal override double ComputeDerivative(double x)
        {
            double Computed = Compute(x);
            double derivative = Computed * (1.0 - Computed);
            return derivative;
        }
    }
}
