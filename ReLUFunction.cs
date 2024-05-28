using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class ReLUFunction : Function
    {
        internal override double Compute(double x)
        {
            if (x < 0)
            {
                return 0;
            } else
            {
                return x;
            }
        }

        internal override double ComputeDerivative(double x)
        {
            if (x < 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
    }
}
