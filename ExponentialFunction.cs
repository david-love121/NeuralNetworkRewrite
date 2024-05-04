using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class ExponentialFunction : Function
    {
        //y = b^x
        private double baseValue;
        internal ExponentialFunction(double baseValue)
        {
            this.baseValue = baseValue;
        }
        internal override double Compute(double x)
        {
            double result = Math.Pow(baseValue, x);
            return result;
        }

        internal override double ComputeDerivative(double x)
        {
            double originalValue = Compute(x);
            double derivative = Math.Log(baseValue) * originalValue;
            return derivative;
        }
    }
}
