using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class LinearFunction : Function
    {
        //y = mx + b
        private double yIntercept;
        private double slope; 
        internal LinearFunction(double yIntercept, double slope)
        {
            this.yIntercept = yIntercept;
            this.slope = slope;
        }
        internal override double Compute(double x)
        {
            double result = yIntercept + x * slope;
            return result;
        }
        internal override double ComputeDerivative(double x)
        {
            return slope;
        }
    }
}
