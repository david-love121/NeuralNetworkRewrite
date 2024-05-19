using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    public class LinearFunction : Function
    {
        //y = mx + b
        public double yIntercept { get; set; }
        public double slope { get; set; }
        [JsonConstructor]
        public LinearFunction(double yIntercept, double slope)
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
        public override string ToString()
        {
            string s = this.GetType().ToString() + "|" + yIntercept + "|" + slope;
            return s;
        }
    }
}
