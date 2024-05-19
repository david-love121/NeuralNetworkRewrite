using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    public class ExponentialFunction : Function
    {
        //y = b^x
        public double baseValue { get; set; }
        [JsonConstructor]
        public ExponentialFunction(double baseValue)
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
        public override string ToString()
        {
            return this.GetType().ToString() + "|" + this.baseValue.ToString();
        }
    }
}
