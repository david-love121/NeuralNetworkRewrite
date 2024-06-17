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
        public double? UpperBound { get; private set; }
        public double? LowerBound { get; private set; }
        [JsonConstructor]
        public ExponentialFunction(double baseValue)
        {
            this.baseValue = baseValue;
        }
        public ExponentialFunction(double baseValue, double lowerBound, double upperBound)
        {
            this.baseValue = baseValue;
            this.LowerBound = lowerBound;
            this.UpperBound = upperBound;
        }

        internal override double Compute(double x)
        {
            double result = Math.Pow(baseValue, x); 
            if (UpperBound is not null && LowerBound is not null)
            {
                
                if (result > UpperBound)
                {
                    return (double)UpperBound;
                } else if (result < LowerBound)
                {
                    return (double)LowerBound;
                } else
                {
                    return result;
                }
            } else
            {
                return result;
            }
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
