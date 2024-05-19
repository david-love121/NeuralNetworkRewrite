using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    [JsonDerivedType(typeof(LinearFunction), nameof(LinearFunction))]
    [JsonDerivedType(typeof(ExponentialFunction), nameof(ExponentialFunction))]
    [JsonDerivedType(typeof(MSEFunction), nameof(MSEFunction))]
    public abstract class Function
    {
        internal abstract double Compute(double x);
        internal abstract double ComputeDerivative(double x);
    }
}
