using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    
    internal class ManagedRandom
    {
        private static Random? random;
        internal static Random getRandom()
        {
            if (random == null)
            {
                random = new Random();
                return random;
            }
            return random;
        }
    }
}
