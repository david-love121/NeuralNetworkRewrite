using System;
using System.Collections.Generic;
using System.Linq;

using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    public class DataContainer
    {
        public double[] Data { get; set; } 
        public int ClassificationNumber { get; set; }
        
        public string ClassificationString { get; set; }
        public int NumCategories { get; private set; }
        public DataContainer(double[] data, int classificationNumber, int numCategories, string classificationString) 
        { 
            Data = data;
            ClassificationNumber = classificationNumber;
            ClassificationString = classificationString;
            this.NumCategories = numCategories;
        }
        public override string ToString()
        {
            return ClassificationString;
        }
        public double[] GetFeatures()
        {
            return Data;
        }
        public int GetClassificationNumber()
        {
            return ClassificationNumber;
        }
        public double[] GetOneHotCoded()
        {
            double[] result = new double[NumCategories];
            for (int i = 0; i < NumCategories; i++)
            {
                if (i == ClassificationNumber)
                {
                    result[i] = 1;
                } else
                {
                    result[i] = 0;
                }
            }
            return result;
        }
        public string GetClassificationString()
        {
            return ClassificationString;
        }
    }
}
