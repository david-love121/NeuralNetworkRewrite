using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    internal class DataGenerator
    {
        public List<double> Data;
        private Function function;
        private int totalPoints;
        private double step;
        public DataGenerator(Function function, int totalPoints, double step = 1) 
        {
            this.function = function;
            this.totalPoints = totalPoints;
            this.step = step;
            Data = new List<double>();
            PopulateData(ref Data);
        }
        void PopulateData(ref List<double> dataList)
        {
            for (double i = 0; i < totalPoints; i += step)
            {
                double y = function.Compute(i);
                dataList.Add(y);
            }
            return;
        }
        internal double GetStep()
        {
            return step;
        }
        internal int GetSizeData()
        {
            return totalPoints;
        }
        internal double GetDataPoint(int index)
        {
            return Data[index];
        }
    }
}
