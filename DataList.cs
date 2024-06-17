using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkRewrite2024
{
    //For data from a file
    public class DataList
    {
        //For classification problems
        public List<DataContainer> data;
        private static Random rng = new Random();
        readonly string[] classifications = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
        public DataList(string path, int points)
        {
            data = new List<DataContainer>();
            StreamReader sr = new StreamReader(path);
            string? line = sr.ReadLine();
            int index = 0;
            while (line != null && index < points)
            {
                string[] dataArray = line.Split(','); 
                double[] dataArrayDouble = new double[dataArray.Length - 1];
                //Minus 1 because the classification needs a special case
                for (int i = 0; i < dataArray.Length - 1; i++)
                {
                    double num = Convert.ToDouble(dataArray[i]);
                    dataArrayDouble[i] = num;
                }
                string classification = dataArray[4];
                int classNumber = Array.IndexOf(classifications, classification);
                DataContainer container = new DataContainer(dataArrayDouble, classNumber, classifications.Length, classification);
                data.Add(container);
                line = sr.ReadLine();
                index++;

            }
        }

        public DataContainer GetContainer(int index)
        {
            DataContainer dc = data.ElementAt(index);
            return dc;
        }
        public int GetSizeData()
        {
            return data.Count;
        }
        public void Shuffle()
        {
            int n = this.data.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                DataContainer value = data[k];
                data[k] = data[n];
                data[n] = value;
            }
        }


    }
}
