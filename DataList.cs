﻿using System;
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
        readonly string[] classifications = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
        public DataList(string path)
        {
            data = new List<DataContainer>();
            StreamReader sr = new StreamReader(path);
            string? line = sr.ReadLine();
            while (line != null)
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
    }
}
