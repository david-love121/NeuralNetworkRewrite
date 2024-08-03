using NeuralNet2023;
using System;
using System.Collections.Generic;
using System.IO;
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
        //May be null, contains validation set seperate from data
        public List<DataContainer> validationData;
        public bool ValidationAvailable { get; set; } 
        readonly string[] classifications = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
        public DataList(string dataset)
        {
            if (dataset == "iris")
            {
                InitializeIrisSet();
                ValidationAvailable = false;
            }
            if (dataset == "fashion")
            {
                InitializeFashionSet();
                ValidationAvailable = true;
            }
        }
        private void InitializeFashionSet()
        {
            const string path = @"C:\Users\david\source\repos\NeuralNetworkRewrite\data\fashion\";
            data = new List<DataContainer>();
            validationData = new List<DataContainer>();
            ReadBinaryFiles(path);
            
        }
        void ReadBinaryFiles(string basePath) 
        {
            string[] validationFileNames = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"];
            string[] trainFileNames = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"];
            string[][] sets = [trainFileNames, validationFileNames];
            bool validation = false;
            foreach (string[] set in sets)
            {
                FileStream imageFileStream = new FileStream(basePath + set[0], FileMode.Open, FileAccess.Read);
                FileStream labelFileStream = new FileStream(basePath + set[1], FileMode.Open, FileAccess.Read);
                BinaryReader imageReader = new BinaryReader(imageFileStream);
                BinaryReader labelReader = new BinaryReader(labelFileStream);
                //Discard
                UInt32 magic1 = ReadAndReverseInt(imageReader);
                UInt32 numImages = ReadAndReverseInt(imageReader);
                UInt32 numRows = ReadAndReverseInt(imageReader);
                UInt32 numCol = ReadAndReverseInt(imageReader);
                //Discard
                UInt32 magic2 = ReadAndReverseInt(labelReader);
                UInt32 numLabels = ReadAndReverseInt(labelReader);
                for (int i = 0; i < numImages; i++)
                {
                    DataContainer dataContainer = new DataContainer((int)(numRows*numCol), -1, 10, "");
                    //Not worrying about dimensionality, flattening 
                    for (int k = 0; k < numRows * numCol; k++)
                    {
                        byte data = imageReader.ReadByte();
                        dataContainer.Data[k] = data;
                    }
                    dataContainer.ClassificationNumber = labelReader.ReadByte();
                    if (validation)
                    {
                        validationData.Add(dataContainer);
                    }
                    else
                    {
                        data.Add(dataContainer);

                    }
                }
                validation = true;
            }
            return;

            
        }
        //Reverses the bits of an unsigned 32 bit int, necessary because fashion set is high endian
        private static UInt32 ReverseBytes(UInt32 value)
        {
            return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
                (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
        }
        private static UInt32 ReadAndReverseInt(BinaryReader reader)
        {

                UInt32 read = reader.ReadUInt32();
                UInt32 reversed = ReverseBytes(read);
                return reversed;
           
        }

        private void InitializeIrisSet()
        {
            const string path = @"C:\Users\david\source\repos\NeuralNetworkRewrite\data\iris\iris.data";
            data = new List<DataContainer>();
            StreamReader sr = new StreamReader(path);
            string? line = sr.ReadLine();
            int index = 0;
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
                index++;

            }
        }
        public DataContainer GetTrainingContainer(int index)
        {
            DataContainer dc = data.ElementAt(index);
            return dc;
        }
        public DataContainer GetValidationContainer(int index)
        {
            if (validationData.Count == 0)
            {
                throw new Exception("validation data is null!");
            }
            DataContainer dc = validationData.ElementAt(index);
            return dc;
        }
        public int GetSizeTrainingData()
        {
            return data.Count;
        }
        public int GetSizeValidationData()
        {
            return validationData.Count;
        }
        public void Shuffle()
        {
            Random rng = ManagedRandom.getRandom();
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
