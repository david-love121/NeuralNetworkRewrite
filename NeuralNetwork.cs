using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using MathNet.Numerics.LinearAlgebra;
using System.Text.Json.Serialization;
namespace NeuralNetworkRewrite2024
{
    internal class NeuralNetwork
    {
        List<Layer> layers;
        
        //Including input and output layers
        internal NeuralNetwork(int[] layerSizes, Function activationFunction, double bias = 1)
        {
            layers = new List<Layer>();
            PopulateLayers(ref layers, layerSizes, activationFunction, bias);
            ConnectNeuronsAndLayers();
        }
        //Construct from storage
        internal NeuralNetwork(NeuralNetworkMetadata metadata) 
        {
            layers = new List<Layer>();
            for (int i = 0; i < metadata.LayerData.Count; i++)
            {
                layers.Add(new Layer(metadata.LayerData[i]));
            }
            ConnectNeuronsAndLayers();
        }
        
        static void PopulateLayers(ref List<Layer> layers, int[] layerSizes, Function activationFunction, double bias)
        {
            for (int i = 0; i < layerSizes.Length; i++)
            {
                Layer layer = new Layer(layerSizes[i], activationFunction, bias);
                layers.Add(layer);
            }
        }
        void ConnectNeuronsAndLayers()
        {
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Layer selectedLayer = layers[i];
                Layer nextLayer = layers[i + 1];
                selectedLayer.ConnectLayer(nextLayer);
                for (int k = 0; k < selectedLayer.GetSize(); k++)
                {
                    for (int j = 0; j < nextLayer.GetSize(); j++)
                    {
                        double weight = 1;
                        if (selectedLayer.FromStorage && selectedLayer.PresetWeights is not null)
                        {
                            weight = selectedLayer.PresetWeights[k][j];
                        }
                        Connector connector = new Connector(selectedLayer.GetNeuron(k), nextLayer.GetNeuron(j), weight);
                        selectedLayer.GetNeuron(k).AddConnectionOut(connector);
                    }
                }
            }
        }
        internal Vector<double> RunNetwork(double input)
        {
            layers[0].RunNeurons(input);
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].RunNeurons();
            }
            Layer outputLayer = layers[layers.Count - 1];
            Vector<double> vectorizedOutput = outputLayer.OutputLayerAsVector();
            return vectorizedOutput;
            
        }
        internal double ScoreOutput(Vector<double> input, Vector<double> expectedOutput)
        {
            MSEFunction mseFunction = new MSEFunction();
            Vector<double> differenceVector = input - expectedOutput;
            Vector<double> MSEVector = mseFunction.Compute(differenceVector);
            double sum = MSEVector.Sum();
            double average = sum / MSEVector.Count;
            return average;
        }
        internal void RandomizeWeights()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].RandomizeWeights();   
            }
        }
        internal List<Matrix<double>> GetWeightsMatrixList()
        {
            List<Matrix<double>> result = new List<Matrix<double>>();
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Matrix<double> nextMatrix = layers[i].WeightsAsMatrix();
                result.Add(nextMatrix);
            }
            return result;
        }
        internal Vector<double> GetBiasVector()
        {
            Vector<double> result = Vector<double>.Build.Dense(layers.Count);
            for (int i = 0; i < layers.Count; i++)
            {
                result[i] = layers[i].GetBias();
            }
            return result;
        }
        internal void SetWeightsToList(List<Matrix<double>> list)
        {
            int total = list.Count;
            for (int i = 0; i < total; i++)
            {
                
                Layer currentLayer = layers[i];
                Matrix<double> currentMatrix = list[i];
                currentLayer.ChangeWeights(currentMatrix);
            }
        }
        
        internal void SetBiasToList(Vector<double> biases)
        {
            for (int i = 0; i < biases.Count; i++)
            {
                layers[i].ChangeBias(biases[i]);
            }
        }
        internal Layer GetLayer(int index)
        {
            return layers[index];
        }
        internal List<Layer> GetLayers()
        {
            return layers;
        }
        internal void SaveNetworkToStorage(string path)
        {
            NeuralNetworkMetadata metadata = new NeuralNetworkMetadata(this);
            string data = metadata.Serialize();
            File.WriteAllText(path, data);
        }
        internal static NeuralNetwork LoadNetworkFromStorage(string path)
        {
            string data = File.ReadAllText(path);
            NeuralNetworkMetadata newNetworkData = JsonSerializer.Deserialize<NeuralNetworkMetadata>(data);
            NeuralNetwork newNetwork = new NeuralNetwork(newNetworkData);
            return newNetwork; 
        }
    }
    internal class NeuralNetworkMetadata
    {
        [JsonInclude]
        internal List<LayerMetadata> LayerData { get; set; }
        public NeuralNetworkMetadata(NeuralNetwork network)
        {
            LayerData = new List<LayerMetadata>();
            for (int i = 0; i < network.GetLayers().Count; i++)
            {
                LayerData.Add(new LayerMetadata(network.GetLayer(i)));
            }
        }
        [JsonConstructor]
        public NeuralNetworkMetadata(List<LayerMetadata> layerData)
        {
            LayerData = layerData;
        }

        internal string Serialize()
        {
            string data = "";
            try
            {
                data = JsonSerializer.Serialize(this);
            } catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            return data;
        }
    }
}
