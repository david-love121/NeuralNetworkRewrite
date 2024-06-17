using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using MathNet.Numerics.LinearAlgebra;
using System.Text.Json.Serialization;
using System.Runtime.InteropServices.Marshalling;
namespace NeuralNetworkRewrite2024
{
    internal class NeuralNetwork
    {
        List<Layer> layers;
        private Random rand;
        internal bool Categorical { get; set; }
        //Including input and output layers
        internal NeuralNetwork(int[] layerSizes, Function activationFunction, bool lastLayerNonlinearity, bool categorical, double bias = 1)
        {
            Categorical = categorical;
            rand = new Random();
            layers = new List<Layer>();
            PopulateLayers(ref layers, layerSizes, activationFunction, lastLayerNonlinearity, bias);
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
        
        static void PopulateLayers(ref List<Layer> layers, int[] layerSizes, Function activationFunction, bool lastLayerNonlinearity, double bias)
        {
            for (int i = 0; i < layerSizes.Length; i++)
            {
                Layer layer = new Layer(layerSizes[i], activationFunction, i, bias);
                if (!lastLayerNonlinearity && (i == layerSizes.Length - 1))
                {
                    layer = new Layer(layerSizes[i], new LinearFunction(0, 1), i, bias);
                }
                
                layers.Add(layer);
            }
        }
        void ConnectNeuronsAndLayers()
        {
            int numInputs = layers[0].size;
            int numOutputs = layers[^1].size;
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Layer selectedLayer = layers[i];
                Layer nextLayer = layers[i + 1];
                selectedLayer.ConnectLayer(nextLayer);
                for (int k = 0; k < selectedLayer.GetSize(); k++)
                {
                    for (int j = 0; j < nextLayer.GetSize(); j++)
                    {
                        double weight = CalculateGloartUniformWeight(numInputs, numOutputs);
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
        private double CalculateGloartUniformWeight(int numInputs, int numOutputs)
        {
            double multipler = (rand.NextDouble() * 2) - 1;
            double x = 6.0 / (double)(numInputs + numOutputs);
            x = Math.Sqrt(x);
            return multipler * x;
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
            if (Categorical)
            {
                //Converts to probability distribution for categorical problems
                SoftmaxFunction softmaxFunction = new SoftmaxFunction();
                vectorizedOutput = softmaxFunction.Compute(vectorizedOutput);
            }
            return vectorizedOutput;
            
        }
        internal Vector<double> RunNetwork(Vector<double> inputs)
        {
            layers[0].RunNeurons(inputs);
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].RunNeurons();
            }
            Layer outputLayer = layers[layers.Count - 1];
            Vector<double> vectorizedOutput = outputLayer.OutputLayerAsVector();
            if (Categorical)
            {
                //Converts to probability distribution for categorical problems
                SoftmaxFunction softmaxFunction = new SoftmaxFunction();
                vectorizedOutput = Driver.ApplyClipping(vectorizedOutput, 500);
                vectorizedOutput = softmaxFunction.Compute(vectorizedOutput);
            }
            return vectorizedOutput;

        }
        internal Vector<double> ScoreOutput(Vector<double> output, Vector<double> expectedOutput)
        {
            
            if (Categorical)
            {
                if (output.Count != expectedOutput.Count)
                {
                    throw new ArgumentException("Vector dimensionality must match!");
                }
                //Todo: CHeck if negative log
                //Getting value higher than 1 here, check
                output = output.PointwiseLog();
                output = -output.PointwiseMultiply(expectedOutput);
                return output;
            }
            MSEFunction mseFunction = new MSEFunction();
            Vector<double> differenceVector = output - expectedOutput;
            Vector<double> MSEVector = mseFunction.Compute(differenceVector);
            double sum = MSEVector.Sum();
            double average = sum / MSEVector.Count;
            Vector<double> outputVector = Vector<double>.Build.Dense(1, average);
            return outputVector;
        }

        internal void RandomizeWeights(double range)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].RandomizeWeights(range);   
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
        internal List<Vector<double>> GetBiasVectorList()
        {
            List<Vector<double>> result = new List<Vector<double>>();
            for (int i = 0; i < layers.Count; i++)
            {
                Layer currentLayer = layers[i];
                result.Add(currentLayer.GetBiasVector());
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
        
        internal void SetBiasToList(List<Vector<double>> biases)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                Vector<double> currentChanges = biases[i];
                for (int j = 0; j < currentChanges.Count; j++)
                {
                    layers[i].ChangeBias(currentChanges);
                }
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
