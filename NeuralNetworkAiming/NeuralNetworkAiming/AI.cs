using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkStuff
{
    public class WeightRandomizer
    {
        private static WeightRandomizer _rnd;
        private Random rnd = new System.Random();
        private static readonly object lockObj = new object();
        public double NextWeight { get { return rnd.NextDouble(); } }

        public static WeightRandomizer Random
        {
            get
            {
                lock (lockObj)
                {
                    if (_rnd == null) _rnd = new WeightRandomizer();
                    return _rnd;
                }
            }
        }
    }

    public class Neuron
    {
        public double bias = WeightRandomizer.Random.NextWeight;
        public double[] weights;

        public double net_sum;
        public double delta;
        public double prev_delta;
        public double bias_delta;
        public double bias_prevdelta;
        public double activation;


        public Neuron(int neuronEntriesCount)
        {
            weights = new double[neuronEntriesCount];
            RandomizeWeights();
        }

        public void RandomizeWeights()
        {
            for (int i = 0; i < weights.Length; i++)
                weights[i] = WeightRandomizer.Random.NextWeight;
        }

        public void processInput(IEnumerator input)
        {
            net_sum = 0;
            IEnumerator neuronInput = input;
            int i = 0; double inputValue = 0;
            while (neuronInput.MoveNext())
            {
                inputValue = neuronInput.Current.GetType() == typeof(Neuron) ? ((Neuron)neuronInput.Current).activation : (double)neuronInput.Current;
                net_sum += weights[i++] * inputValue;
            }
            activation = activationFunc(net_sum + bias);
            //activation = activationFunc(net_sum);
        }

        // using sigmoid as activation function
        private double activationFunc(double x)
        {
            //if (x < -45.0) return 0.0;
            //else if (x > 45.0) return 1.0;
            //double result = 1f / (1 + (double)Math.Pow(Math.E, -x));
            //return result;
            return Math.Tanh(x);
        }
    }

    public class NeuronLayer : List<Neuron>
    {
        public NeuronLayer(int neuronEntriesCount, int capacity) : base(capacity)
        {
            for (int i = 0; i < capacity; i++)
                Add(new Neuron(neuronEntriesCount));
        }
    }

    public class NeuralNetwork
    {
        private double _learningRate;
        private double _momentum;
        private double _decay;

        public double LearningRate { get { return _learningRate; } }
        public double Momentum { get { return _momentum; } }
        public double Decay { get { return _decay; } }


        List<NeuronLayer> neuronLayers;
        public double mse = 0;

        public NeuralNetwork(int[] LayersMap, double LearningRate, double Momentum, double Decay)
        {
            _learningRate = LearningRate;
            _momentum = Momentum;
            _decay = Decay;
            neuronLayers = new List<NeuronLayer>();
            for (int i = 1; i < LayersMap.Length; i++)
            {
                int NeuronsInLayer = LayersMap[i];
                neuronLayers.Add(new NeuronLayer(LayersMap[i-1], NeuronsInLayer));
            }
        }

        /// <summary>
        /// Processing entries
        /// </summary>
        /// <param name="Entries">entry items collection</param>
        /// <returns></returns>
        public NeuronLayer FeedForward(IEnumerator Entries)
        {
            foreach (NeuronLayer layer in neuronLayers)
            {
                foreach (Neuron neuron in layer)
                {
                    neuron.processInput(Entries);
                    //reset position to initial state;
                    Entries.Reset();
                }
                Entries = layer.GetEnumerator();
            }

            return neuronLayers.Last<NeuronLayer>();
        }

        public double ErorrCost(double[] ExpectedResult)
        {
            double errorCostValue = 0;
            NeuronLayer outputLayer = neuronLayers.Last<NeuronLayer>();
            for (int i = 0; i < ExpectedResult.Length; i++)
            {
                errorCostValue += Math.Pow(outputLayer[i].activation - ExpectedResult[i],2);
            }
            return errorCostValue;
        }

        public void Train(double[][] LearningData, double[][] ExpectedResults, int MaxEpoches, float MSE)
        {
            var curMSE = 0d;
            var epoch = 0;
            do
            {
                for (int i = 0; i < LearningData.Length; i++)
                {
                    var inputs = LearningData[i].GetEnumerator();
                    FeedForward(inputs);
                    curMSE += ErorrCost(ExpectedResults[i]);
                    BackProp(ExpectedResults[i]);
                    UpdateWeights(inputs);
                }
                epoch++;
                curMSE = curMSE / LearningData.Length;
                if(epoch % 100 == 0)
                    Console.WriteLine(String.Format("Epoch #{0};\t mse={1}",epoch, Math.Round(curMSE, 2)));
            } while (epoch < MaxEpoches && curMSE > MSE);
            Console.WriteLine(String.Format("Training result:\t epoch #{0};\t mse={1}", epoch, Math.Round(curMSE, 2)));
        }

        private void UpdateWeights(IEnumerator Inputs)
        {
            for (int i = 0; i < neuronLayers.Count; i++)
            {
                NeuronLayer processingLayer = neuronLayers[i];
                double entry;
                int j = 0;
                while (Inputs.MoveNext())
                {
                    entry = Inputs.Current.GetType() == typeof(Neuron) ? ((Neuron)Inputs.Current).activation : (double)Inputs.Current;
                    foreach (Neuron neuron in processingLayer)
                    {
                        neuron.weights[j] += _learningRate * neuron.delta * entry;
                        neuron.weights[j] += _momentum * neuron.prev_delta;
                        neuron.weights[j] -= _decay * neuron.weights[j];

                        neuron.bias += _learningRate * neuron.bias_delta * 1;
                        neuron.bias += _momentum * neuron.bias_prevdelta;
                        neuron.bias -= _decay * neuron.bias;
                        neuron.bias_prevdelta = _learningRate * neuron.bias_delta * 1;
                    }
                    j++;
                }
                Inputs = processingLayer.GetEnumerator();
            }
        }

        /// <summary>
        /// Backward - Propagation 
        /// </summary>
        /// <param name="ExpectedResult">Expected output of Neural network</param>
        private void BackProp(double[] ExpectedResult)
        {   
            for (int i = 0; i < ExpectedResult.Length; i++)
            {
                Neuron currNeuron = neuronLayers.Last<NeuronLayer>()[i];
                currNeuron.delta = (1 + currNeuron.activation) * (1 - currNeuron.activation) * (ExpectedResult[i] - currNeuron.activation);
            }

            // -2 because the last one processed above
            for (int i = neuronLayers.Count - 2; i >= 0; i--)
            {
                NeuronLayer processingLayer = neuronLayers[i];
                Neuron processingNeuron;
                NeuronLayer backLayer = neuronLayers[i + 1];

                for (int j = 0; j < processingLayer.Count; j++)
                {
                    processingNeuron = processingLayer[j];
                    double total = 0;
                    foreach (Neuron backLayerNeuron in backLayer)
                        total += backLayerNeuron.delta * backLayerNeuron.weights[j];
                    processingNeuron.delta = (1 + processingNeuron.activation) * (1 - processingNeuron.activation) * total;
                    processingNeuron.bias_delta = (1 + processingNeuron.activation) * (1 - processingNeuron.activation) * 1;
                }
            }
        }

        public void RandomizeWeights()
        {
            foreach (NeuronLayer layer in neuronLayers)
                foreach (Neuron neuron in layer)
                    neuron.RandomizeWeights();
        }
    }

}
