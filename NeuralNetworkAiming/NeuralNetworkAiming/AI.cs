using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkStuff
{
    public sealed class WeightRandomizer
    {
        private static WeightRandomizer _rnd;
        private Random rnd = new System.Random();
        private static readonly object lockObj = new object();
        public double NextDouble { get { return rnd.NextDouble() - rnd.NextDouble(); } }
        public int NextInt { get { return rnd.Next(0,100) - rnd.Next(0,100); } }

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
        public WeightRandomizer()
        {
        }
    }

    public class Neuron
    {
        public readonly double beta = (double)0.05;
        public double[] weights;

        public double net_sum;
        public double delta;
        public double[] newGradient;
        public double[] OldGradient;
        public double[] newStep;
        public double[] oldStep ;
        public double value;

        public Neuron(int neuronEntriesCount)
        {
            newGradient = new double[neuronEntriesCount];
            OldGradient = new double[neuronEntriesCount];
            newStep = new double[neuronEntriesCount];
            oldStep = new double[neuronEntriesCount];

            for (int i = 0; i < oldStep.Length; i++)
                oldStep[i] = 0.1f;
            weights = new double[neuronEntriesCount];
            double w;
            for (int i = 0; i < neuronEntriesCount; i++)
            {
                w = WeightRandomizer.Random.NextInt;
                weights[i] = w;
            }
        }

        public void processInput(IEnumerator input)
        {
            net_sum = 0;
            int i = 0; double inputValue = 0;
            while (input.MoveNext())
            {
                inputValue = input.Current.GetType() == typeof(Neuron) ? ((Neuron)input.Current).value : (double)input.Current;
                net_sum += weights[i++] * inputValue;
            }  
            value = activationFunc(net_sum);
        }

        // using bipolar sigmoid as activation function
        private double activationFunc(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            double result = ( 2f / (1 + (double)Math.Pow(Math.E, -beta * x)) ) - 1;
            return result;
        }

        private double deriv_activeFunc()
        {
            double result = (beta / 2) * (1 + net_sum) * (1 - net_sum);
            return result;
        }

        public void deltaCalc(double multiplier)
        {
            delta = (double)(deriv_activeFunc() * multiplier);
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
        List<NeuronLayer> neuronLayers;

        public NeuralNetwork(int EntryCount, int ExitCount, int[] MiddlyLayersMap)
        {
            neuronLayers = new List<NeuronLayer>();
            neuronLayers.Add(new NeuronLayer(EntryCount,EntryCount));
            int prevLayerNeuronsCount = EntryCount;
            for (int i = 0; i < MiddlyLayersMap.Length; i++)
            {
                int NeuronsInLayer = MiddlyLayersMap[i];
                neuronLayers.Add(new NeuronLayer(prevLayerNeuronsCount, NeuronsInLayer));
                prevLayerNeuronsCount = NeuronsInLayer;
            }
            if(ExitCount>0)
               neuronLayers.Add(new NeuronLayer(prevLayerNeuronsCount,ExitCount));
        }

        /// <summary>
        /// Processing entries
        /// </summary>
        /// <param name="Entries">entry items collection</param>
        /// <returns></returns>
        public NeuronLayer ProcessData(IEnumerator Entries)
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

        public void Learn(double[] LearningEntry, double[] ExpectedResult)
        {
          // Backward - Propagation 
            for (int i = 0; i < ExpectedResult.Length; i++)
            {
                Neuron currNeuron = neuronLayers.Last<NeuronLayer>()[i];
                currNeuron.deltaCalc(ExpectedResult[i] - currNeuron.value);
            }

            // -2 because lasone processed above.
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
                    processingNeuron.deltaCalc(total);
                }
            }
            // END Backward-Propagation 


            IEnumerator Entries = LearningEntry.GetEnumerator();
            for (int i = 0; i < neuronLayers.Count; i++)
            {
                NeuronLayer processingLayer = neuronLayers[i];
                double entry;
                foreach (Neuron neuron in processingLayer)
                {
                    int j = 0;
                    while (Entries.MoveNext())
                    {
                        entry = Entries.Current.GetType() == typeof(Neuron) ? ((Neuron)Entries.Current).net_sum : (double)Entries.Current;
                        neuron.newGradient[j] = (-1) * neuron.delta * entry;
                        if (neuron.OldGradient[j] - neuron.newGradient[j] != 0)
                            neuron.newStep[j] = neuron.newGradient[j] / (neuron.OldGradient[j] - neuron.newGradient[j]) * neuron.oldStep[j];
                        else
                        {
                            neuron.newStep[j] = 0.00000001f;
                            Console.WriteLine("BAM");
                        }
                        neuron.weights[j] += neuron.newStep[j];
                        neuron.oldStep[j] = neuron.newStep[j];
                        neuron.OldGradient[j] = neuron.newGradient[j];
                        j++;
                    }
                }
                Entries = processingLayer.GetEnumerator();
            }

        }
    }

}
