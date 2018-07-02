using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AimingAI
{
    public class Neuron
    {
        public readonly float beta = (float)0.05;
        public float[] weight;

        public float sum;
        public float delta;
        public float[] newGradient;
        public float[] OldGradient;
        public float[] newStep;
        public float[] oldStep ;
        Random rnd = new Random();
        public float value;

        public Neuron(int neuronEntriesCount)
        {
            oldStep = new float[neuronEntriesCount];
            for (int i = 0; i < oldStep.Length; i++)
                oldStep[i] = 0.5f;
            weight = new float[neuronEntriesCount];
            float w;
            for (int i = 0; i < neuronEntriesCount; i++)
            {
                w = rnd.Next(-10, 10);
                weight[i] = w != 0 ? w : rnd.Next(1, 10);
            }
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

        public NeuralNetwork(int EntryCount, int ExitCount, int MiddlyLayersCount = 1)
        {
            neuronLayers = new List<NeuronLayer>();
            neuronLayers.Add(new NeuronLayer(EntryCount,EntryCount));
            int prevLayerNeuronsCount = EntryCount;
            for (int i = 0; i < MiddlyLayersCount; i++)
            {
                neuronLayers.Add(new NeuronLayer(prevLayerNeuronsCount, EntryCount * (i + 1)));
                prevLayerNeuronsCount = EntryCount * (i + 1);
            }
            neuronLayers.Add(new NeuronLayer(prevLayerNeuronsCount,ExitCount));
        }

        /// <summary>
        /// Processing float entries
        /// </summary>
        /// <param name="Entries">Float type entry items collection</param>
        /// <returns></returns>
        public NeuronLayer ProcessData(float[] Entries, Func<float,float,float> func)
        {

            for (int i = 0; i < Entries.Length; i++)
                foreach (Neuron neuron in neuronLayers[0])
                {
                    neuron.sum += neuron.weight[i] * Entries[i];
                    neuron.value = func(neuron.beta, neuron.sum);
                }
            for(int layerNum = 1; layerNum<neuronLayers.Count;layerNum++)
            {
                NeuronLayer prevLayer = neuronLayers[layerNum - 1];
                for (int prevLayerNeuronIndex = 0; prevLayerNeuronIndex < prevLayer.Count; prevLayerNeuronIndex++)
                {
                    foreach (Neuron neuron in neuronLayers[layerNum])
                    {
                        neuron.sum += neuron.weight[prevLayerNeuronIndex] * prevLayer[prevLayerNeuronIndex].sum;
                        neuron.value = func(neuron.beta, neuron.sum);
                    }
                }
            }
            return neuronLayers.Last<NeuronLayer>();
        }

        public void Learn(float[] LearningEntry, float[] ExpectedResult)
        {

        }
    }

}
