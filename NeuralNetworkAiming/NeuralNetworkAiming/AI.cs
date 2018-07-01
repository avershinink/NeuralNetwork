using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AimingAI
{
    public class Neuron
    {
        public readonly float beta = (float)0.05;
        private float weight;
        private float sum;
        private float delta;
        private float newGradient;
        private float OldGradient;
        private float newStep;
        private float oldStep = 0.5f;
        Random rnd;
        public Neuron()
        {
            weight = rnd.Next(-10, 10);
        }
    }

    public class NeuronLayer : List<Neuron>
    {
        public NeuronLayer(int capacity) : base(capacity)
        {
        }
    }
    public class NeuralNetwork
    {
        List<NeuronLayer> neuronLayers;
        public NeuralNetwork(int EntryCount, int ExitCount, int MiddlyLayersCount = 1)
        {
            neuronLayers = new List<NeuronLayer>();
            neuronLayers.Add(new NeuronLayer(EntryCount));
            for (int i = 0; i < MiddlyLayersCount; i++)
                neuronLayers.Add(new NeuronLayer(EntryCount * i));
            neuronLayers.Add(new NeuronLayer(ExitCount));
        }
    }
}
