using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AimingAI;

namespace NeuralNetworkAiming
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork NN = new NeuralNetwork(4, 2, 2);
            float[] entries = new float[4] { 1, 2, 3, 4 };

            NN.ProcessData(entries, ((beta, sum) => { return (2f / (1 + (float)Math.Pow(Math.E, -beta * sum))) - 1; }));
        }
    }
}
