using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetworkStuff;

namespace NeuralNetworkTest
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] expectedResult;
            int[] middleLayerMap = new int[2] {3,2};
            NeuralNetwork NN = new NeuralNetwork(2, 1, middleLayerMap);
            double[] entries;
            double[] entries1 = new double[2] { 0.2, 0.4};
            double[] entries2 = new double[2] { 0.1, 0.1 };

            NeuronLayer result;
            result = NN.ProcessData(entries1.GetEnumerator());
            Console.WriteLine(result[0].value);

            expectedResult = new double[1] { 0.2 };
            for (int i = 0; i < 80; i++)
            {
                Console.WriteLine(string.Format("Result {1} for 1 = {0}", Math.Round(NN.ProcessData(entries1.GetEnumerator())[0].value, 15), i));
                NN.Learn(entries1, expectedResult);
            }


            result = NN.ProcessData(entries1.GetEnumerator());
            Console.WriteLine(result[0].value);
            //result = NN.ProcessData(entries2.GetEnumerator());
            //Console.WriteLine(result[0].value);
            Console.Read();
        }
    }
}
