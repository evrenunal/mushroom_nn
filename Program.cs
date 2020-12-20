using CsvHelper;
using NumpyDotNet;
using NumpyDotNet.RandomAPI;
using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;

namespace mushroomgues
{
    class Program
    {
        static void Main(string[] args)
        {
            var lines = File.ReadLines("mushrooms.csv");

            (List<int> labels, List<int[]> features) = ParseInput(lines);

            ((ndarray trainx, ndarray trainy), (ndarray testx, ndarray testy)) = SplitTable(labels, features,20);

            var rand = new np.random();
            rand.seed(1);
            long colsize = trainx.shape.iDims[1];
            var weights01 = 0.2 * rand.randn(new shape(colsize, 1)) - 0.1;
            var alpha = 0.01;


            for (int j = 0; j < 100; j++)
            {
                double error = 0;
                int correctCnt = 0;
                for (int i = 0; i < trainx.Count(); i++)
                {
                    var layer0 = (ndarray)trainx[new Slice(i, i + 1)];
                    var layer1 = np.dot(layer0, weights01);
                    var label = trainy[i];
                    var delta1 = layer1 - label;

                    error = (double)(delta1 * delta1);

                    var w01update = np.dot(layer0.T, delta1);
                    weights01 -= w01update * alpha;
                   
                    if ((double)error < 0.25) correctCnt++;

                }
                var percentCorrect = correctCnt * 100 / testx.Count();

                (double percentcorrectT, double errorT)  = Test(testx, testy, weights01);
                Console.WriteLine(
                    $"{j}: Train[ error={error},correct={percentcorrectT}%]   || Test [error={errorT},  correct={percentcorrectT}%]");
             
                alpha *= 0.99;
            }
        }

        private static (double percentcorrect,double error) Test(ndarray testx, ndarray testy, ndarray weights01)
        {
            int correctCnt = 0;
            double error=0;
            for (int i = 0; i < testx.Count(); i++)
            {
                var layer0 = (ndarray)testx[new Slice(i, i + 1)];
                var layer1 = np.dot(layer0, weights01);
                var label = testy[i];
                var delta1 = layer1 - label;
                error = (double)(delta1 * delta1);               
                if ((double)error < 0.25) correctCnt++;
            }
            var percentCorrect = correctCnt * 100 / testx.Count();
            return (percentCorrect,error) ;
        }

        private static ((ndarray trainx,ndarray trainy), (ndarray testx, ndarray testy))
            SplitTable(List<int> labels, List<int[]> features,int testPercent)
        {
            long colcnt = features[0].Length;
            long rowcnt = features.Count;

            var x = np.array(features.SelectMany(s => s).ToArray());
            x = x.reshape((rowcnt, colcnt));
           
            var y = np.array(labels.ToArray());
            long testSize = rowcnt * testPercent / 100;

            var trainx = (ndarray)x[new Slice(0, rowcnt - testSize)];
            var trainy = (ndarray)y[new Slice(0, rowcnt - testSize)];

            var testx = (ndarray)x[new Slice(rowcnt - testSize)];
            var testy = (ndarray)y[new Slice(rowcnt - testSize)];

            return ((trainx, trainy), (testx, testy));
        }

        private static (List<int> labels, List<int[]> features) ParseInput(IEnumerable<string> lines)
        {
            var label = new List<int>();
            var input = new List<string[]>();

            foreach (var line in lines)
            {

                var items = line.Split(",", StringSplitOptions.RemoveEmptyEntries).ToArray();
                label.Add(items[0]=="e"?1:0);
                items = items.Skip(1).ToArray();
                var nl = new List<string>();
                nl.AddRange(items.Take(10));
                nl.AddRange(items.Skip(11));
                input.Add(nl.ToArray());
            }

            var headers = input[0];
            var colList = headers.Select(h => new Dictionary<string, int>()).ToArray();

            foreach (var items in input.Skip(1))
            {
                for (int i = 0; i < items.Length; i++)
                {
                    if (!colList[i].ContainsKey(items[i])) colList[i].Add(items[i], 0);

                    colList[i][items[i]]++;
                }
            }

            for (int i = 0; i < colList.Length; i++)
            {
                colList[i] = colList[i].Where(c => c.Value > 10).ToDictionary(s => s.Key, s => s.Value);
            }



            var featureLst = new List<int[]>();

            foreach (var items in input.Skip(1))
            {
                var rowlst = new List<int>();
                for (int i = 0; i < colList.Length; i++)
                {
                    foreach (var key in colList[i].Keys)
                    {
                        if (key == items[i]) rowlst.Add(1);
                        else rowlst.Add(0);
                    }
                }
                featureLst.Add(rowlst.ToArray());
            }
            return (label, featureLst);
        }
    }
}
