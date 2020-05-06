using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Faiss;
using System.Text;
using System.Diagnostics;

namespace FaissManagedUnitTest
{

    
    [TestClass]
    public class FaissTest 
    {

        private void DumpArray<T>(T[] dumpMe, int rows, int cols, string name="")
        {
            if(!string.IsNullOrWhiteSpace(name))
            {
                TestContext.WriteLine(name);
            }
            for(int i=0; i < rows; ++i)
            {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < cols; ++j)
                {
                    sb.Append(dumpMe[i * cols + j]);
                    sb.Append(" ");
                }
                TestContext.WriteLine(sb.ToString());
            }
        }

        public TestContext TestContext { get; set; }

        static bool FloatClose(float f1, float f2, float tol=0.001f)
        {
            return Math.Abs(f1 - f2) < tol;
        }

        [TestMethod]
        public void RenormTest()
        {
            float[] renormMe = new[] {1.0f, 2.0f, 3.0f, 4.0f};
            float norm1 = (float) Math.Sqrt(1.0 + 4.0);
            float norm2 = (float)Math.Sqrt(9.0 + 16.0);
            Utilities.RenormL2(2, 2, renormMe);


            Assert.IsTrue(FloatClose(renormMe[0], 1.0f / norm1));
            Assert.IsTrue(FloatClose(renormMe[1], 2.0f / norm1));
            Assert.IsTrue(FloatClose(renormMe[2], 3.0f / norm2));
            Assert.IsTrue(FloatClose(renormMe[3], 4.0f / norm2));
        }

        [TestMethod]
        public void IndexFlatL2Test()
        {
            // Dimension of vectors (# cols)
            int d = 3;
            // Num vectors
            int v = 5;
            // k-best to search
            int k = 3;
            var idx = new IndexFlatL2(d);
            float[] addMe = new float[d * v];
            for(int i=0; i < v; ++i)
            {
                for (int j = 0; j < d; ++j)
                {
                    addMe[i * d + j] = (float)j + (.1f * (float)i);
                }
            }
            DumpArray(addMe, v, d, "addMe");
            idx.Add(v, addMe);
            float[] dist = new float[v * k];
            Int64[] labels = new long[v * k];
            idx.Search(v, addMe, k, dist, labels);
            DumpArray(labels, v, k, "labels");
            DumpArray(dist, v, k, "distances");

            for(int i=0; i < v; ++i)
            {
                // Distance from self should be zero
                Assert.IsTrue(dist[i * k] == 0.0f);
                // Distance to anyone else should be > 0
                Assert.IsTrue(dist[(i * k) + 1] > 0.0f);
                // We should always be our own one-best
                Assert.IsTrue(labels[i * k] == i);
            }
            
            Assert.IsTrue(1 == 1);
        }
    }
}
