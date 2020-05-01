using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Faiss;
using System.Text;

namespace FaissManagedUnitTest
{

    
    [TestClass]
    public class FaissTest 
    {

        private void DumpArray(float[] dumpMe, int rows, int cols, string name="")
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
        [TestMethod]
        public void IndexFlatL2Test()
        {
            int d = 3;
            int v = 2;
            int k = 2;
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
            DumpArray(dist, v, k, "distances");
            
            Assert.IsTrue(1 == 1);
        }
    }
}
