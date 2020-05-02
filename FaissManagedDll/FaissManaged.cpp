#define NOMINMAX // Windows max macro collides with std::
#include <Windows.h>
#include <memory>
#include "IndexFlat.h"
#include <vcclr.h>
#include <msclr/marshal_cppstd.h>
#using <System.dll>

using namespace System;
using namespace msclr::interop;

// A wrapper around Faiss that lets you build indexes
// Right now just proof-of-concept code to makes sure it all works from C#,
// eventually may want to rework the interface, or possibly look at extending
// FaissSharp to support the windows dll 

namespace Faiss {
    public ref class IndexFlatL2
    {
    public:
        IndexFlatL2() { m_pWrapper = new faiss::IndexFlatL2(); }
        IndexFlatL2(int d) { m_pWrapper = new faiss::IndexFlatL2(d); }
        ~IndexFlatL2() { delete m_pWrapper; m_pWrapper = nullptr; }

        void Add(int n, array<float>^ addMe)
        {
            pin_ptr<float> pinnedArray = &addMe[0];
            m_pWrapper->add((int64_t)n, pinnedArray);
        }

        void Search(int n, array<float>^ x, int k,  array<float>^ distances, array<int64_t>^ labels)
        {
            pin_ptr<float> pinnedX = &x[0];
            pin_ptr<float> pinnedDist = &distances[0];
            pin_ptr<int64_t> pinnedLabels = &labels[0];
            m_pWrapper->search(n, pinnedX, (int64_t)k, pinnedDist, pinnedLabels);
        }
        

    protected:
        !IndexFlatL2() { delete m_pWrapper; m_pWrapper = nullptr; }

    private:
        // Review: I'm not using e.g. unique_ptr here because I don't know the lifetime  
        // semantics behind ref classes. 
        faiss::IndexFlatL2* m_pWrapper;
    };

}
